"""Shared-memory ring, analysis and backpressure tests -- no hardware needed."""

from __future__ import annotations

import numpy as np
import pytest

from sniffer.analysis import (
    Baseline,
    one_dimensional_split,
    EnrollmentProgress,
    InterferenceMonitor,
    LiveAnalyzer,
    RunningStats,
    StreamingKMeans,
)
from sniffer.features import FEATURE_VECTOR_KEYS, Measurement, PacketFeatures
from sniffer.packet import PacketRecord
from sniffer.shmring import SharedRing


# --------------------------------------------------------------------------
# shared-memory ring
# --------------------------------------------------------------------------

@pytest.fixture
def ring():
    r = SharedRing(capacity=8, block_size=1024, n_channels=1, sample_rate=8e6)
    yield r
    r.close()


def _fill(ring, index, value, timestamp_valid=1):
    slot = ring.slot_view(index)
    slot[:] = value
    ring.publish(index, n_samples=ring.block_size, timestamp=index * ring.block_size,
                 wall_time=0.0, epoch=0, gain_db=40, channel=37,
                 temperature_c=30.0, peak=0.1, frequency_hz=2.402e9, calibrated=0,
                 timestamp_valid=timestamp_valid)


def test_ring_roundtrip(ring):
    _fill(ring, 0, 7)
    got = ring.read(0)
    assert got is not None
    raw, meta = got
    assert raw.size == ring.block_size * 2
    assert int(meta["n_samples"]) == ring.block_size
    assert np.all(raw == 7)


def test_ring_cursor_advances_only_on_publish(ring):
    assert ring.written() == 0
    ring.slot_view(0)[:] = 1
    assert ring.written() == 0, "slot_view must not publish"
    _fill(ring, 0, 1)
    assert ring.written() == 1


def test_ring_reports_lapped_slots_rather_than_tearing(ring):
    for i in range(ring.capacity + 3):
        _fill(ring, i, i % 100)
    # the three oldest have been overwritten and must be refused
    assert ring.read(0) is None
    assert ring.read(2) is None
    assert ring.read(3) is not None
    assert ring.read(ring.capacity + 2) is not None


def test_ring_read_ahead_returns_none(ring):
    _fill(ring, 0, 1)
    assert ring.read(1) is None


def test_ring_sample_addressing_spans_blocks(ring):
    for i in range(4):
        slot = ring.slot_view(i)
        slot[0::2] = i        # I
        slot[1::2] = -i       # Q
        ring.publish(i, n_samples=ring.block_size, timestamp=i * ring.block_size,
                     wall_time=0.0, epoch=0, gain_db=40, channel=37,
                     temperature_c=30.0, peak=0.1, frequency_hz=2.402e9,
                     calibrated=0, timestamp_valid=1)
    bs = ring.block_size
    out = ring.read_samples(bs - 4, 8)
    assert out is not None and out.size == 8
    # samples 1020..1023 come from block 0 (I=0), 1024..1027 from block 1 (I=1)
    assert np.allclose(out[:4].real, 0.0)
    assert np.allclose(out[4:].real, 1 / 2048.0)
    assert np.allclose(out[4:].imag, -1 / 2048.0)


def test_ring_sample_addressing_refuses_aged_out(ring):
    for i in range(ring.capacity + 4):
        _fill(ring, i, 1)
    assert ring.read_samples(0, 16) is None


def test_verified_read_refuses_a_lapped_slot(ring):
    """The reader must check after copying, not only before.

    A writer that comes all the way round during the copy hands back a slot
    that is part old and part new; its timestamp then looks like an enormous
    jump in the radio's sample counter, and the receiver gets blamed for the
    host being slow.
    """
    for i in range(ring.capacity + 2):
        _fill(ring, i, i % 100)
    assert ring.read_verified(0) is None
    assert ring.read_verified(ring.capacity + 1) is not None


def test_verified_read_returns_a_snapshot_not_a_live_view(ring):
    _fill(ring, 0, 5)
    got = ring.read_verified(0)
    assert got is not None
    data, meta = got
    ts_before = int(meta["timestamp"])
    # overwrite the same slot; the snapshot must not change underneath us
    _fill(ring, ring.capacity, 9)
    assert np.all(data == 5), "sample data was a live view"
    assert int(meta["timestamp"]) == ts_before, "metadata was a live view"


def test_verified_read_rejects_an_implausible_length(ring):
    _fill(ring, 0, 1)
    ring.meta[0]["n_samples"] = ring.block_size * 4
    assert ring.read_verified(0) is None


def test_ring_carries_the_timestamp_validity_flag(ring):
    """A MIMO stream has no FPGA timestamp; the reader must be told.

    Without the flag the running sample counter used in its place would look
    like a perfectly continuous hardware clock and the loss accounting would
    report a fabricated 0%.
    """
    _fill(ring, 0, 1, timestamp_valid=1)
    _fill(ring, 1, 1, timestamp_valid=0)
    _, m0 = ring.read_verified(0)
    _, m1 = ring.read_verified(1)
    assert int(m0["timestamp_valid"]) == 1
    assert int(m1["timestamp_valid"]) == 0


def test_ring_spec_roundtrip(ring):
    spec = ring.spec
    other = SharedRing(0, 0, create=False, spec=spec)
    try:
        _fill(ring, 0, 42)
        other.cursor = ring.cursor
        raw, _ = other.read(0)
        assert np.all(raw == 42), "second attachment must see the writer's data"
    finally:
        other.close()


# --------------------------------------------------------------------------
# streaming statistics and clustering
# --------------------------------------------------------------------------

def test_running_stats_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, (500, 4))
    st = RunningStats(4)
    for row in data:
        st.update(row)
    assert np.allclose(st.mean, data.mean(axis=0), atol=1e-9)
    assert np.allclose(st.std, data.std(axis=0, ddof=1), atol=1e-9)


def test_running_stats_ignores_nan_per_dimension():
    st = RunningStats(2)
    for v in ([1.0, np.nan], [3.0, 5.0], [5.0, 7.0]):
        st.update(np.array(v))
    assert np.isclose(st.mean[0], 3.0)
    assert np.isclose(st.mean[1], 6.0)
    assert st.counts[1] == 2


def test_streaming_kmeans_splits_two_populations():
    km = StreamingKMeans(dim=3, split_sigma=3.0)
    rng = np.random.default_rng(1)
    for _ in range(80):
        km.update(rng.normal(0, 0.3, 3))
        km.update(rng.normal(8, 0.3, 3))
    assert len(km.populous()) >= 2
    # separation is reported relative to the clusters' own spread
    assert km.separation() > 3.0


def test_streaming_kmeans_single_population_does_not_split():
    km = StreamingKMeans(dim=3, split_sigma=4.0)
    rng = np.random.default_rng(2)
    for _ in range(200):
        km.update(rng.normal(0, 0.3, 3))
    assert km.separation() == 0.0


def test_separation_is_scale_free():
    """A wide, single population must not read as two, at any scale.

    This is the property that stopped a single ordinary phone from raising
    repeated "two radios" alerts: the test is separation relative to the
    clusters' own spread, so scaling every feature changes nothing.
    """
    seps = []
    for scale in (0.1, 1.0, 50.0):
        km = StreamingKMeans(dim=4, split_sigma=4.0)
        rng = np.random.default_rng(11)
        for _ in range(400):
            km.update(rng.normal(0, scale, 4))
        seps.append(km.separation())
    assert max(seps) < 3.0, f"wide single population read as split: {seps}"


def test_separation_needs_both_clusters_populated():
    """A handful of outliers is not a second transmitter."""
    km = StreamingKMeans(dim=3, split_sigma=2.0)
    rng = np.random.default_rng(12)
    for _ in range(200):
        km.update(rng.normal(0, 0.2, 3))
    for _ in range(3):
        km.update(np.array([20.0, 20.0, 20.0]))
    assert km.separation() == 0.0


# --------------------------------------------------------------------------
# enrollment gating
# --------------------------------------------------------------------------

def test_enrollment_requires_temperatures_and_distances():
    p = EnrollmentProgress()
    for _ in range(300):
        p.observe(25.0, -50.0)          # one temperature, one distance
    assert p.packets >= 300
    assert not p.valid
    assert "temperatures" in " ".join(p.blocking())

    for t in (26.0, 27.0, 28.0):
        for r in (-50.0, -70.0):
            p.observe(t, r)
    assert p.valid


def test_incomplete_baseline_raises_no_alerts():
    keys = FEATURE_VECTOR_KEYS
    an = LiveAnalyzer(enroll_addresses=("AA:BB:CC:DD:EE:FF",))
    rng = np.random.default_rng(3)

    def rec(vals, addr="AA:BB:CC:DD:EE:FF"):
        feats = PacketFeatures(
            measurements={k: Measurement(float(v)) for k, v in zip(keys, vals)}
        )
        return PacketRecord(adva=addr, crc_ok=True, features=feats,
                            temperature_c=25.0, rssi_dbfs=-50.0, calibrated=True)

    base = np.zeros(len(keys))
    for _ in range(60):
        an.observe(rec(base + rng.normal(0, 0.05, len(keys))))
    wild = rec(base + 50.0)
    an.observe(wild)
    # the baseline is nowhere near valid, so nothing may be asserted about it
    assert not any("outside baseline" in a for a in wild.alerts)


# --------------------------------------------------------------------------
# per-feature bimodality
# --------------------------------------------------------------------------

def test_one_dimensional_split_finds_two_modes():
    """The 1-D test is what catches a spoofer the 14-D distance dilutes."""
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(-3.9, 3.8, 250), rng.normal(53.6, 3.4, 800)])
    sigma, lo, hi, n_lo = one_dimensional_split(v)
    assert sigma > 10
    assert lo == pytest.approx(-3.9, abs=1.5)
    assert hi == pytest.approx(53.6, abs=1.5)
    assert 200 < n_lo < 300


def test_one_dimensional_split_is_quiet_on_one_population():
    """Any unimodal sample splits somewhat; it must stay well under threshold."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for seed in range(12):
        v = np.random.default_rng(seed).normal(0, 3.5, 400)
        worst = max(worst, one_dimensional_split(v)[0])
    assert worst < 4.0, f"unimodal data reached {worst:.2f} sigma"


def test_one_dimensional_split_needs_both_modes_populated():
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(0, 1.0, 400), np.full(4, 60.0)])
    sigma, _, _, _ = one_dimensional_split(v)
    assert sigma < 6.0, "a handful of outliers must not read as a second radio"


def test_one_dimensional_split_needs_enough_samples():
    assert one_dimensional_split(np.arange(5.0))[0] == 0.0


def test_bimodal_alert_fires_for_two_radios_under_one_address():
    keys = FEATURE_VECTOR_KEYS
    i_cfo = list(keys).index("cfo_ppm")
    an = LiveAnalyzer()
    rng = np.random.default_rng(3)
    alerts = 0
    for i in range(300):
        vals = rng.normal(0, 0.3, len(keys))
        # alternate between two radios that differ only in carrier offset
        vals[i_cfo] = rng.normal(54.0, 3.4) if i % 2 else rng.normal(-4.0, 3.8)
        feats = PacketFeatures(
            measurements={k: Measurement(float(v)) for k, v in zip(keys, vals)}
        )
        r = PacketRecord(adva="AA:BB:CC:DD:EE:01", crc_ok=True, features=feats,
                         timestamp_us=i * 20_000.0, rssi_dbfs=-50.0,
                         temperature_c=25.0)
        an.observe(r)
        alerts += any("splits into two populations" in a for a in r.alerts)
    assert alerts > 0, "two radios under one address raised no bimodality alert"


def test_bearing_is_scanned_for_bimodality_but_is_not_a_clustering_feature():
    """AoA is usually absent, so it must not join the feature vector.

    It is still worth scanning: two transmitters sharing an address are rarely
    in the same direction, and that holds even if their oscillators match.
    Measured on air with a real impersonation the bearing split reached about
    4 sigma against a 3 sigma single-transmitter baseline -- real evidence, but
    below the 6 sigma the carrier offset reaches, so it corroborates rather
    than fires on its own at this array quality.
    """
    an = LiveAnalyzer()
    assert "aoa_deg" not in FEATURE_VECTOR_KEYS
    assert "aoa_deg" in an.bimodal_keys
    for k in FEATURE_VECTOR_KEYS:
        assert k in an.bimodal_keys


def test_bimodal_alert_silent_for_one_radio():
    keys = FEATURE_VECTOR_KEYS
    an = LiveAnalyzer()
    rng = np.random.default_rng(4)
    for i in range(400):
        vals = rng.normal(0, 1.0, len(keys))
        feats = PacketFeatures(
            measurements={k: Measurement(float(v)) for k, v in zip(keys, vals)}
        )
        r = PacketRecord(adva="AA:BB:CC:DD:EE:02", crc_ok=True, features=feats,
                         timestamp_us=i * 20_000.0, rssi_dbfs=-50.0,
                         temperature_c=25.0)
        an.observe(r)
        assert not any("splits into two populations" in a for a in r.alerts)


# --------------------------------------------------------------------------
# anomaly scoring
# --------------------------------------------------------------------------

def _feats(vals):
    return PacketFeatures(
        measurements={k: Measurement(float(v)) for k, v in zip(FEATURE_VECTOR_KEYS, vals)}
    )


def test_anomaly_score_is_low_for_matching_and_high_for_outlier():
    bl = Baseline("X")
    rng = np.random.default_rng(4)
    base = np.linspace(1, 5, len(FEATURE_VECTOR_KEYS))
    for _ in range(200):
        bl.observe(_feats(base + rng.normal(0, 0.1, len(base))), 25.0)
    inlier, _ = bl.score(_feats(base + rng.normal(0, 0.1, len(base))), True)
    outlier, contrib = bl.score(_feats(base + 3.0), True)
    assert inlier < 2.0
    assert outlier > 10.0
    assert contrib, "per-feature contributions must be reported"
    assert max(contrib.values()) > 5.0


def test_uncalibrated_scoring_excludes_reference_dependent_features():
    bl = Baseline("X")
    rng = np.random.default_rng(5)
    base = np.zeros(len(FEATURE_VECTOR_KEYS))
    for _ in range(200):
        bl.observe(_feats(base + rng.normal(0, 0.1, len(base))), 25.0)

    # move only the carrier offset, which is meaningless without a reference
    probe = base.copy()
    probe[list(FEATURE_VECTOR_KEYS).index("cfo_ppm")] = 30.0
    cal, cal_contrib = bl.score(_feats(probe), True)
    uncal, uncal_contrib = bl.score(_feats(probe), False)
    assert cal > uncal
    assert "cfo_ppm" in cal_contrib
    assert "cfo_ppm" not in uncal_contrib


# --------------------------------------------------------------------------
# interference monitor
# --------------------------------------------------------------------------

def test_interference_idle_channel_is_benign():
    m = InterferenceMonitor(8e6)
    rng = np.random.default_rng(6)
    noise = (rng.normal(0, 1e-3, 16384) + 1j * rng.normal(0, 1e-3, 16384)).astype(np.complex64)
    rep = m.observe_block(noise, -60.0)
    assert rep.classification == "benign"


def test_interference_detects_narrowband_cw():
    m = InterferenceMonitor(8e6)
    t = np.arange(16384) / 8e6
    cw = (0.2 * np.exp(2j * np.pi * 100e3 * t)).astype(np.complex64)
    rep = m.observe_block(cw, -60.0)
    assert rep.duty_cycle > 0.9
    assert rep.occupied_bandwidth_hz < 400e3
    assert rep.classification == "cw"


def test_reaction_latency_flags_a_reactive_emitter():
    m = InterferenceMonitor(8e6)
    for k in range(10):
        m.note_preamble(k * 1000.0)
        m.note_energy_onset(k * 1000.0 + 12.0)  # 12 us after every preamble
    assert m.reaction_latency() == pytest.approx(12.0, abs=1.0)


def test_reaction_latency_needs_evidence():
    m = InterferenceMonitor(8e6)
    m.note_preamble(0.0)
    assert np.isnan(m.reaction_latency())


def test_pdr_versus_rssi_and_bit_error_profile():
    m = InterferenceMonitor(8e6)
    for _ in range(20):
        m.note_delivery(-40.0, True)
        m.note_delivery(-80.0, False)
    pdr = dict((b, r) for b, r, _ in m.pdr_vs_rssi())
    assert pdr[float(int(-40.0 // 3) * 3)] == pytest.approx(1.0)
    assert pdr[float(int(-80.0 // 3) * 3)] == pytest.approx(0.0)

    m.note_crc_failure(np.arange(0, 300, 3))
    hist, verdict = m.bit_error_profile()
    assert hist.size > 0 and "uniform" in verdict


def test_bit_error_profile_detects_tail_clustering():
    m = InterferenceMonitor(8e6)
    m.note_crc_failure(np.arange(250, 300))
    _, verdict = m.bit_error_profile()
    assert "late" in verdict or "collision" in verdict
