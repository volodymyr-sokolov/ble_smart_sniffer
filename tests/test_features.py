"""Estimator accuracy against the synthetic generator's ground truth.

Each test injects a known impairment and asserts the estimator recovers it to a
stated tolerance.  The tolerances are the numbers quoted in the README's feature
table -- if one of these loosens, that table is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from sniffer import features as F
from sniffer.channels import ChannelPlan
from sniffer.dsp import Demodulator
from tests.synth import TxImpairments, make_packet

FS = 8e6


def analyse(imp: TxImpairments, channel: int = 37, seed: int = 0, **kw):
    """Modulate, demodulate and extract features, returning (features, truth)."""
    rng = np.random.default_rng(seed)
    sig, truth = make_packet(channel=channel, imp=imp, sample_rate=FS, rng=rng, **kw)
    demod = Demodulator(FS, channel=channel)
    dets = demod.process(sig, gate=False)
    assert dets, "synthetic packet was not detected"
    det = dets[0]
    assert det.pdu.crc_ok, "synthetic packet failed CRC"
    sl = sig[det.slice_start : det.slice_end]
    feats = F.extract_features(
        sl,
        FS,
        sync_offset=det.sync_index - det.slice_start,
        sym_offset=det.sym_offset,
        bits=det.bits,
        sym_freq=det.freq_symbols,
        calibrated=True,
    )
    return feats, truth, det


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

@pytest.mark.parametrize("channel", [37, 38, 39, 0, 11, 36])
def test_decodes_on_every_channel(channel):
    """The whitening seed must follow the channel index, not be hard-coded."""
    kw = {}
    if channel not in (37, 38, 39):
        kw = {}  # synth always uses the advertising AA; only the seed changes
    sig, truth = make_packet(channel=channel, imp=TxImpairments(snr_db=30))
    demod = Demodulator(FS, channel=channel)
    dets = demod.process(sig, gate=False)
    assert dets and dets[0].pdu.crc_ok
    assert dets[0].pdu.adva == truth["adva"]


def test_wrong_channel_seed_fails_crc():
    """De-whitening with the wrong seed must fail, not silently pass."""
    sig, _ = make_packet(channel=37, imp=TxImpairments(snr_db=30))
    demod = Demodulator(FS, channel=38)  # deliberate mismatch
    dets = demod.process(sig, gate=False)
    assert all(not d.pdu.crc_ok for d in dets)


def test_channel_plan_rejects_lo_seed_mismatch():
    plan = ChannelPlan.from_args(channel=37)
    bad = ChannelPlan(
        channel=37, frequency_hz=2_426_000_000.0, access_address=plan.access_address,
        crc_init=plan.crc_init, whitening_channel=37, label="bad",
    )
    with pytest.raises(AssertionError):
        bad.assert_consistent()


# --------------------------------------------------------------------------
# carrier and oscillator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cfo", [-120e3, -40e3, 0.0, 35e3, 140e3])
def test_carrier_frequency_offset(cfo):
    feats, truth, _ = analyse(TxImpairments(cfo_hz=cfo, snr_db=35))
    est = feats.value("cfo_hz")
    assert abs(est - cfo) < 4e3, f"CFO {cfo:.0f} Hz estimated as {est:.0f} Hz"


def test_cfo_uncertainty_smaller_than_between_device_spread():
    """Acceptance criterion 8: uncertainty must beat the spread it resolves."""
    unc = []
    for seed in range(6):
        feats, _, _ = analyse(TxImpairments(cfo_hz=20e3, snr_db=25), seed=seed)
        unc.append(feats.get("cfo_hz").uncertainty)
    # Between-device spread is set by crystal tolerance: +/-20 ppm parts at
    # 2.4 GHz are +/-48 kHz apart, so the population spread is tens of kHz.
    # The per-packet uncertainty has to be well inside that to separate devices.
    assert np.nanmedian(unc) < 5e3


@pytest.mark.parametrize("drift", [-30e3, 20e3, 45e3])
def test_frequency_drift(drift):
    feats, _, _ = analyse(TxImpairments(drift_hz=drift, snr_db=35))
    est = feats.value("drift_hz")
    assert abs(est - drift) < 0.25 * abs(drift) + 4e3


def test_drift_rate_reported():
    feats, _, _ = analyse(TxImpairments(drift_hz=40e3, snr_db=35))
    assert np.isfinite(feats.value("drift_rate"))


# --------------------------------------------------------------------------
# modulation quality
# --------------------------------------------------------------------------

@pytest.mark.parametrize("h", [0.45, 0.50, 0.55])
def test_modulation_index(h):
    feats, _, _ = analyse(TxImpairments(modulation_index=h, snr_db=35))
    est = feats.value("modulation_index")
    assert abs(est - h) < 0.015, f"h={h} estimated as {est:.4f}"


def test_modulation_index_ordering():
    """Even where absolute accuracy is limited, ordering must be preserved."""
    vals = [
        analyse(TxImpairments(modulation_index=h, snr_db=35))[0].value("modulation_index")
        for h in (0.45, 0.50, 0.55)
    ]
    assert vals[0] < vals[1] < vals[2]


@pytest.mark.parametrize("asym", [-0.08, 0.0, 0.08])
def test_deviation_asymmetry_is_degenerate_with_cfo(asym):
    """Documents why dev_asymmetry is excluded from the feature vector.

    Deviation asymmetry and carrier offset produce mathematically identical
    waveforms, so the estimator absorbs the injected asymmetry into the carrier
    offset and reports ~0.  This is not a tolerance failure; it is the reason
    the feature is not clustered on.
    """
    feats, _, _ = analyse(TxImpairments(dev_asymmetry=asym, snr_db=35))
    assert abs(feats.value("dev_asymmetry")) < 0.03
    assert "dev_asymmetry" not in F.FEATURE_VECTOR_KEYS


def test_transition_asymmetry_is_identifiable():
    """The identifiable replacement must order correctly and cancel the offset.

    Driven by `slew_asymmetry` (a direction-dependent slew rate), not by
    `dev_asymmetry` -- a deviation imbalance is degenerate with carrier offset
    and produces no measurable change in transition duration.
    """
    vals = [
        analyse(TxImpairments(slew_asymmetry=a, snr_db=40))[0].value("transition_asymmetry")
        for a in (-0.4, 0.0, 0.4)
    ]
    assert vals[0] > vals[1] > vals[2], f"not monotonic: {vals}"

    # A pure carrier offset must not move it: that is what "identifiable" means.
    flat = [
        analyse(TxImpairments(cfo_hz=c, snr_db=35))[0].value("transition_asymmetry")
        for c in (0.0, 100e3)
    ]
    assert abs(flat[0] - flat[1]) < 0.08


@pytest.mark.parametrize("bt", [0.35, 0.5, 0.8])
def test_effective_bt(bt):
    """BT is calibrated at SNR 35 dB; the transition broadens with noise."""
    feats, _, _ = analyse(TxImpairments(bt=bt, snr_db=35))
    est = feats.value("effective_bt")
    assert abs(est - bt) < 0.06, f"BT={bt} estimated as {est:.4f}"


# A longer payload gives more transitions, and the symbol-clock estimator
# averages over transitions; the accuracy quoted in the README assumes a
# typical 30-byte advertising payload rather than a minimal one.
LONG_PAYLOAD = bytes([0x02, 0x01, 0x06, 0x1B, 0xFF]) + bytes(range(26))


@pytest.mark.parametrize("ppm", [-80.0, -40.0, 0.0, 40.0, 80.0])
def test_symbol_clock_offset(ppm):
    feats, _, _ = analyse(
        TxImpairments(symbol_clock_ppm=ppm, snr_db=35), ad_payload=LONG_PAYLOAD
    )
    m = feats.get("symbol_clock_ppm")
    assert np.isfinite(m.value)
    assert abs(m.value - ppm) < 20.0, f"{ppm} ppm estimated as {m.value:.1f}"
    assert m.uncertainty < 15.0


def test_symbol_clock_sign_convention():
    """Positive means the transmitter's symbol clock runs fast, as for CFO."""
    fast, _, _ = analyse(
        TxImpairments(symbol_clock_ppm=80.0, snr_db=40), ad_payload=LONG_PAYLOAD
    )
    slow, _, _ = analyse(
        TxImpairments(symbol_clock_ppm=-80.0, snr_db=40), ad_payload=LONG_PAYLOAD
    )
    assert fast.value("symbol_clock_ppm") > slow.value("symbol_clock_ppm")


def test_timing_jitter_is_monotonic_above_its_noise_floor():
    """Jitter is measured from interpolated crossings and has a noise floor.

    At 8 MSPS the crossing estimate itself scatters by about 6 ns, so jitter
    below that is not resolvable and the estimator correctly reports the floor
    rather than zero.
    """
    vals = [
        analyse(TxImpairments(timing_jitter_ps=j, snr_db=40), ad_payload=LONG_PAYLOAD)[0]
        .value("symbol_jitter_ps")
        for j in (0, 3000, 10000)
    ]
    assert vals[0] < vals[1] < vals[2]
    assert vals[0] < 9000  # the floor, not an arbitrary number


def test_frequency_error_grows_with_noise():
    clean, _, _ = analyse(TxImpairments(snr_db=40), seed=3)
    noisy, _, _ = analyse(TxImpairments(snr_db=15), seed=3)
    assert noisy.value("freq_error_rms") > clean.value("freq_error_rms")


# --------------------------------------------------------------------------
# envelope and transient
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ramp", [1.0, 3.0, 6.0])
def test_pa_rise_time(ramp):
    feats, _, _ = analyse(TxImpairments(ramp_us=ramp, snr_db=40))
    est = feats.value("rise_time_us")
    assert np.isfinite(est), "rise time not measured"
    assert abs(est - ramp) < 0.5 * ramp + 1.0


def test_ramp_vector_is_normalised():
    feats, _, _ = analyse(TxImpairments(ramp_us=2.0, snr_db=40))
    v = feats.ramp_vector
    assert v.shape == (32,)
    assert 0.0 <= v.min() <= 0.05 and 0.95 <= v.max() <= 1.0


def test_overshoot_detected():
    none, _, _ = analyse(TxImpairments(ramp_overshoot=0.0, snr_db=40))
    over, _, _ = analyse(TxImpairments(ramp_overshoot=0.25, snr_db=40))
    assert over.value("overshoot") > none.value("overshoot") + 0.05


# --------------------------------------------------------------------------
# amplitude, multipath, spatial
# --------------------------------------------------------------------------

def test_rssi_tracks_amplitude():
    lo, _, _ = analyse(TxImpairments(amplitude=0.05, snr_db=30))
    hi, _, _ = analyse(TxImpairments(amplitude=0.40, snr_db=30))
    delta = hi.value("rssi_dbfs") - lo.value("rssi_dbfs")
    assert abs(delta - 20 * np.log10(0.40 / 0.05)) < 3.0


def test_rssi_dbm_uncalibrated_without_table():
    feats, _, _ = analyse(TxImpairments(snr_db=30))
    assert not feats.get("rssi_dbm").calibrated
    assert np.isnan(feats.value("rssi_dbm"))


def test_delay_spread_small_for_clean_channel():
    feats, _, _ = analyse(TxImpairments(snr_db=35))
    ds = feats.value("delay_spread_us")
    assert np.isfinite(ds) and ds < 1.0


def test_delay_spread_increases_with_echo():
    rng = np.random.default_rng(7)
    sig, _ = make_packet(imp=TxImpairments(snr_db=35), rng=rng)
    echo = np.zeros_like(sig)
    d = 12  # 1.5 us at 8 MSPS
    echo[d:] = 0.5 * sig[:-d]
    multi = (sig + echo).astype(np.complex64)

    def spread(x):
        demod = Demodulator(FS, channel=37)
        det = demod.process(x, gate=False)[0]
        sl = x[det.slice_start : det.slice_end]
        f = F.extract_features(
            sl, FS, det.sync_index - det.slice_start, det.sym_offset,
            det.bits, det.freq_symbols, calibrated=True,
        )
        return f.value("delay_spread_us")

    assert spread(multi) > spread(sig) + 0.1


def test_uncalibrated_flag_propagates():
    """Without a locked reference every ppm-scale feature must say so."""
    rng = np.random.default_rng(1)
    sig, _ = make_packet(imp=TxImpairments(cfo_hz=20e3), rng=rng)
    demod = Demodulator(FS, channel=37)
    det = demod.process(sig, gate=False)[0]
    sl = sig[det.slice_start : det.slice_end]
    feats = F.extract_features(
        sl, FS, det.sync_index - det.slice_start, det.sym_offset,
        det.bits, det.freq_symbols, calibrated=False,
    )
    for key in ("cfo_hz", "cfo_ppm", "drift_hz", "symbol_clock_ppm"):
        assert not feats.get(key).calibrated, f"{key} should be UNCALIBRATED"
    # amplitude-domain features do not depend on the frequency reference
    assert feats.get("rssi_dbfs").calibrated


def test_feature_vector_shape_and_finiteness():
    feats, _, _ = analyse(TxImpairments(snr_db=30))
    v = feats.vector()
    assert v.shape == (len(F.FEATURE_VECTOR_KEYS),)
    assert np.isfinite(v).sum() >= len(v) - 2


def test_spec_limit_flagging():
    inside, _, _ = analyse(TxImpairments(cfo_hz=50e3, snr_db=35))
    outside, _, _ = analyse(TxImpairments(cfo_hz=400e3, snr_db=35))
    assert inside.get("cfo_hz").in_spec is True
    assert outside.get("cfo_hz").in_spec is False
