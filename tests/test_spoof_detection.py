"""End-to-end spoofer detection and its false-positive rate.

Acceptance criterion: detect a software spoofer replaying a captured AdvA from a
different radio, at a stated false-positive rate below 1 %.  Both halves are
measured here on synthetic signals, because only a generator can guarantee that
two streams really do come from two different "radios" while sharing one
address -- and that a control stream really does come from one.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace

from sniffer.analysis import LiveAnalyzer
from sniffer.dsp import Demodulator
from sniffer.features import extract_features
from sniffer.packet import PacketRecord
from tests.synth import TxImpairments, make_packet

FS = 8e6
ADDR = b"\xde\xad\xbe\xef\x12\x34"

# Two radios that a spoofer might plausibly present: same address, same payload,
# different silicon.  The differences here are modest on purpose -- well inside
# what two units of the same part number can differ by.
LEGIT = TxImpairments(
    cfo_hz=18e3, modulation_index=0.505, bt=0.50, ramp_us=2.0,
    slew_asymmetry=0.05, drift_hz=6e3, snr_db=30,
)
SPOOFER = TxImpairments(
    cfo_hz=-24e3, modulation_index=0.478, bt=0.44, ramp_us=3.4,
    slew_asymmetry=-0.18, drift_hz=-11e3, snr_db=30,
)


def _record(imp, seed, number, addr=ADDR):
    rng = np.random.default_rng(seed)
    sig, _ = make_packet(imp=imp, sample_rate=FS, adva=addr, rng=rng)
    demod = Demodulator(FS, channel=37)
    dets = demod.process(sig, gate=False)
    if not dets or not dets[0].pdu.crc_ok:
        return None
    det = dets[0]
    sl = sig[det.slice_start : det.slice_end]
    feats = extract_features(
        sl, FS, det.sync_index - det.slice_start, det.sym_offset,
        det.bits, det.freq_symbols, calibrated=True,
    )
    return PacketRecord(
        number=number,
        timestamp_us=number * 100_000.0,
        wall_time=1.7e9 + number * 0.1,
        adva=det.pdu.adva_str,
        adva_kind=det.pdu.adva_kind,
        pdu_name=det.pdu.pdu_name,
        crc_ok=True,
        calibrated=True,
        temperature_c=25.0 + (number % 4),
        rssi_dbfs=feats.value("rssi_dbfs"),
        features=feats,
    )


def _run(sequence) -> tuple[LiveAnalyzer, list]:
    an = LiveAnalyzer()
    recs = []
    for i, imp in enumerate(sequence):
        r = _record(imp, seed=1000 + i, number=i + 1)
        if r is None:
            continue
        an.observe(r)
        recs.append(r)
    return an, recs


def _alerted(recs) -> bool:
    return any(
        any("two separated feature clusters" in a for a in r.alerts) for r in recs
    )


@pytest.mark.slow
@pytest.mark.parametrize("every", [2, 3, 5])
def test_detects_a_spoofer_sharing_one_address(every):
    """A second radio under the same AdvA must be caught.

    Measured separation (per-dimension, relative to the clusters' own spread):
    1.15 at one spoofed packet in two, rising to 1.55 at one in five -- a
    sparser spoofer is easier, because it contaminates the legitimate cluster
    less.  A single radio measures at most 0.63 under the same conditions, so
    the 0.9 alert threshold has margin on both sides.
    """
    seq = [SPOOFER if i % every == 0 else LEGIT for i in range(150)]
    an, recs = _run(seq)
    assert recs, "no synthetic packets decoded"
    st = an.addresses[recs[0].adva]
    assert len(st.kmeans.populous()) >= 2, "the two radios did not separate"
    assert st.kmeans.separation() > 0.9
    assert _alerted(recs), "spoofer under a shared address raised no alert"


@pytest.mark.slow
def test_false_positive_rate_below_one_percent_for_a_single_radio():
    """One radio, 200 packets, ordinary noise: the alert must stay silent.

    This is the number quoted in the README.  Run as a proportion of packets:
    an alert on any packet in a single-radio stream is a false positive.
    """
    total = 0
    false_alerts = 0
    for snr in (18, 24, 30, 36):
        for seed0 in (1000, 7000):
            an = LiveAnalyzer()
            for i in range(80):
                r = _record(replace(LEGIT, snr_db=snr), seed0 + i, i + 1)
                if r is None:
                    continue
                an.observe(r)
                total += 1
                if any("two separated feature clusters" in a for a in r.alerts):
                    false_alerts += 1
    assert total > 400
    rate = false_alerts / total
    assert rate < 0.01, f"false-positive rate {rate:.1%} over {total} packets exceeds 1%"


@pytest.mark.slow
def test_a_second_radio_is_separable_in_feature_space():
    """The feature vectors themselves must differ, independent of the clusterer."""
    a = [_record(LEGIT, 2000 + i, i) for i in range(24)]
    b = [_record(SPOOFER, 3000 + i, i) for i in range(24)]
    a = [r for r in a if r is not None]
    b = [r for r in b if r is not None]
    assert len(a) > 18 and len(b) > 18

    va = np.array([r.features.vector() for r in a])
    vb = np.array([r.features.vector() for r in b])
    ok = np.isfinite(va).all(axis=0) & np.isfinite(vb).all(axis=0)
    va, vb = va[:, ok], vb[:, ok]

    # Fisher-style separation per feature: how many within-device standard
    # deviations apart the two device means sit.
    sep = np.abs(va.mean(0) - vb.mean(0)) / np.sqrt(
        (va.std(0) ** 2 + vb.std(0) ** 2) / 2 + 1e-12
    )
    assert np.nanmax(sep) > 3.0, f"no feature separates the two radios: {sep}"
    assert (sep > 1.0).sum() >= 3, "separation rests on a single feature"


@pytest.mark.slow
def test_address_rotation_does_not_hide_the_radio():
    """Physical-layer features persist when the advertised address changes.

    This is the observation the tool is built on, and the limit of what it
    claims: rotating the address does not rotate the radio.
    """
    addr_a = b"\x11\x22\x33\x44\x55\x66"
    addr_b = b"\xaa\xbb\xcc\xdd\xee\xff"
    first = [_record(LEGIT, 4000 + i, i, addr_a) for i in range(20)]
    second = [_record(LEGIT, 5000 + i, i, addr_b) for i in range(20)]
    first = [r for r in first if r is not None]
    second = [r for r in second if r is not None]

    v1 = np.array([r.features.vector() for r in first])
    v2 = np.array([r.features.vector() for r in second])
    ok = np.isfinite(v1).all(axis=0) & np.isfinite(v2).all(axis=0)
    v1, v2 = v1[:, ok], v2[:, ok]

    sep = np.abs(v1.mean(0) - v2.mean(0)) / np.sqrt(
        (v1.std(0) ** 2 + v2.std(0) ** 2) / 2 + 1e-12
    )
    # Same radio, different advertised address: the features must NOT separate.
    assert np.nanmedian(sep) < 1.5, (
        f"the same radio looked different across an address change: {sep}"
    )
