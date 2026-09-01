"""Calibration measurement, judgement and history."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from sniffer.calibration import (
    CalibrationHistory,
    CalibrationResult,
    calibrate_from_samples,
    judge,
    measure,
)


def synth(n=200_000, dc=0.0 + 0.0j, gain_imbalance=1.0, skew_deg=0.0,
          amp=0.02, seed=0):
    """Receiver noise with known, deliberately injected impairments."""
    rng = np.random.default_rng(seed)
    i = rng.normal(0, amp, n)
    q = rng.normal(0, amp, n)
    # apply a quadrature skew and a gain error, then a DC offset
    theta = np.radians(skew_deg)
    q = (q / gain_imbalance) + np.tan(theta) * i
    return (i + 1j * q + dc).astype(np.complex64)


def test_measures_dc_offset():
    out = measure(synth(dc=0.01 - 0.004j), 8e6)
    assert out["dc_i"] == pytest.approx(0.01, abs=5e-4)
    assert out["dc_q"] == pytest.approx(-0.004, abs=5e-4)
    assert out["dc_magnitude_dbfs"] == pytest.approx(
        20 * np.log10(abs(0.01 - 0.004j)), abs=0.5
    )


def test_measures_gain_imbalance():
    for g in (0.9, 1.0, 1.15):
        out = measure(synth(gain_imbalance=g), 8e6)
        assert out["gain_imbalance"] == pytest.approx(g, rel=0.03)


def test_measures_quadrature_skew():
    for skew in (-4.0, 0.0, 6.0):
        out = measure(synth(skew_deg=skew), 8e6)
        assert out["quadrature_skew_deg"] == pytest.approx(skew, abs=0.6)


def test_measures_levels_and_detects_clipping():
    quiet = measure(synth(amp=0.01), 8e6)
    assert quiet["noise_floor_dbfs"] < -20
    assert not quiet["clipping"]

    loud = synth(amp=0.01)
    loud[100] = 1.0 + 0.0j
    out = measure(loud, 8e6)
    assert out["clipping"]
    assert out["peak_dbfs"] == pytest.approx(0.0, abs=0.5)


def test_measure_refuses_a_short_block():
    assert measure(np.zeros(16, dtype=np.complex64), 8e6) == {}


def test_judgement_flags_clipping_as_actionable():
    r = CalibrationResult(peak_dbfs=-0.2, clipping=True)
    judge(r)
    assert not r.ok
    assert any("CLIPPING" in n for n in r.notes)


def test_judgement_suggests_a_gain_change():
    hot = CalibrationResult(peak_dbfs=-3.0)
    judge(hot)
    assert any("reducing gain" in n for n in hot.notes)

    cold = CalibrationResult(peak_dbfs=-55.0)
    judge(cold)
    assert any("raised" in n for n in cold.notes)

    good = CalibrationResult(peak_dbfs=-12.0)
    judge(good)
    assert good.ok
    assert any("in range" in n for n in good.notes)


def test_judgement_states_the_reference_situation():
    uncal = CalibrationResult(peak_dbfs=-12.0, calibrated_reference=False)
    judge(uncal)
    assert any("UNCALIBRATED" in n for n in uncal.notes)
    cal = CalibrationResult(peak_dbfs=-12.0, calibrated_reference=True)
    judge(cal)
    assert any("absolute" in n for n in cal.notes)


def test_calibrate_from_samples_records_context():
    r = calibrate_from_samples(
        synth(), 8e6, source="live ring", channel=38,
        frequency_hz=2.426e9, gain_db=42, temperature_c=41.5,
        calibrated_reference=False, clock_detail="onboard VCTCXO",
    )
    assert r.channel == 38 and r.gain_db == 42
    assert r.source == "live ring"
    assert r.samples > 0
    assert r.notes
    assert "ch38" in r.summary()


def test_as_dict_keeps_the_human_readable_timestamp():
    """asdict() drops properties; the history and report both key off `when`."""
    r = calibrate_from_samples(
        synth(n=40_000), 8e6, source="t", channel=37,
        frequency_hz=2.402e9, gain_db=40,
    )
    d = r.as_dict()
    assert d["when"] == r.when
    assert len(d["when"]) == 19  # YYYY-mm-dd HH:MM:SS


def test_history_round_trip_and_cap():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hist.json")
        h = CalibrationHistory(path)
        assert h.entries == []
        for i in range(5):
            h.add(calibrate_from_samples(
                synth(n=50_000, seed=i), 8e6, source="test", channel=37,
                frequency_hz=2.402e9, gain_db=40 + i,
            ))
        assert len(h.entries) == 5
        again = CalibrationHistory(path)
        assert len(again.entries) == 5
        assert again.latest()["gain_db"] == 44
        again.clear()
        assert CalibrationHistory(path).entries == []


def test_history_survives_a_corrupt_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hist.json")
        open(path, "w").write("{not json")
        assert CalibrationHistory(path).entries == []


def test_history_reports_drift():
    with tempfile.TemporaryDirectory() as d:
        h = CalibrationHistory(os.path.join(d, "h.json"))
        assert h.drift("noise_floor_dbfs") is None
        for amp in (0.01, 0.02):
            h.add(calibrate_from_samples(
                synth(n=60_000, amp=amp), 8e6, source="t", channel=37,
                frequency_hz=2.402e9, gain_db=40,
            ))
        first, last = h.drift("noise_floor_dbfs")
        assert last > first  # louder noise the second time


# --------------------------------------------------------------------------
# the dialog renders whatever it is given
# --------------------------------------------------------------------------

def test_report_html_renders_and_handles_empty():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from sniffer.gui.dialogs import calibration_html

    assert "No calibration" in calibration_html({})
    r = calibrate_from_samples(
        synth(n=60_000), 8e6, source="live ring", channel=37,
        frequency_hz=2.402e9, gain_db=45,
    )
    html = calibration_html(r.as_dict())
    for section in ("Configuration", "Levels", "Receiver impairments",
                    "Environment", "Notes"):
        assert section in html
