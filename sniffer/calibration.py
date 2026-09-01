"""Receiver calibration: measure it, judge it, and keep the history.

Calibration here means measuring the *receiver's* own impairments so they are
not later attributed to a transmitter, plus recording the facts that decide
which features are trustworthy at all (reference lock, gain, temperature).

It can run in two ways, and both matter:

* while a capture is running, straight out of the shared IQ ring -- no device
  access, no interruption to the capture;
* while stopped, by opening the device briefly on its own.

Every run is appended to a JSON history so an operator can see whether the
receiver has drifted between sessions, which is the only way to tell a change
in a transmitter from a change in the measuring instrument.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

HISTORY_FILENAME = "calibration_history.json"
MAX_HISTORY = 200

# SC16_Q11 full scale; a peak this close to it is clipping.
CLIP_THRESHOLD = 0.98
# The spec asks for peaks 10-15 dB below full scale.
TARGET_PEAK_DBFS = (-15.0, -10.0)


@dataclass
class CalibrationResult:
    """One calibration run."""

    timestamp: float = 0.0
    source: str = ""  # "live ring" or "dedicated capture"
    channel: int = 37
    frequency_hz: float = 2.402e9
    sample_rate: float = 8e6
    gain_db: int = 0
    samples: int = 0

    # receiver impairments
    dc_i: float = float("nan")
    dc_q: float = float("nan")
    dc_magnitude_dbfs: float = float("nan")
    gain_imbalance: float = float("nan")  # Q/I amplitude ratio
    quadrature_skew_deg: float = float("nan")
    image_rejection_db: float = float("nan")

    # levels
    noise_floor_dbfs: float = float("nan")
    peak_dbfs: float = float("nan")
    rms_dbfs: float = float("nan")
    clipping: bool = False

    # antenna array (dual-antenna captures only)
    antenna_phase_offset_deg: float = float("nan")
    antenna_phase_spread_deg: float = float("nan")
    antenna_coherence: float = float("nan")
    antenna_packets: int = 0

    # environment / reference
    temperature_c: float = float("nan")
    calibrated_reference: bool = False
    clock_detail: str = ""

    notes: list = field(default_factory=list)
    ok: bool = True

    def as_dict(self) -> dict:
        # `when` is a property, so asdict() drops it; the history and the report
        # both key off it, and without it entries render with a blank timestamp.
        d = asdict(self)
        d["when"] = self.when
        return d

    @property
    def when(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

    def summary(self) -> str:
        return (
            f"{self.when}  ch{self.channel}  gain {self.gain_db} dB  "
            f"noise {self.noise_floor_dbfs:.1f} dBFS  peak {self.peak_dbfs:.1f} dBFS"
        )


def measure(iq: np.ndarray, sample_rate: float) -> dict:
    """Compute receiver impairments from a block of samples.

    The DC offset and quadrature error are estimated by Gram-Schmidt
    orthogonalisation of the I and Q rails, the same estimator the live pipeline
    applies per block -- this just reports it so an operator can see it.
    """
    out: dict = {}
    x = np.asarray(iq)
    if x.size < 4096:
        return out

    dc = complex(np.mean(x))
    out["dc_i"] = float(dc.real)
    out["dc_q"] = float(dc.imag)
    out["dc_magnitude_dbfs"] = 20.0 * np.log10(max(abs(dc), 1e-12))

    centred = x - dc
    i = centred.real.astype(np.float64)
    q = centred.imag.astype(np.float64)
    eii = float(np.mean(i * i))
    if eii > 0:
        theta = float(np.mean(i * q)) / eii
        q1 = q - theta * i
        eqq = float(np.mean(q1 * q1))
        if eqq > 0:
            out["gain_imbalance"] = float(np.sqrt(eii / eqq))
        out["quadrature_skew_deg"] = float(np.degrees(np.arcsin(np.clip(theta, -1, 1))))

    # Image rejection: with only receiver noise present the two halves of the
    # spectrum should hold equal power; a systematic difference is the mirror
    # image leaking through an uncorrected imbalance.
    n = int(2 ** np.floor(np.log2(min(x.size, 1 << 16))))
    seg = centred[:n] * np.hanning(n)
    spec = np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    half = n // 2
    lo, hi = float(spec[:half].sum()), float(spec[half:].sum())
    if lo > 0 and hi > 0:
        out["image_rejection_db"] = float(abs(10.0 * np.log10(hi / lo)))

    power = (x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2)
    win = max(int(4e-6 * sample_rate), 1)
    if power.size >= win * 4:
        trimmed = power[: (power.size // win) * win].reshape(-1, win).mean(axis=1)
        out["noise_floor_dbfs"] = float(10.0 * np.log10(max(np.quantile(trimmed, 0.15), 1e-15)))
    out["rms_dbfs"] = float(10.0 * np.log10(max(power.mean(), 1e-15)))
    peak = float(np.max(np.maximum(np.abs(x.real), np.abs(x.imag))))
    out["peak_dbfs"] = float(20.0 * np.log10(max(peak, 1e-12)))
    out["clipping"] = bool(peak >= CLIP_THRESHOLD)
    out["samples"] = int(x.size)
    return out


def judge(result: CalibrationResult) -> None:
    """Attach plain-language notes and set `ok`.

    The notes are the point of the whole exercise: a number an operator cannot
    act on is not a calibration.
    """
    notes: list[str] = []
    ok = True

    if result.clipping:
        notes.append(
            f"CLIPPING at {result.peak_dbfs:.1f} dBFS -- reduce gain; every "
            "amplitude and modulation feature is invalid while the ADC saturates"
        )
        ok = False
    elif np.isfinite(result.peak_dbfs):
        lo, hi = TARGET_PEAK_DBFS
        if result.peak_dbfs > hi:
            notes.append(
                f"peak {result.peak_dbfs:.1f} dBFS is above the {hi:.0f} dBFS "
                f"target; consider reducing gain by "
                f"{result.peak_dbfs - (lo + hi) / 2:.0f} dB"
            )
        elif result.peak_dbfs < lo - 15:
            notes.append(
                f"peak {result.peak_dbfs:.1f} dBFS is well below the target "
                f"{lo:.0f}..{hi:.0f} dBFS; gain could be raised by "
                f"{(lo + hi) / 2 - result.peak_dbfs:.0f} dB for better SNR"
            )
        else:
            notes.append(f"level in range (target {lo:.0f}..{hi:.0f} dBFS peak)")

    if np.isfinite(result.dc_magnitude_dbfs):
        if result.dc_magnitude_dbfs > -40:
            notes.append(
                f"large residual DC at {result.dc_magnitude_dbfs:.1f} dBFS; "
                "it is corrected before feature extraction, but this much "
                "suggests an RFIC DC calibration is due"
            )
        else:
            notes.append(f"DC offset {result.dc_magnitude_dbfs:.1f} dBFS (corrected)")

    if np.isfinite(result.gain_imbalance):
        err_pct = abs(result.gain_imbalance - 1.0) * 100
        if err_pct > 5:
            notes.append(
                f"IQ gain imbalance {err_pct:.1f}% is high; uncorrected it would "
                "bias modulation index by about the between-device spread"
            )
        else:
            notes.append(f"IQ gain imbalance {err_pct:.2f}% (corrected)")

    if np.isfinite(result.quadrature_skew_deg):
        notes.append(f"quadrature skew {result.quadrature_skew_deg:+.2f} deg (corrected)")

    if result.calibrated_reference:
        notes.append("reference locked: ppm-scale features are absolute")
    else:
        notes.append(
            "UNCALIBRATED: no disciplined reference locked. Carrier-offset and "
            "symbol-clock values include receiver drift and are comparable only "
            "within this session, not against a stored baseline"
        )

    if np.isfinite(result.temperature_c):
        notes.append(f"RFIC temperature {result.temperature_c:.1f} C")

    result.notes = notes
    result.ok = ok


def measure_antenna_phase(pairs) -> dict:
    """Fixed phase offset between the two RX chains, from received packets.

    `pairs` is a sequence of (phase difference in degrees, RSSI) taken from
    packets of a transmitter the operator has placed at broadside -- equally
    distant from both antennas.  Whatever phase difference is measured there is
    the array's own, not the source's, and subtracting it is what turns a
    meaningless number into a bearing.

    The offset is a circular mean, because phase wraps: averaging +179 and -179
    arithmetically gives 0, which is exactly wrong.  The resultant length of
    that mean doubles as a coherence figure -- near 1 when both antennas see
    the same signal, near 0 when one of them is unplugged and its phase is
    uniformly random.
    """
    out = {"antenna_packets": 0}
    if pairs is None:
        return out
    angles = np.array(
        [a for a, _ in pairs if a is not None and np.isfinite(a)], dtype=float
    )
    if angles.size < 8:
        return out
    th = np.radians(angles)
    c, sn = float(np.mean(np.cos(th))), float(np.mean(np.sin(th)))
    resultant = float(np.hypot(c, sn))
    out["antenna_packets"] = int(angles.size)
    out["antenna_phase_offset_deg"] = float(np.degrees(np.arctan2(sn, c)))
    out["antenna_coherence"] = resultant
    # circular standard deviation
    out["antenna_phase_spread_deg"] = float(
        np.degrees(np.sqrt(-2.0 * np.log(max(resultant, 1e-9))))
    )
    return out


def judge_antenna(result: "CalibrationResult") -> None:
    """Notes for the array measurement, appended to the ordinary ones."""
    if not result.antenna_packets:
        return
    n = result.antenna_packets
    coh = result.antenna_coherence
    if not np.isfinite(coh):
        return
    if coh < 0.3:
        result.notes.append(
            f"antenna phase incoherent over {n} packets (resultant {coh:.2f}): "
            "the two chains are not seeing the same signal. Check that both "
            "antennas are connected and that RX1 is not terminated"
        )
        result.ok = False
    elif coh < 0.7:
        result.notes.append(
            f"antenna phase only partly coherent over {n} packets "
            f"(resultant {coh:.2f}, spread {result.antenna_phase_spread_deg:.0f} deg); "
            "bearings will be noisy"
        )
    else:
        result.notes.append(
            f"antenna phase offset {result.antenna_phase_offset_deg:+.1f} deg "
            f"over {n} packets (resultant {coh:.2f}, spread "
            f"{result.antenna_phase_spread_deg:.0f} deg) -- subtract this to get "
            "a bearing"
        )


def calibrate_from_samples(
    iq: np.ndarray,
    sample_rate: float,
    *,
    source: str,
    channel: int,
    frequency_hz: float,
    gain_db: int,
    temperature_c: float = float("nan"),
    calibrated_reference: bool = False,
    clock_detail: str = "",
    antenna_pairs=None,
) -> CalibrationResult:
    r = CalibrationResult(
        timestamp=time.time(),
        source=source,
        channel=channel,
        frequency_hz=frequency_hz,
        sample_rate=sample_rate,
        gain_db=gain_db,
        temperature_c=temperature_c,
        calibrated_reference=calibrated_reference,
        clock_detail=clock_detail,
    )
    for k, v in measure(iq, sample_rate).items():
        setattr(r, k, v)
    for k, v in measure_antenna_phase(antenna_pairs).items():
        setattr(r, k, v)
    judge(r)
    judge_antenna(r)
    return r


def calibrate_device(cfg, seconds: float = 0.4) -> CalibrationResult:
    """Open the radio briefly, capture, measure, close.

    Used when no capture is running.  Keeps the device open for as short a time
    as possible so it does not collide with a capture the operator starts next.
    """
    import queue as _q

    from .radio import BladeRF, CaptureStats, CaptureThread

    radio = BladeRF(cfg, log=lambda *_a: None)
    radio.start()
    try:
        q: "_q.Queue" = _q.Queue(maxsize=64)
        stats = CaptureStats()
        thread = CaptureThread(radio, q, stats)
        thread.start()
        blocks = []
        want = int(seconds * cfg.sample_rate)
        deadline = time.time() + max(seconds * 6, 4.0)
        got = 0
        while got < want and time.time() < deadline:
            try:
                b = q.get(timeout=0.5)
            except _q.Empty:
                continue
            blocks.append(b.as_complex(0))
            got += b.n_samples
        thread.stop()
        thread.join(timeout=1.0)
        iq = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.complex64)
        return calibrate_from_samples(
            iq,
            cfg.sample_rate,
            source="dedicated capture",
            channel=cfg.plan.channel,
            frequency_hz=cfg.plan.frequency_hz,
            gain_db=cfg.gain_db,
            temperature_c=radio.rfic_temperature(max_age=0.0),
            calibrated_reference=radio.clock.calibrated,
            clock_detail=radio.clock.detail,
        )
    finally:
        radio.close()


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------

class CalibrationHistory:
    """Append-only JSON log of calibration runs."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.path.join(os.getcwd(), HISTORY_FILENAME)
        self.entries: list[dict] = []
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.entries = data if isinstance(data, list) else []
        except (OSError, ValueError):
            self.entries = []

    def add(self, result: CalibrationResult) -> None:
        self.entries.append(result.as_dict())
        del self.entries[:-MAX_HISTORY]
        self.save()

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.entries, fh, indent=2, default=str)
        except OSError:
            pass

    def clear(self) -> None:
        self.entries = []
        self.save()

    def latest(self) -> dict | None:
        return self.entries[-1] if self.entries else None

    def drift(self, key: str) -> tuple[float, float] | None:
        """(first, last) of a numeric field, for spotting receiver drift."""
        vals = [e.get(key) for e in self.entries if isinstance(e.get(key), (int, float))]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) < 2:
            return None
        return float(vals[0]), float(vals[-1])
