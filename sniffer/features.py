"""Per-packet physical-layer feature extraction.

One function per feature, each independently testable against the synthetic
generator in `tests/synth.py`.  Every function takes explicit arrays and returns
a `Measurement` carrying value, uncertainty and calibration state, so nothing
downstream can accidentally treat a receiver artefact as a transmitter property.

Two rules run through the whole module:

* Frequency-domain features are computed from the *residual* -- measured minus
  the ideal trajectory implied by the decoded bits -- not from raw statistics of
  the signal.  A mean over "roughly balanced" data is biased by whatever the
  data happened to be; a mean over the residual is not.
* Any feature whose value depends on the receiver's own frequency reference is
  marked UNCALIBRATED unless a disciplined external clock was locked.  A 40 ppm
  VCTCXO drifting with room temperature moves the carrier-offset estimate by
  about 100 kHz at 2.4 GHz, which is larger than the entire between-device
  spread this application is trying to resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import signal as sps_signal

from .channels import BLE_SYMBOL_RATE

# Spec limits, Core spec Vol 6 Part A 3.1 / 4.1 (LE 1M PHY).
SPEC_LIMITS = {
    "cfo_hz": (-150e3, 150e3),
    "drift_hz": (-50e3, 50e3),
    "drift_rate_hz_per_50us": (-20e3, 20e3),
    "modulation_index": (0.45, 0.55),
    "freq_dev_hz": (225e3, 275e3),
}

CARRIER_HZ_DEFAULT = 2.402e9

# The demodulator filters hard (1.1 MHz) because it only has to make bit
# decisions.  Measurement cannot use that signal: a lowpass narrow enough to
# help the slicer is not flat across the occupied band, and it compresses the
# instantaneous frequency toward DC.  Measured against the synthetic generator,
# a 1.1 MHz filter recovers only 0.907 of a 400 kHz carrier offset -- a 9%
# scale error, far larger than the between-device spread being resolved.
# 2.5 MHz recovers 0.9974 while still removing 30% of the noise that the
# unfiltered signal carries, so features are measured through this one.
MEASUREMENT_FILTER_HZ = 2.5e6

# Effective-BT calibration (see `effective_bt`).  Fitted against the synthetic
# generator at SNR 35 dB, which is representative of a packet strong enough to
# fingerprint; the transition time broadens with noise, so a fit taken on a
# noiseless signal reads about 0.05 low at realistic levels.
BT_FIT_SLOPE = 0.4192
BT_FIT_INTERCEPT = -0.0655


@dataclass
class Measurement:
    """A number with an uncertainty and an honest calibration flag."""

    value: float
    uncertainty: float = float("nan")
    units: str = ""
    calibrated: bool = True
    spec_low: float | None = None
    spec_high: float | None = None

    @property
    def in_spec(self) -> bool | None:
        if self.spec_low is None and self.spec_high is None:
            return None
        if not np.isfinite(self.value):
            return None
        lo = -np.inf if self.spec_low is None else self.spec_low
        hi = np.inf if self.spec_high is None else self.spec_high
        return bool(lo <= self.value <= hi)

    def format(self) -> str:
        if not np.isfinite(self.value):
            return "n/a"
        s = f"{self.value:.4g}"
        if np.isfinite(self.uncertainty):
            s += f" +/- {self.uncertainty:.3g}"
        if self.units:
            s += f" {self.units}"
        if not self.calibrated:
            s += "  [UNCALIBRATED]"
        return s

    @staticmethod
    def _num(v: float) -> str:
        """Plain decimal where it is readable, exponent only where it is not.

        A spec limit rendered as 1.5e+05 Hz makes an operator do arithmetic to
        compare it with a measurement; 150000 does not.
        """
        if v is None:
            return "?"
        if v == int(v) and abs(v) < 1e9:
            return f"{int(v)}"
        return f"{v:.6g}"

    def spec_text(self) -> str:
        if self.spec_low is None and self.spec_high is None:
            return "-"
        lo = "-inf" if self.spec_low is None else self._num(self.spec_low)
        hi = "+inf" if self.spec_high is None else self._num(self.spec_high)
        return f"{lo} .. {hi} {self.units}".strip()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def ideal_symbol_frequency(bits: np.ndarray, deviation_hz: float) -> np.ndarray:
    """Ideal per-symbol frequency for a bit sequence: +dev for 1, -dev for 0."""
    return (bits.astype(np.float64) * 2.0 - 1.0) * deviation_hz


def settled_symbol_mask(bits: np.ndarray, run: int = 3) -> np.ndarray:
    """Symbols at the centre of a run of `run` identical bits.

    Gaussian shaping means an isolated bit never reaches full deviation, so a
    naive peak-deviation estimate reads low by 20-30% and varies with the data.
    Restricting to settled symbols makes the modulation-index estimate a
    property of the transmitter rather than of the payload.
    """
    n = len(bits)
    mask = np.zeros(n, dtype=bool)
    if n < run:
        return mask
    half = run // 2
    same = np.ones(n, dtype=bool)
    for k in range(1, half + 1):
        same[k:] &= bits[k:] == bits[:-k]
        same[:-k] &= bits[:-k] == bits[k:]
    mask[half : n - half] = same[half : n - half]
    return mask


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size > 1 else float("nan")


def _sem(x: np.ndarray) -> float:
    """Standard error of the mean -- the uncertainty attached to an average."""
    return float(np.std(x) / np.sqrt(len(x))) if x.size > 1 else float("nan")


# --------------------------------------------------------------------------
# carrier and oscillator
# --------------------------------------------------------------------------

def carrier_frequency_offset(
    sym_freq: np.ndarray, bits: np.ndarray, deviation_hz: float, calibrated: bool = False
) -> Measurement:
    """Mean residual frequency: the carrier offset, free of data bias.

    Subtracting the ideal trajectory first matters.  Advertising payloads are
    whitened but not DC-free over only 300-odd symbols, and taking a plain mean
    leaves a data-dependent error of several kHz -- comparable to the
    between-device spread the fingerprint depends on.
    """
    n = min(len(sym_freq), len(bits))
    if n < 8:
        return Measurement(float("nan"), units="Hz", calibrated=calibrated)
    resid = sym_freq[:n] - ideal_symbol_frequency(bits[:n], deviation_hz)
    return Measurement(
        float(np.mean(resid)),
        _sem(resid),
        "Hz",
        calibrated,
        *SPEC_LIMITS["cfo_hz"],
    )


def cfo_ppm(cfo: Measurement, carrier_hz: float = CARRIER_HZ_DEFAULT) -> Measurement:
    """Carrier offset expressed in ppm of the RF carrier."""
    scale = 1e6 / carrier_hz
    return Measurement(
        cfo.value * scale,
        cfo.uncertainty * scale,
        "ppm",
        cfo.calibrated,
        SPEC_LIMITS["cfo_hz"][0] * scale,
        SPEC_LIMITS["cfo_hz"][1] * scale,
    )


def initial_frequency_offset(
    sym_freq: np.ndarray, bits: np.ndarray, deviation_hz: float, n_preamble: int = 8,
    calibrated: bool = False,
) -> Measurement:
    """Offset measured over the preamble only (spec limit +/-150 kHz)."""
    n = min(n_preamble, len(sym_freq), len(bits))
    if n < 4:
        return Measurement(float("nan"), units="Hz", calibrated=calibrated)
    resid = sym_freq[:n] - ideal_symbol_frequency(bits[:n], deviation_hz)
    return Measurement(
        float(np.mean(resid)), _sem(resid), "Hz", calibrated, *SPEC_LIMITS["cfo_hz"]
    )


def frequency_drift(
    sym_freq: np.ndarray, bits: np.ndarray, deviation_hz: float, calibrated: bool = False
) -> tuple[Measurement, Measurement, np.ndarray]:
    """Total drift, max drift rate per 50 us, and the fitted trajectory.

    The trajectory coefficients are the thermal signature of the transmitter
    after PA turn-on: a crystal pulled by the heat of its own power amplifier
    traces a repeatable curve that survives address rotation.
    """
    n = min(len(sym_freq), len(bits))
    nan = Measurement(float("nan"), units="Hz", calibrated=calibrated)
    if n < 32:
        return nan, Measurement(float("nan"), units="Hz/50us", calibrated=calibrated), np.zeros(3)

    resid = sym_freq[:n] - ideal_symbol_frequency(bits[:n], deviation_hz)
    t_us = np.arange(n) * (1e6 / BLE_SYMBOL_RATE)

    # Quadratic fit; the residual scatter sets the uncertainty on the endpoints.
    coeffs = np.polyfit(t_us, resid, 2)
    fitted = np.polyval(coeffs, t_us)
    scatter = float(np.std(resid - fitted))

    total = float(fitted[-1] - fitted[0])
    drift = Measurement(total, scatter * np.sqrt(2), "Hz", calibrated, *SPEC_LIMITS["drift_hz"])

    # Max change over any 50 us window, evaluated on the smooth fit so that
    # per-symbol noise does not masquerade as drift rate.
    step = max(int(50.0 * BLE_SYMBOL_RATE / 1e6), 1)
    if n > step:
        rates = fitted[step:] - fitted[:-step]
        worst = float(rates[np.argmax(np.abs(rates))])
    else:
        worst = float("nan")
    rate = Measurement(
        worst, scatter * np.sqrt(2), "Hz/50us", calibrated,
        *SPEC_LIMITS["drift_rate_hz_per_50us"]
    )
    return drift, rate, coeffs


def phase_noise_psd(
    iq: np.ndarray,
    sample_rate: float,
    offsets_hz: tuple[float, ...] = (10e3, 50e3, 100e3, 500e3),
    calibrated: bool = False,
) -> dict[str, Measurement]:
    """Residual phase-noise PSD sampled at fixed offsets, in dBc/Hz.

    Measured on the unwrapped phase after removing a low-order polynomial, so
    the modulation and the carrier offset are both suppressed and what remains
    is oscillator noise plus receiver noise.
    """
    out: dict[str, Measurement] = {}
    if iq.size < 512:
        for f in offsets_hz:
            out[f"phase_noise_{int(f/1e3)}kHz"] = Measurement(
                float("nan"), units="dBc/Hz", calibrated=calibrated
            )
        return out

    phase = np.unwrap(np.angle(iq.astype(np.complex128)))
    t = np.arange(len(phase))
    phase = phase - np.polyval(np.polyfit(t, phase, 3), t)

    nper = min(512, len(phase))
    f, pxx = sps_signal.welch(
        phase, fs=sample_rate, nperseg=nper, noverlap=nper // 2, scaling="density"
    )
    for target in offsets_hz:
        key = f"phase_noise_{int(target/1e3)}kHz"
        if target >= sample_rate / 2:
            out[key] = Measurement(float("nan"), units="dBc/Hz", calibrated=calibrated)
            continue
        i = int(np.argmin(np.abs(f - target)))
        val = 10.0 * np.log10(max(pxx[i], 1e-20))
        # Welch with K averages has roughly 4.34/sqrt(K) dB of scatter.
        k = max(len(phase) // (nper // 2) - 1, 1)
        out[key] = Measurement(val, 4.34 / np.sqrt(k), "dBc/Hz", calibrated)
    return out


def lo_leakage_and_image(iq: np.ndarray, sample_rate: float, cfo_hz: float) -> tuple[Measurement, Measurement]:
    """Transmitter LO leakage at DC and image-rejection ratio, both in dBc.

    Only meaningful after the receiver's own DC offset and quadrature imbalance
    have been corrected -- otherwise this measures the bladeRF, not the target.
    """
    nan = (Measurement(float("nan"), units="dBc"), Measurement(float("nan"), units="dBc"))
    if iq.size < 256:
        return nan
    n = int(2 ** np.floor(np.log2(iq.size)))
    x = iq[:n].astype(np.complex128) * np.hanning(n)
    spec = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))
    total = float(spec.sum())
    if total <= 0:
        return nan

    # LO leakage: energy in a narrow bin at baseband DC.
    bw = max(sample_rate / n * 3, 20e3)
    dc_band = np.abs(freqs) < bw
    leak = 10.0 * np.log10(max(float(spec[dc_band].sum()) / total, 1e-15))

    # Image rejection: signal sits near +cfo plus deviation; compare the
    # occupied band with its mirror image about DC.
    sig = (freqs > cfo_hz - 400e3) & (freqs < cfo_hz + 400e3)
    img = (freqs > -cfo_hz - 400e3) & (freqs < -cfo_hz + 400e3)
    p_sig = float(spec[sig].sum())
    p_img = float(spec[img].sum())
    irr = 10.0 * np.log10(max(p_sig, 1e-20) / max(p_img, 1e-20))
    return (
        Measurement(leak, 1.0, "dBc"),
        Measurement(irr, 1.0, "dB"),
    )


# --------------------------------------------------------------------------
# modulation quality
# --------------------------------------------------------------------------

def deviation_estimates(
    sym_freq: np.ndarray, bits: np.ndarray, cfo_hz: float
) -> tuple[Measurement, Measurement, Measurement]:
    """Mean deviation for ones, for zeros, and their (unidentifiable) asymmetry.

    The asymmetry is reported for the detail tree but is deliberately NOT part
    of the clustering feature vector, because it is exactly degenerate with the
    carrier offset.  Writing the instantaneous frequency as f = c + g with g in
    {+A, -B}, the substitution c' = c + (A-B)/2, g' in {+(A+B)/2, -(A+B)/2}
    produces a bit-for-bit identical waveform.  No estimator can separate the
    two from the received signal alone, at any SNR, with or without a GPSDO --
    so an asymmetry "measurement" is really a restatement of whatever the
    carrier-offset estimator happened to choose.

    `transition_asymmetry` measures the modulator rail imbalance in a way that
    is identifiable, and that is what goes into the feature vector.
    """
    n = min(len(sym_freq), len(bits))
    nan = Measurement(float("nan"), units="Hz")
    if n < 16:
        return nan, nan, Measurement(float("nan"), units="")

    settled = settled_symbol_mask(bits[:n])
    f = sym_freq[:n] - cfo_hz
    b = bits[:n]
    ones = f[settled & (b == 1)]
    zeros = f[settled & (b == 0)]
    if ones.size < 3 or zeros.size < 3:
        return nan, nan, Measurement(float("nan"), units="")

    dev_one = Measurement(float(np.mean(ones)), _sem(ones), "Hz", True, *SPEC_LIMITS["freq_dev_hz"])
    dev_zero = Measurement(
        float(-np.mean(zeros)), _sem(zeros), "Hz", True, *SPEC_LIMITS["freq_dev_hz"]
    )
    mean_dev = 0.5 * (dev_one.value + dev_zero.value)
    if mean_dev <= 0:
        asym = Measurement(float("nan"), units="")
    else:
        val = (dev_one.value - dev_zero.value) / mean_dev
        unc = np.hypot(dev_one.uncertainty, dev_zero.uncertainty) / mean_dev
        asym = Measurement(val, unc, "")
    return dev_one, dev_zero, asym


def transition_asymmetry(
    freq: np.ndarray, bits: np.ndarray, sps: float, start: float, sample_rate: float
) -> Measurement:
    """Rise/fall duration imbalance of the frequency trajectory.

    Unlike deviation asymmetry this is identifiable: adding a constant to f(t)
    shifts both transitions equally and leaves their durations untouched, so
    the carrier offset cancels exactly.  A modulator whose two rails have
    different slew behaviour shows up here and nowhere else.

    Returns (t_rise - t_fall) / (t_rise + t_fall).
    """
    n = len(bits)
    if n < 16 or freq.size < 32:
        return Measurement(float("nan"), units="")

    rises: list[float] = []
    falls: list[float] = []
    for k in range(2, n - 2):
        if bits[k] == bits[k - 1]:
            continue
        if not (bits[k - 1] == bits[k - 2] and bits[k] == bits[k + 1]):
            continue  # isolated transitions only
        i0 = int(round(start + (k - 1) * sps))
        i1 = int(round(start + (k + 1) * sps))
        if i0 < 0 or i1 >= freq.size or i1 - i0 < 4:
            continue
        dur = _transition_duration(freq[i0:i1], sample_rate)
        if not np.isfinite(dur) or dur <= 0:
            continue
        (rises if bits[k] == 1 else falls).append(dur)

    if len(rises) < 2 or len(falls) < 2:
        return Measurement(float("nan"), units="")
    tr, tf = float(np.mean(rises)), float(np.mean(falls))
    if tr + tf <= 0:
        return Measurement(float("nan"), units="")
    val = (tr - tf) / (tr + tf)
    unc = np.hypot(_sem(np.array(rises)), _sem(np.array(falls))) / (tr + tf)
    return Measurement(val, float(unc), "")


def modulation_index(dev_one: Measurement, dev_zero: Measurement) -> Measurement:
    """h = 2 * f_dev / symbol_rate, from the mean of the two rails."""
    if not (np.isfinite(dev_one.value) and np.isfinite(dev_zero.value)):
        return Measurement(float("nan"), units="")
    dev = 0.5 * (dev_one.value + dev_zero.value)
    unc = 0.5 * np.hypot(dev_one.uncertainty, dev_zero.uncertainty)
    return Measurement(
        2.0 * dev / BLE_SYMBOL_RATE,
        2.0 * unc / BLE_SYMBOL_RATE,
        "",
        True,
        *SPEC_LIMITS["modulation_index"],
    )


@dataclass
class TransitionMetrics:
    """Per-transition timing, measured once and shared by three features.

    `effective_bt`, `transition_asymmetry` and `symbol_clock_offset` all need
    the same thing -- where each symbol transition actually crossed, and how
    long it took -- and each walked the transitions in its own Python loop.
    Together that was over half the per-packet feature cost.  This computes all
    of them in one vectorised pass over a 2D window array.
    """

    index: np.ndarray  # symbol index of each transition
    rising: np.ndarray  # True where the transition goes 0 -> 1
    isolated: np.ndarray  # True where the neighbours are settled either side
    crossing: np.ndarray  # sub-sample time of the mid-level crossing
    expected: np.ndarray  # where the nominal symbol clock puts that crossing
    duration: np.ndarray  # 10-90% transition time in seconds (NaN if unclean)


def _empty_metrics() -> "TransitionMetrics":
    z = np.zeros(0)
    return TransitionMetrics(
        z.astype(int), z.astype(bool), z.astype(bool), z, z, z
    )


def _first_crossing(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised first upward zero crossing of every row of `d`."""
    if d.shape[0] == 0 or d.shape[1] < 2:
        return np.zeros(d.shape[0]), np.zeros(d.shape[0], dtype=bool)
    up = (d[:, :-1] < 0) & (d[:, 1:] >= 0)
    valid = up.any(axis=1)
    j = up.argmax(axis=1)
    rows = np.arange(d.shape[0])
    a = d[rows, j]
    b = d[rows, j + 1]
    den = b - a
    ok = valid & (den != 0)
    frac = np.zeros(d.shape[0])
    np.divide(-a, den, out=frac, where=ok)
    return j + frac, ok


def transition_metrics(
    freq: np.ndarray,
    bits: np.ndarray,
    sps: float,
    start: float,
    sample_rate: float,
    level: float = 0.0,
) -> TransitionMetrics:
    """Locate and time every symbol transition in one vectorised pass."""
    n = len(bits)
    if n < 8 or freq.size < 8:
        return _empty_metrics()

    k = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    k = k[(k >= 2) & (k < n - 2)]
    if k.size == 0:
        return _empty_metrics()

    rising = bits[k] == 1
    isolated = (bits[k - 1] == bits[k - 2]) & (bits[k] == bits[k + 1])

    expected = start + (k - 0.5) * sps
    # A window of two symbol periods centred on the transition.  One period is
    # too narrow: the 10-90% levels are taken from the window's own min and max,
    # so a window that barely spans the transition clips the excursion and
    # biases the duration.
    i0 = np.floor(expected - sps).astype(np.int64)
    width = int(np.ceil(2 * sps)) + 2
    idx = i0[:, None] + np.arange(width)[None, :]

    inside = (idx[:, 0] >= 0) & (idx[:, -1] < freq.size)
    if not inside.any():
        return _empty_metrics()
    k, rising, isolated = k[inside], rising[inside], isolated[inside]
    expected, i0, idx = expected[inside], i0[inside], idx[inside]

    seg = freq[idx].astype(np.float64)
    # Orient every transition upward so one crossing routine serves both.
    orient = np.where(rising, 1.0, -1.0)[:, None]
    up_seg = seg * orient

    # Mid-level crossing, taken against the global level (the carrier offset)
    # rather than a per-window midpoint -- see symbol_clock_from.
    pos, ok = _first_crossing(up_seg - level * orient)
    crossing = np.where(ok, i0 + pos, np.nan)

    lo = up_seg.min(axis=1)
    hi = up_seg.max(axis=1)
    span = hi - lo
    good = span >= 50e3
    p10, ok10 = _first_crossing(up_seg - (lo + 0.1 * span)[:, None])
    p90, ok90 = _first_crossing(up_seg - (lo + 0.9 * span)[:, None])
    dur = np.where(good & ok10 & ok90, np.abs(p90 - p10) / sample_rate, np.nan)

    return TransitionMetrics(k, rising, isolated, crossing, expected, dur)


def effective_bt_from(tm: TransitionMetrics) -> Measurement:
    """Bandwidth-time product inferred from the measured 10-90% slope.

    The 10-90% transition time is not exactly proportional to 1/BT but is very
    close to affine in it.  Fitted against the synthetic generator over
    BT = 0.30..0.80 at 35 dB SNR, the relation holds to better than 0.01.
    Outside that range the value is extrapolation and is clipped.
    """
    if tm.duration.size == 0:
        return Measurement(float("nan"), units="")
    d = tm.duration[tm.isolated]
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return Measurement(float("nan"), units="")
    t_mean = float(np.mean(d))
    inv = 1.0 / max(t_mean * BLE_SYMBOL_RATE, 1e-9)
    bt = BT_FIT_SLOPE * inv + BT_FIT_INTERCEPT
    unc = BT_FIT_SLOPE * inv * (_safe_std(d) / max(t_mean, 1e-12)) / np.sqrt(d.size)
    return Measurement(float(np.clip(bt, 0.05, 5.0)), float(unc), "")


def transition_asymmetry_from(tm: TransitionMetrics) -> Measurement:
    """Rise/fall duration imbalance of the frequency trajectory.

    Unlike deviation asymmetry this is identifiable: adding a constant to f(t)
    shifts both transitions equally and leaves their durations untouched, so
    the carrier offset cancels exactly.  A modulator whose two rails slew at
    different rates shows up here and nowhere else.
    """
    if tm.duration.size == 0:
        return Measurement(float("nan"), units="")
    sel = tm.isolated & np.isfinite(tm.duration) & (tm.duration > 0)
    rises = tm.duration[sel & tm.rising]
    falls = tm.duration[sel & ~tm.rising]
    if rises.size < 2 or falls.size < 2:
        return Measurement(float("nan"), units="")
    tr, tf = float(np.mean(rises)), float(np.mean(falls))
    if tr + tf <= 0:
        return Measurement(float("nan"), units="")
    unc = np.hypot(_sem(rises), _sem(falls)) / (tr + tf)
    return Measurement((tr - tf) / (tr + tf), float(unc), "")


def symbol_clock_from(
    tm: TransitionMetrics, sample_rate: float
) -> tuple[Measurement, Measurement]:
    """Symbol-rate error in ppm and timing jitter, from transition crossings.

    Scanning the sampling phase segment by segment cannot resolve this: a
    40 ppm symbol-rate error accumulates only about 0.1 samples across a whole
    advertising packet, an order of magnitude below the step of any practical
    phase search.  Interpolated crossings measure the same drift to a
    hundredth of a sample, and there are enough of them per packet for the fit
    to average down to single-digit ppm on a long payload.
    """
    nan = Measurement(float("nan"), units="ppm")
    ok = np.isfinite(tm.crossing)
    if ok.sum() < 12:
        return nan, Measurement(float("nan"), units="ps")
    e = tm.expected[ok]
    err = tm.crossing[ok] - e
    slope, intercept = np.polyfit(e, err, 1)
    resid = err - (slope * e + intercept)
    # Positive means the transmitter's symbol clock runs fast, matching the
    # sign convention used for the carrier offset; the fit yields a period error.
    ppm = float(-slope * 1e6)
    spread = float(np.std(e))
    unc = float(_safe_std(resid) / max(spread * np.sqrt(e.size), 1e-9) * 1e6)
    jitter_ps = float(_safe_std(resid) / sample_rate * 1e12)
    return Measurement(ppm, unc, "ppm"), Measurement(jitter_ps, float("nan"), "ps")


def _transition_duration(seg: np.ndarray, sample_rate: float) -> float:
    """10-90% duration of one frequency transition, to sub-sample resolution.

    Counting whole samples between the 10% and 90% crossings is far too coarse:
    at 8 samples/symbol a BLE transition spans only three or four of them, so a
    whole-sample duration takes a handful of discrete values and any feature
    derived from it (effective BT, rise/fall asymmetry) quantises into steps
    bigger than the effect being measured.  Linear interpolation across the
    crossing samples recovers roughly a hundredth of a sample.

    Returns NaN when the segment does not contain a clean monotonic transition.
    """
    seg = np.asarray(seg, dtype=np.float64)
    if seg.size < 4:
        return float("nan")
    lo, hi = float(seg.min()), float(seg.max())
    span = hi - lo
    if span < 50e3:
        return float("nan")
    t10, t90 = lo + 0.1 * span, lo + 0.9 * span
    rising = int(np.argmax(seg)) > int(np.argmin(seg))

    def cross(level: float) -> float:
        d = seg - level
        if rising:
            idx = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
        else:
            idx = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0))
        if idx.size == 0:
            return float("nan")
        j = int(idx[0] if rising else idx[-1])
        den = d[j + 1] - d[j]
        if den == 0:
            return float("nan")
        return j - d[j] / den

    a, b = cross(t10), cross(t90)
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return abs(b - a) / sample_rate


def effective_bt(freq: np.ndarray, sample_rate: float, bits: np.ndarray, sps: float,
                 start: float) -> Measurement:
    """Bandwidth-time product inferred from measured 10-90% transition slope.

    For Gaussian shaping the transition time between opposite symbols scales
    roughly as 1/BT; this inverts that relation on isolated transitions.
    """
    n = len(bits)
    if n < 16 or freq.size < 32:
        return Measurement(float("nan"), units="")

    trans = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    slopes = []
    for k in trans:
        if k < 2 or k >= n - 2:
            continue
        # isolated transition only: ...a a b b...
        if not (bits[k - 1] == bits[k - 2] and bits[k] == bits[k + 1]):
            continue
        i0 = int(round(start + (k - 1) * sps))
        i1 = int(round(start + (k + 1) * sps))
        if i0 < 0 or i1 >= freq.size or i1 - i0 < 4:
            continue
        t_rise = _transition_duration(freq[i0:i1], sample_rate)
        if np.isfinite(t_rise) and t_rise > 0:
            slopes.append(t_rise)
    if not slopes:
        return Measurement(float("nan"), units="")
    t_mean = float(np.mean(slopes))
    # The 10-90% transition time is not exactly proportional to 1/BT, but it is
    # very close to affine in it.  Fitted against the synthetic generator over
    # BT = 0.30..0.80 with sub-sample crossing interpolation, the relation
    #     BT = 0.4035 / (t_1090 * Rs) - 0.1253
    # holds to better than 0.02 across that whole range.  Outside it the fit is
    # extrapolation and the value is clipped.
    inv = 1.0 / max(t_mean * BLE_SYMBOL_RATE, 1e-9)
    bt = BT_FIT_SLOPE * inv + BT_FIT_INTERCEPT
    unc = 0.40347 * inv * (_safe_std(np.array(slopes)) / max(t_mean, 1e-12)) / np.sqrt(
        len(slopes)
    )
    return Measurement(float(np.clip(bt, 0.05, 5.0)), float(unc), "")


def frequency_error(
    sym_freq: np.ndarray, bits: np.ndarray, deviation_hz: float, cfo_hz: float
) -> tuple[Measurement, Measurement]:
    """RMS and peak per-symbol frequency error against the ideal trajectory."""
    n = min(len(sym_freq), len(bits))
    if n < 8:
        return Measurement(float("nan"), units="Hz"), Measurement(float("nan"), units="Hz")
    err = sym_freq[:n] - cfo_hz - ideal_symbol_frequency(bits[:n], deviation_hz)
    settled = settled_symbol_mask(bits[:n])
    e = err[settled] if settled.sum() >= 8 else err
    rms = float(np.sqrt(np.mean(e**2)))
    peak = float(np.max(np.abs(e)))
    return (
        Measurement(rms, rms / np.sqrt(2 * max(e.size, 1)), "Hz"),
        Measurement(peak, float("nan"), "Hz"),
    )


def symbol_clock_offset(
    freq: np.ndarray, bits: np.ndarray, sps: float, start: float, sample_rate: float,
    level: float = 0.0,
) -> tuple[Measurement, Measurement]:
    """Symbol-rate error in ppm and timing jitter, estimated independently of CFO.

    The symbol clock and the carrier usually come from the same crystal, but not
    always -- a transmitter that synthesises them separately shows a mismatch
    that is a strong identifier.  Estimated by finding the sampling phase that
    maximises the eye in successive segments and fitting the drift.
    """
    n = len(bits)
    nan = Measurement(float("nan"), units="ppm")
    if n < 64 or freq.size < 64:
        return nan, Measurement(float("nan"), units="ps")

    # Scanning the sampling phase segment by segment cannot resolve this: a
    # 40 ppm symbol-rate error accumulates only 0.12 samples of drift across a
    # whole 376-symbol advertising packet, an order of magnitude below the step
    # of any practical phase search.  Interpolated transition crossings measure
    # the same drift to a hundredth of a sample and there are ~180 of them per
    # packet, so the fit averages down to single-digit ppm.
    crossings: list[float] = []
    expected: list[float] = []
    for k in range(1, n):
        if bits[k] == bits[k - 1]:
            continue
        centre = start + (k - 0.5) * sps
        i0 = int(np.floor(centre - sps / 2))
        i1 = int(np.ceil(centre + sps / 2))
        if i0 < 0 or i1 >= freq.size or i1 - i0 < 3:
            continue
        # The decision level must be global (the carrier offset), not the
        # midpoint of this window.  A per-window min/max level is measured from
        # a window positioned by the *nominal* clock, so as the true clock
        # drifts away the level moves with it and drags the crossing estimate
        # back toward the window centre -- compressing the measured drift by
        # about 20% and making a 80 ppm error read as 60.
        seg = freq[i0:i1].astype(np.float64)
        rising = bits[k] == 1
        s = seg - level
        if rising:
            idx = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
        else:
            idx = np.flatnonzero((s[:-1] > 0) & (s[1:] <= 0))
        if idx.size != 1:
            continue  # ambiguous crossing: skip rather than guess
        j = int(idx[0])
        denom = s[j + 1] - s[j]
        if denom == 0:
            continue
        frac = -s[j] / denom
        crossings.append(i0 + j + frac)
        expected.append(centre)

    if len(crossings) < 12:
        return nan, Measurement(float("nan"), units="ps")

    c = np.asarray(crossings)
    e = np.asarray(expected)
    err = c - e  # timing error in samples versus the nominal symbol clock
    slope, intercept = np.polyfit(e, err, 1)  # samples of error per sample of time
    resid = err - (slope * e + intercept)

    # Reported as a symbol *rate* error, matching the sign convention used for
    # the carrier offset: a positive value means the transmitter's symbol clock
    # runs fast.  The fitted slope is a period error, hence the negation.
    ppm = float(-slope * 1e6)
    spread = float(np.std(e))
    unc = float(_safe_std(resid) / max(spread * np.sqrt(len(e)), 1e-9) * 1e6)
    jitter_ps = float(_safe_std(resid) / sample_rate * 1e12)
    return (
        Measurement(ppm, unc, "ppm"),
        Measurement(jitter_ps, float("nan"), "ps"),
    )


def eye_and_isi(sym_freq: np.ndarray, bits: np.ndarray, cfo_hz: float,
                deviation_hz: float) -> tuple[Measurement, Measurement]:
    """Eye opening (normalised) and residual ISI."""
    n = min(len(sym_freq), len(bits))
    if n < 16 or deviation_hz <= 0:
        return Measurement(float("nan"), units=""), Measurement(float("nan"), units="")
    f = (sym_freq[:n] - cfo_hz) / deviation_hz
    b = bits[:n]
    ones, zeros = f[b == 1], f[b == 0]
    if ones.size < 4 or zeros.size < 4:
        return Measurement(float("nan"), units=""), Measurement(float("nan"), units="")
    opening = float(np.min(ones) - np.max(zeros))

    settled = settled_symbol_mask(b)
    if settled.sum() >= 8:
        ref_one = float(np.mean(f[settled & (b == 1)]))
        ref_zero = float(np.mean(f[settled & (b == 0)]))
        ideal = np.where(b == 1, ref_one, ref_zero)
        isi = float(np.sqrt(np.mean((f - ideal) ** 2)))
    else:
        isi = float("nan")
    return Measurement(opening, float("nan"), ""), Measurement(isi, float("nan"), "")


# --------------------------------------------------------------------------
# envelope and transient
# --------------------------------------------------------------------------

def envelope_features(
    iq: np.ndarray, sample_rate: float, burst_start: int, burst_end: int,
    resample_to: int = 32,
) -> dict:
    """PA turn-on and turn-off shape.

    The retained slice starts 50 us before the preamble specifically so that the
    whole turn-on transient is inside it.  The ramp is where a power amplifier
    is least like its neighbours: the same part number from the same reel still
    differs in bias settling by tens of nanoseconds.
    """
    out = {
        "rise_time_us": Measurement(float("nan"), units="us"),
        "overshoot": Measurement(float("nan"), units=""),
        "fall_time_us": Measurement(float("nan"), units="us"),
        "ramp_vector": np.zeros(resample_to, dtype=np.float32),
    }
    if iq.size < 64:
        return out

    env = np.abs(iq).astype(np.float64)
    # Smooth over roughly a quarter symbol: enough to kill noise, short enough
    # to leave a 1-2 us ramp intact.
    k = max(int(sample_rate / 4e6), 1)
    if k > 1:
        env = np.convolve(env, np.ones(k) / k, mode="same")

    b0 = max(min(burst_start, len(env) - 2), 0)
    b1 = max(min(burst_end, len(env)), b0 + 2)
    steady = float(np.median(env[b0:b1])) if b1 > b0 else 0.0
    floor = float(np.median(env[: max(b0 // 2, 1)])) if b0 > 8 else 0.0
    span = steady - floor
    if span <= 0:
        return out

    lo_t, hi_t = floor + 0.1 * span, floor + 0.9 * span

    # --- rise ---------------------------------------------------------
    search0 = max(b0 - int(20e-6 * sample_rate), 0)
    seg = env[search0:b0 + max(int(10e-6 * sample_rate), 4)]
    if seg.size > 4:
        above = np.flatnonzero(seg >= hi_t)
        below = np.flatnonzero(seg <= lo_t)
        if above.size and below.size and below[0] < above[0]:
            i_lo = below[below < above[0]][-1]
            i_hi = above[0]
            out["rise_time_us"] = Measurement(
                (i_hi - i_lo) / sample_rate * 1e6, 1.0 / sample_rate * 1e6, "us"
            )
            # overshoot in the 5 us following the ramp
            os_end = min(i_hi + int(5e-6 * sample_rate), seg.size)
            if os_end > i_hi + 2:
                peak = float(np.max(seg[i_hi:os_end]))
                out["overshoot"] = Measurement((peak - steady) / span, float("nan"), "")

    # --- fall ---------------------------------------------------------
    tail = env[b1 - 2 :]
    if tail.size > 4:
        below = np.flatnonzero(tail <= lo_t)
        above = np.flatnonzero(tail >= hi_t)
        if below.size and above.size and above[0] < below[-1]:
            i_hi = above[0]
            after = below[below > i_hi]
            if after.size:
                out["fall_time_us"] = Measurement(
                    (after[0] - i_hi) / sample_rate * 1e6, 1.0 / sample_rate * 1e6, "us"
                )

    # --- normalised ramp vector, for clustering ------------------------
    r0 = max(b0 - int(10e-6 * sample_rate), 0)
    r1 = min(b0 + int(10e-6 * sample_rate), len(env))
    if r1 - r0 > 4:
        seg = env[r0:r1]
        x = np.linspace(0, 1, len(seg))
        xi = np.linspace(0, 1, resample_to)
        v = np.interp(xi, x, seg)
        rng = v.max() - v.min()
        if rng > 0:
            out["ramp_vector"] = ((v - v.min()) / rng).astype(np.float32)
    return out


def spectral_splatter(iq: np.ndarray, sample_rate: float, burst_start: int) -> Measurement:
    """Out-of-band energy during the turn-on window, relative to in-band.

    A transmitter that keys its PA abruptly splatters across neighbouring
    channels; the amount is a device characteristic and also an interference
    concern in its own right.
    """
    n_win = int(10e-6 * sample_rate)
    lo = max(burst_start - n_win // 2, 0)
    hi = min(lo + n_win, iq.size)
    if hi - lo < 64:
        return Measurement(float("nan"), units="dB")
    seg = iq[lo:hi].astype(np.complex128)
    n = int(2 ** np.floor(np.log2(seg.size)))
    seg = seg[:n] * np.hanning(n)
    spec = np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))
    inband = np.abs(freqs) <= 500e3
    outband = np.abs(freqs) > 1e6
    p_in = float(spec[inband].sum())
    p_out = float(spec[outband].sum())
    if p_in <= 0:
        return Measurement(float("nan"), units="dB")
    return Measurement(10.0 * np.log10(max(p_out, 1e-20) / p_in), 1.0, "dB")


# --------------------------------------------------------------------------
# amplitude and multipath
# --------------------------------------------------------------------------

def rssi(iq: np.ndarray, burst_start: int, burst_end: int, gain_db: float,
         cal_offset_db: float | None = None) -> tuple[Measurement, Measurement]:
    """Mean burst power in dBFS, and in dBm when a calibration table exists."""
    b0 = max(min(burst_start, iq.size - 2), 0)
    b1 = max(min(burst_end, iq.size), b0 + 2)
    p = float(np.mean(np.abs(iq[b0:b1]) ** 2))
    if p <= 0:
        nan = Measurement(float("nan"), units="dBFS")
        return nan, Measurement(float("nan"), units="dBm")
    dbfs = 10.0 * np.log10(p)
    m_dbfs = Measurement(dbfs, 0.5, "dBFS")
    if cal_offset_db is None:
        return m_dbfs, Measurement(float("nan"), units="dBm", calibrated=False)
    return m_dbfs, Measurement(dbfs - gain_db + cal_offset_db, 2.0, "dBm", True)


def snr(iq: np.ndarray, burst_start: int, burst_end: int) -> Measurement:
    """Burst power over the noise measured in the pre-trigger window."""
    b0 = max(min(burst_start, iq.size - 2), 0)
    b1 = max(min(burst_end, iq.size), b0 + 2)
    if b0 < 32:
        return Measurement(float("nan"), units="dB")
    n = float(np.mean(np.abs(iq[: b0 - 8]) ** 2))
    s = float(np.mean(np.abs(iq[b0:b1]) ** 2))
    if n <= 0 or s <= n:
        return Measurement(float("nan"), units="dB")
    return Measurement(10.0 * np.log10((s - n) / n), 1.0, "dB")


def reconstruct_ideal(
    bits: np.ndarray, sps: float, deviation_hz: float, cfo_hz: float,
    sample_rate: float, bt: float = 0.5,
) -> np.ndarray:
    """Rebuild the transmitted waveform from decoded bits, for use as a reference."""
    n = len(bits)
    total = int(np.ceil(n * sps)) + 8
    up = np.zeros(total)
    idx = np.clip(np.rint(np.arange(n) * sps).astype(int), 0, total - 1)
    np.add.at(up, idx, bits.astype(np.float64) * 2 - 1)

    span = int(round(4 * sps)) | 1
    t = (np.arange(span) - (span - 1) / 2.0) / sps
    h = np.exp(-2.0 * (np.pi**2) * (bt**2) * (t**2) / np.log(2.0))
    h /= h.sum()
    shaped = np.convolve(up, h * sps, mode="same")

    freq = shaped * deviation_hz + cfo_hz
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    return np.exp(1j * phase).astype(np.complex64)


def multipath_profile(
    iq: np.ndarray, ideal: np.ndarray, sample_rate: float, max_delay_us: float = 3.0
) -> tuple[Measurement, np.ndarray]:
    """RMS delay spread and the delay-power profile.

    Correlating the received burst against the waveform rebuilt from its own
    decoded bits gives the channel impulse response.  Unlike carrier offset this
    is available from a single channel, and for a device known to be stationary
    a step change in the profile is strong evidence that the transmitter moved
    -- or that a different transmitter is using the address.
    """
    nan = Measurement(float("nan"), units="us")
    if iq.size < 128 or ideal.size < 128:
        return nan, np.zeros(0, dtype=np.float32)

    n = min(iq.size, ideal.size)
    a = iq[:n].astype(np.complex128)
    b = ideal[:n].astype(np.complex128)
    a = a - a.mean()
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    corr = np.fft.ifft(np.fft.fft(a, nfft) * np.conj(np.fft.fft(b, nfft)))
    power = np.abs(corr) ** 2

    max_lag = max(int(max_delay_us * 1e-6 * sample_rate), 4)
    prof = power[: max_lag].astype(np.float64)
    if prof.max() <= 0:
        return nan, np.zeros(0, dtype=np.float32)

    peak = int(np.argmax(prof))
    prof = prof / prof.max()
    # Only components above -20 dB contribute; below that it is noise.
    mask = prof > 0.01
    lags = (np.arange(len(prof)) - peak) / sample_rate * 1e6
    w = prof[mask]
    l = lags[mask]
    if w.sum() <= 0:
        return nan, prof.astype(np.float32)
    mean_delay = float(np.sum(w * l) / np.sum(w))
    rms = float(np.sqrt(np.sum(w * (l - mean_delay) ** 2) / np.sum(w)))
    return Measurement(rms, float("nan"), "us"), prof.astype(np.float32)


def antenna_phase_difference(
    iq0: np.ndarray, iq1: np.ndarray, burst_start: int, burst_end: int,
    calibration_rad: float = 0.0, spacing_wavelengths: float = 0.5,
) -> tuple[Measurement, Measurement]:
    """Inter-antenna phase difference and the AoA it implies.

    Only meaningful with `--dual-antenna`, and only after the per-channel phase
    offset has been calibrated out -- the two RX chains share an LO but not a
    signal path, and their fixed offset is tens of degrees.
    """
    nan = Measurement(float("nan"), units="deg")
    if iq0.size == 0 or iq1.size == 0:
        return nan, nan
    n = min(iq0.size, iq1.size, burst_end) - burst_start
    if n < 32:
        return nan, nan
    a = iq0[burst_start : burst_start + n].astype(np.complex128)
    b = iq1[burst_start : burst_start + n].astype(np.complex128)
    prod = np.mean(a * np.conj(b))
    dphi = float(np.angle(prod)) - calibration_rad
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    phase = Measurement(np.degrees(dphi), float("nan"), "deg")

    arg = dphi / (2 * np.pi * spacing_wavelengths)
    if abs(arg) > 1:
        return phase, Measurement(float("nan"), units="deg")
    return phase, Measurement(float(np.degrees(np.arcsin(arg))), float("nan"), "deg")


# --------------------------------------------------------------------------
# the aggregate
# --------------------------------------------------------------------------

# Order matters: this is the feature vector used for clustering and anomaly
# scoring, and the baselines are stored against these names.
FEATURE_VECTOR_KEYS = (
    "cfo_ppm",
    "modulation_index",
    # dev_asymmetry is deliberately absent: it is exactly degenerate with
    # cfo_ppm (see deviation_estimates).  transition_asymmetry measures the
    # same physical rail imbalance in an identifiable way.
    "transition_asymmetry",
    "freq_error_rms",
    "drift_hz",
    "drift_rate",
    "symbol_clock_ppm",
    "rise_time_us",
    "overshoot",
    "effective_bt",
    "eye_opening",
    "residual_isi",
    "delay_spread_us",
    "splatter_db",
)


@dataclass
class PacketFeatures:
    """Every per-packet feature, each with its uncertainty and calibration flag."""

    measurements: dict = field(default_factory=dict)
    ramp_vector: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.float32))
    delay_profile: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    drift_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    calibrated: bool = False
    full: bool = True  # False when spectral features were shed under load

    def get(self, key: str) -> Measurement:
        return self.measurements.get(key, Measurement(float("nan")))

    def value(self, key: str) -> float:
        return self.get(key).value

    def vector(self) -> np.ndarray:
        """The fixed-order feature vector used by clustering and anomaly scoring."""
        return np.array([self.value(k) for k in FEATURE_VECTOR_KEYS], dtype=np.float64)

    def as_row(self) -> dict:
        row = {}
        for k, m in self.measurements.items():
            row[k] = m.value
            row[k + "_unc"] = m.uncertainty
        row["calibrated"] = self.calibrated
        row["features_full"] = self.full
        return row


def extract_features(
    iq: np.ndarray,
    sample_rate: float,
    sync_offset: int,
    sym_offset: float,
    bits: np.ndarray,
    sym_freq: np.ndarray,
    gain_db: float = 0.0,
    calibrated: bool = False,
    carrier_hz: float = CARRIER_HZ_DEFAULT,
    rssi_cal_db: float | None = None,
    iq_second: np.ndarray | None = None,
    antenna_cal_rad: float = 0.0,
    full: bool = True,
) -> PacketFeatures:
    """Run every extractor over one retained packet slice.

    `iq` is the unfiltered retained window; `sync_offset` is the index of the
    first preamble symbol within it.  `sym_freq` is the already-sliced
    per-symbol frequency, reused rather than recomputed.
    """
    from .dsp import apply_channel_filter, design_channel_filter, instantaneous_frequency

    sps = sample_rate / BLE_SYMBOL_RATE
    n_sym = min(len(bits), len(sym_freq))
    bits = bits[:n_sym]

    # Re-derive the per-symbol frequency from the unfiltered retained slice
    # through the wide measurement filter.  The demodulator's own symbol values
    # were taken through its narrow slicing filter and are biased low.
    meas_taps = design_channel_filter(sample_rate, MEASUREMENT_FILTER_HZ)
    meas = apply_channel_filter(iq, meas_taps)
    freq_full = instantaneous_frequency(meas, sample_rate)
    idx = np.rint(sync_offset + sym_offset + np.arange(n_sym) * sps).astype(np.int64)
    if freq_full.size and idx.max(initial=0) < freq_full.size and idx.min(initial=0) >= 0:
        sym_freq = freq_full[idx].astype(np.float64)
    else:
        sym_freq = np.asarray(sym_freq[:n_sym], dtype=np.float64)

    burst_start = int(sync_offset)
    burst_end = int(min(sync_offset + sym_offset + n_sym * sps, iq.size))

    m: dict[str, Measurement] = {}

    # --- a first deviation estimate, needed by the CFO estimator ----------
    coarse_dev = float(np.mean(np.abs(sym_freq - np.median(sym_freq)))) if n_sym else 250e3
    coarse_dev = float(np.clip(coarse_dev, 100e3, 400e3))

    cfo = carrier_frequency_offset(sym_freq, bits, coarse_dev, calibrated)
    # Re-estimate deviation with the offset removed, then the offset again: two
    # passes are enough, the two estimates are nearly orthogonal.
    dev_one, dev_zero, asym = deviation_estimates(sym_freq, bits, cfo.value)
    dev = 0.5 * (dev_one.value + dev_zero.value)
    if not np.isfinite(dev) or dev <= 0:
        dev = coarse_dev
    cfo = carrier_frequency_offset(sym_freq, bits, dev, calibrated)

    m["cfo_hz"] = cfo
    m["cfo_ppm"] = cfo_ppm(cfo, carrier_hz)
    n_pre = 8
    m["cfo_preamble_hz"] = initial_frequency_offset(sym_freq, bits, dev, n_pre, calibrated)
    if n_sym > 48:
        pay = carrier_frequency_offset(sym_freq[40:], bits[40:], dev, calibrated)
    else:
        pay = Measurement(float("nan"), units="Hz", calibrated=calibrated)
    m["cfo_payload_hz"] = pay

    drift, rate, coeffs = frequency_drift(sym_freq, bits, dev, calibrated)
    m["drift_hz"] = drift
    m["drift_rate"] = rate

    m["dev_one_hz"] = dev_one
    m["dev_zero_hz"] = dev_zero
    m["dev_asymmetry"] = asym
    m["modulation_index"] = modulation_index(dev_one, dev_zero)

    rms_err, peak_err = frequency_error(sym_freq, bits, dev, cfo.value)
    m["freq_error_rms"] = rms_err
    m["freq_error_peak"] = peak_err

    eye, isi = eye_and_isi(sym_freq, bits, cfo.value, dev)
    m["eye_opening"] = eye
    m["residual_isi"] = isi

    # --- things that need the sample-rate waveform ------------------------
    tm = transition_metrics(
        freq_full, bits, sps, burst_start + sym_offset, sample_rate, level=cfo.value
    )
    m["effective_bt"] = effective_bt_from(tm)
    m["transition_asymmetry"] = transition_asymmetry_from(tm)
    clk, jitter = symbol_clock_from(tm, sample_rate)
    m["symbol_clock_ppm"] = Measurement(
        clk.value, clk.uncertainty, "ppm", calibrated
    )
    m["symbol_jitter_ps"] = jitter

    burst = iq[burst_start:burst_end]
    # The spectral features below are the expensive third of the budget.  When
    # the pipeline is shedding load they are skipped and reported as NaN rather
    # than quietly approximated; `PacketFeatures.full` records which happened so
    # the GUI and the exports can say so instead of showing a blank as a zero.
    if full:
        m.update(phase_noise_psd(burst, sample_rate, calibrated=calibrated))
        leak, irr = lo_leakage_and_image(burst, sample_rate, cfo.value)
        m["lo_leakage_dbc"] = leak
        m["image_rejection_db"] = irr

    env = envelope_features(iq, sample_rate, burst_start, burst_end)
    m["rise_time_us"] = env["rise_time_us"]
    m["overshoot"] = env["overshoot"]
    m["fall_time_us"] = env["fall_time_us"]
    m["splatter_db"] = spectral_splatter(iq, sample_rate, burst_start)

    dbfs, dbm = rssi(iq, burst_start, burst_end, gain_db, rssi_cal_db)
    m["rssi_dbfs"] = dbfs
    m["rssi_dbm"] = dbm
    m["snr_db"] = snr(iq, burst_start, burst_end)

    if full:
        ideal = reconstruct_ideal(bits, sps, dev, cfo.value, sample_rate)
        spread, profile = multipath_profile(burst, ideal, sample_rate)
        m["delay_spread_us"] = spread
    else:
        profile = np.zeros(0, dtype=np.float32)
        m["delay_spread_us"] = Measurement(float("nan"), units="us")

    if iq_second is not None and iq_second.size:
        dphi, aoa = antenna_phase_difference(
            iq, iq_second, burst_start, burst_end, antenna_cal_rad
        )
        m["antenna_phase_deg"] = dphi
        m["aoa_deg"] = aoa

    return PacketFeatures(
        measurements=m,
        ramp_vector=env["ramp_vector"],
        delay_profile=profile,
        drift_coeffs=coeffs,
        calibrated=calibrated,
        full=full,
    )
