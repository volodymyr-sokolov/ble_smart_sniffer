"""Synthetic BLE GFSK generator with injectable, known-exact impairments.

This is the ground truth the estimators are graded against.  Every impairment
the feature extractor claims to measure -- carrier offset, drift, drift rate,
modulation index, deviation asymmetry, BT, symbol-clock error, PA ramp shape --
is generated here from a number the test already knows, so a test can assert on
the recovered value rather than on "it did not crash".

Nothing in this module touches a radio.  It produces complex baseband arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sniffer.channels import (
    ADV_ACCESS_ADDRESS,
    ADV_CRC_INIT,
    crc24_bytes,
    dewhiten,
)


@dataclass
class TxImpairments:
    """Ground-truth transmitter parameters.  All defaults are ideal/spec-centre."""

    cfo_hz: float = 0.0  # constant carrier frequency offset
    drift_hz: float = 0.0  # total linear drift from first to last symbol
    drift_curve_hz: float = 0.0  # additional quadratic term over the packet
    modulation_index: float = 0.5  # spec 0.45..0.55
    dev_asymmetry: float = 0.0  # (dev_one - dev_zero)/mean, fractional
    bt: float = 0.5  # Gaussian shaping bandwidth-time product
    slew_asymmetry: float = 0.0  # fractional BT difference between the two rails
    symbol_clock_ppm: float = 0.0  # symbol rate error, independent of CFO
    timing_jitter_ps: float = 0.0  # RMS per-symbol jitter
    phase_noise_dbc: float = -110.0  # flat phase-noise floor, dBc/Hz
    ramp_us: float = 2.0  # PA turn-on 10-90% rise time
    ramp_overshoot: float = 0.0  # fractional envelope overshoot after turn-on
    ramp_down_us: float = 2.0
    amplitude: float = 0.25  # peak |s|, i.e. roughly -12 dBFS
    lo_leakage_dbc: float = -60.0  # residual carrier at DC
    image_rejection_db: float = 45.0  # IQ imbalance expressed as image rejection
    snr_db: float = 30.0  # AWGN added after all of the above


def _byte_bits_lsb_first(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")


def build_adv_pdu(
    pdu_type: int = 0x00,  # ADV_IND
    tx_add_random: bool = True,
    adva: bytes = b"\xde\xad\xbe\xef\x12\x34",
    ad_payload: bytes = b"\x02\x01\x06",
) -> bytes:
    """Assemble an unwhitened advertising PDU (header + payload)."""
    if len(adva) != 6:
        raise ValueError("AdvA must be 6 bytes")
    payload = bytes(adva) + bytes(ad_payload)
    header0 = (pdu_type & 0x0F) | (0x40 if tx_add_random else 0x00)
    header1 = len(payload) & 0xFF
    return bytes((header0, header1)) + payload


def build_air_bytes(
    pdu: bytes,
    channel: int = 37,
    access_address: int = ADV_ACCESS_ADDRESS,
    crc_init: int = ADV_CRC_INIT,
    corrupt_bits: tuple[int, ...] = (),
) -> bytes:
    """Preamble + access address + whitened (PDU || CRC), exactly as transmitted."""
    crc = crc24_bytes(pdu, crc_init)
    body = bytes(pdu) + crc
    whitened = bytes(dewhiten(np.frombuffer(body, dtype=np.uint8), channel))

    aa = bytes(
        (
            access_address & 0xFF,
            (access_address >> 8) & 0xFF,
            (access_address >> 16) & 0xFF,
            (access_address >> 24) & 0xFF,
        )
    )
    preamble = b"\xaa" if (aa[0] & 1) == 0 else b"\x55"
    air = preamble + aa + whitened

    if corrupt_bits:
        bits = _byte_bits_lsb_first(air)
        for b in corrupt_bits:
            if 0 <= b < len(bits):
                bits[b] ^= 1
        air = np.packbits(bits, bitorder="little").tobytes()
    return air


def gaussian_pulse(bt: float, sps: float, span: int = 4) -> np.ndarray:
    """Normalised Gaussian pulse for a given BT and oversampling factor."""
    n = int(round(span * sps))
    if n % 2 == 0:
        n += 1
    t = (np.arange(n) - (n - 1) / 2.0) / sps  # in symbol periods
    if bt <= 0:
        h = np.zeros(n)
        h[n // 2] = 1.0
        return h
    alpha = np.sqrt(np.log(2.0) / 2.0) / bt
    h = np.exp(-(np.pi**2) * (t**2) / (alpha**2) * 2.0)
    # The exact constant matters less than the resulting -3 dB bandwidth; the
    # feature extractor measures BT from transition slope, and the test asserts
    # monotonicity in BT rather than an absolute value.
    h = np.exp(-2.0 * (np.pi**2) * (bt**2) * (t**2) / np.log(2.0))
    return h / h.sum()


def modulate(
    air_bytes: bytes,
    sample_rate: float = 8e6,
    imp: TxImpairments | None = None,
    pre_pad_us: float = 80.0,
    post_pad_us: float = 40.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Modulate `air_bytes` to complex baseband with the requested impairments.

    Returns (samples, truth) where `truth` records the exact injected values and
    the sample index of the first preamble symbol, so tests can measure error
    against the generator rather than against another estimator.
    """
    imp = imp or TxImpairments()
    rng = rng or np.random.default_rng(0)

    symbol_rate = 1e6 * (1.0 + imp.symbol_clock_ppm * 1e-6)
    sps = sample_rate / symbol_rate

    bits = _byte_bits_lsb_first(air_bytes)
    nrz = bits.astype(np.float64) * 2.0 - 1.0

    # --- shaped frequency, synthesised at exact fractional symbol times ----
    # Each symbol contributes a continuous frequency pulse evaluated at its own
    # fractional position, rather than being written into a sample-quantised NRZ
    # waveform and filtered afterwards.  That distinction is the whole point:
    # any construction that places symbol edges on whole samples destroys the
    # sub-sample impairments this generator exists to inject.  An 80 ppm
    # symbol-clock error accumulates about 0.1 samples over a short advertising
    # packet, so a quantised generator emits a bit-identical waveform for 0 and
    # 80 ppm and the estimator under test looks broken when it is correct.
    #
    # The GFSK frequency pulse is rect(T) convolved with a Gaussian, which has
    # the closed form Phi((u+0.5)/sigma) - Phi((u-0.5)/sigma) in symbol units.
    from scipy.special import ndtr

    nsym = len(nrz)
    if imp.timing_jitter_ps > 0:
        jitter = rng.normal(0.0, imp.timing_jitter_ps * 1e-12 * sample_rate, nsym)
    else:
        jitter = np.zeros(nsym)

    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * max(imp.bt, 1e-3))
    span = int(np.ceil(4 * sps))
    total = int(np.ceil(nsym * sps)) + 2 * span + 8
    shaped = np.zeros(total)
    for k in range(nsym):
        centre = k * sps + jitter[k] + span
        lo = max(int(np.floor(centre - span)), 0)
        hi = min(int(np.ceil(centre + span)), total)
        if hi <= lo:
            continue
        u = (np.arange(lo, hi) - centre) / sps
        shaped[lo:hi] += nrz[k] * (
            ndtr((u + 0.5) / sigma) - ndtr((u - 0.5) / sigma)
        )
    # trim the leading guard so sample `span` is the first symbol's centre
    shaped = shaped[span:]

    # --- asymmetric slew ---------------------------------------------------
    # Rail imbalance has to be modelled as a direction-dependent slew, not as a
    # per-symbol shaping difference.  Giving ones and zeros different Gaussian
    # widths does nothing measurable: every transition involves one symbol of
    # each rail, so the two effects cancel and rise and fall durations come out
    # identical.  An asymmetric first-order lag does change them, which is what
    # a modulator with unequal pull-up and pull-down actually does.
    if imp.slew_asymmetry != 0.0:
        tau0 = 0.30  # samples; small enough to leave BT essentially intact
        tau_up = tau0 * (1.0 - imp.slew_asymmetry / 2.0)
        tau_dn = tau0 * (1.0 + imp.slew_asymmetry / 2.0)
        a_up = 1.0 / (1.0 + max(tau_up, 1e-6))
        a_dn = 1.0 / (1.0 + max(tau_dn, 1e-6))
        y = np.empty_like(shaped)
        prev = shaped[0]
        for i in range(len(shaped)):
            x = shaped[i]
            prev = prev + (x - prev) * (a_up if x > prev else a_dn)
            y[i] = prev
        shaped = y

    # --- deviation, asymmetric between ones and zeros ----------------------
    nominal_dev = imp.modulation_index * symbol_rate / 2.0
    dev_one = nominal_dev * (1.0 + imp.dev_asymmetry / 2.0)
    dev_zero = nominal_dev * (1.0 - imp.dev_asymmetry / 2.0)
    freq = np.where(shaped >= 0, shaped * dev_one, shaped * dev_zero)

    # --- pad, then add the carrier terms -----------------------------------
    pre = int(round(pre_pad_us * 1e-6 * sample_rate))
    post = int(round(post_pad_us * 1e-6 * sample_rate))
    freq = np.concatenate([np.zeros(pre), freq, np.zeros(post)])
    n = len(freq)
    t = np.arange(n) / sample_rate

    # Drift is referenced to the start of the burst, not the start of the array.
    burst_len = n - pre - post
    tb = (np.arange(n) - pre) / sample_rate
    burst_dur = max(burst_len / sample_rate, 1e-9)
    frac = np.clip(tb / burst_dur, 0.0, 1.0)
    drift = imp.drift_hz * frac + imp.drift_curve_hz * frac**2
    freq_total = freq + imp.cfo_hz + drift

    phase = 2.0 * np.pi * np.cumsum(freq_total) / sample_rate
    sig = np.exp(1j * phase)

    # --- PA envelope: ramp up, optional overshoot, ramp down ---------------
    env = np.zeros(n)
    env[pre : pre + burst_len] = 1.0
    rise = max(int(round(imp.ramp_us * 1e-6 * sample_rate)), 1)
    fall = max(int(round(imp.ramp_down_us * 1e-6 * sample_rate)), 1)
    # 10-90% of a raised-cosine edge spans the middle ~58% of the transition
    rise_n = max(int(rise / 0.58), 2)
    fall_n = max(int(fall / 0.58), 2)
    r0 = max(pre - rise_n // 2, 0)
    edge = 0.5 * (1 - np.cos(np.pi * np.arange(rise_n) / rise_n))
    env[r0 : r0 + rise_n] = edge[: max(0, min(rise_n, n - r0))]
    env[r0 + rise_n : pre + burst_len] = 1.0
    if imp.ramp_overshoot > 0:
        os_n = max(int(round(3e-6 * sample_rate)), 4)
        s = slice(r0 + rise_n, min(r0 + rise_n + os_n, n))
        k = np.arange(s.stop - s.start)
        env[s] *= 1.0 + imp.ramp_overshoot * np.exp(-k / (os_n / 3.0))
    f0 = pre + burst_len
    fedge = 0.5 * (1 + np.cos(np.pi * np.arange(fall_n) / fall_n))
    fs_ = slice(f0, min(f0 + fall_n, n))
    env[fs_] = fedge[: fs_.stop - fs_.start]
    env[min(f0 + fall_n, n) :] = 0.0

    sig = sig * env * imp.amplitude

    # --- LO leakage: a residual carrier at DC that does not ramp -----------
    if np.isfinite(imp.lo_leakage_dbc):
        leak = imp.amplitude * 10 ** (imp.lo_leakage_dbc / 20.0)
        sig = sig + leak

    # --- IQ imbalance expressed as a target image-rejection ratio ----------
    if np.isfinite(imp.image_rejection_db):
        eps = 2.0 * 10 ** (-imp.image_rejection_db / 20.0)
        sig = sig + (eps / 2.0) * np.conj(sig)

    # --- phase noise: flat, then integrated to give the usual 1/f^2 skirt --
    if np.isfinite(imp.phase_noise_dbc):
        pn_var = 10 ** (imp.phase_noise_dbc / 10.0) * sample_rate
        pn = np.cumsum(rng.normal(0.0, np.sqrt(pn_var), n)) / sample_rate
        sig = sig * np.exp(1j * 2 * np.pi * pn)

    # --- receiver noise ----------------------------------------------------
    if np.isfinite(imp.snr_db):
        sig_pow = imp.amplitude**2 / 2.0
        npow = sig_pow / (10 ** (imp.snr_db / 10.0))
        sig = sig + (rng.normal(0, np.sqrt(npow / 2), n) + 1j * rng.normal(0, np.sqrt(npow / 2), n))

    truth = {
        "preamble_sample": pre,
        "samples_per_symbol": sps,
        "symbol_rate": symbol_rate,
        "nsym": nsym,
        "burst_len": burst_len,
        "cfo_hz": imp.cfo_hz,
        "drift_hz": imp.drift_hz,
        "modulation_index": imp.modulation_index,
        "dev_one_hz": dev_one,
        "dev_zero_hz": dev_zero,
        "bt": imp.bt,
        "symbol_clock_ppm": imp.symbol_clock_ppm,
        "ramp_us": imp.ramp_us,
        "amplitude": imp.amplitude,
        "snr_db": imp.snr_db,
        "air_bytes": air_bytes,
    }
    return sig.astype(np.complex64), truth


def make_packet(
    channel: int = 37,
    imp: TxImpairments | None = None,
    sample_rate: float = 8e6,
    adva: bytes = b"\xde\xad\xbe\xef\x12\x34",
    ad_payload: bytes = b"\x02\x01\x06",
    pdu_type: int = 0x00,
    corrupt_bits: tuple[int, ...] = (),
    rng: np.random.Generator | None = None,
    **kw,
) -> tuple[np.ndarray, dict]:
    """One complete synthetic advertising packet, padded, with truth metadata."""
    pdu = build_adv_pdu(pdu_type=pdu_type, adva=adva, ad_payload=ad_payload)
    air = build_air_bytes(pdu, channel=channel, corrupt_bits=corrupt_bits)
    sig, truth = modulate(air, sample_rate=sample_rate, imp=imp, rng=rng, **kw)
    truth["pdu"] = pdu
    truth["adva"] = adva
    truth["channel"] = channel
    return sig, truth


def make_stream(
    n_packets: int = 10,
    channel: int = 37,
    sample_rate: float = 8e6,
    gap_us: float = 500.0,
    imp: TxImpairments | None = None,
    devices: list[tuple[bytes, TxImpairments]] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """A stream of packets separated by idle noise, for pipeline-level tests."""
    rng = rng or np.random.default_rng(1)
    chunks: list[np.ndarray] = []
    truths: list[dict] = []
    pos = 0
    noise_amp = 10 ** (-45 / 20.0)
    for i in range(n_packets):
        if devices:
            adva, dimp = devices[i % len(devices)]
        else:
            adva, dimp = b"\xde\xad\xbe\xef\x12\x34", (imp or TxImpairments())
        sig, truth = make_packet(
            channel=channel, imp=dimp, sample_rate=sample_rate, adva=adva, rng=rng
        )
        truth["stream_offset"] = pos + truth["preamble_sample"]
        truths.append(truth)
        chunks.append(sig)
        pos += len(sig)
        gap = int(gap_us * 1e-6 * sample_rate)
        idle = (rng.normal(0, noise_amp, gap) + 1j * rng.normal(0, noise_amp, gap)).astype(
            np.complex64
        )
        chunks.append(idle)
        pos += gap
    return np.concatenate(chunks), truths
