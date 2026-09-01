"""Detection, GFSK demodulation, de-whitening, CRC and PDU parsing.

Everything here is vectorised.  Python touches whole arrays or short per-packet
byte slices, never individual samples of the 8 MSPS stream.  The order of the
stages is chosen for cost: the energy gate is a cumulative sum and rejects
almost every block in a quiet moment, so the expensive correlation only runs
where there is something to find.

Hot loops that genuinely cannot be expressed as array operations -- the bit
slicer and the LFSR -- are compiled with numba when it is importable and fall
back to equivalent numpy/Python otherwise.  `HAVE_NUMBA` reports which.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps_signal

from .channels import (
    ADV_ACCESS_ADDRESS,
    crc24,
    crc24_from_air,
    keystream_for,
)

try:  # pragma: no cover - depends on the environment
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


# --------------------------------------------------------------------------
# instantaneous frequency
# --------------------------------------------------------------------------

def design_channel_filter(sample_rate: float, cutoff_hz: float = 1.1e6, ntaps: int = 65) -> np.ndarray:
    """Low-pass FIR isolating one BLE channel from the captured bandwidth.

    Without this the demodulator integrates noise over the full 3 MHz analog
    bandwidth while the signal occupies barely a third of it, and the sync word
    picks up bit errors at perfectly healthy signal levels.  Measured on air:
    sync-word agreement goes from 36-39 of 40 bits to 40 of 40.

    The cutoff is deliberately generous.  It has to pass the +/-250 kHz
    deviation plus a carrier offset that non-compliant transmitters routinely
    push past the +/-150 kHz spec limit, and the whole point of this application
    is to characterise those transmitters rather than to reject them.
    """
    if ntaps % 2 == 0:
        ntaps += 1
    nyquist = sample_rate / 2.0
    # At 4 MSPS the wide measurement cutoff would sit above Nyquist.  Clamp
    # rather than fail: the anti-alias filtering in the RFIC has already limited
    # the band, so a filter at 0.9 Nyquist is close to a pass-through.
    cutoff = min(cutoff_hz, 0.9 * nyquist)
    return sps_signal.firwin(ntaps, cutoff / nyquist).astype(np.float32)


def apply_channel_filter(iq: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Zero-delay channel filtering; `mode='same'` keeps sample indices aligned.

    Index alignment matters: the detection index is used to cut the retained
    slice out of the *unfiltered* stream, so a group delay here would offset
    every transient feature by half the filter length.
    """
    if iq.size < taps.size:
        return iq
    return sps_signal.fftconvolve(iq, taps, mode="same").astype(np.complex64)


def instantaneous_frequency(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    """Frequency in Hz per sample, from the angle of the conjugate product.

    One sample shorter than the input; element k is the frequency between
    samples k and k+1.
    """
    if iq.size < 2:
        return np.zeros(0, dtype=np.float32)
    prod = iq[1:] * np.conj(iq[:-1])
    return (np.angle(prod) * (sample_rate / (2.0 * np.pi))).astype(np.float32)


def boxcar(x: np.ndarray, n: int) -> np.ndarray:
    """Running sum of length `n` via cumulative sum; same length as `x`."""
    if n <= 1:
        return x.astype(np.float64, copy=False)
    c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    out = np.empty_like(x, dtype=np.float64)
    out[: len(x) - n + 1] = c[n:] - c[:-n]
    out[len(x) - n + 1 :] = out[len(x) - n] if len(x) >= n else 0.0
    return out


# --------------------------------------------------------------------------
# energy gate
# --------------------------------------------------------------------------

@dataclass
class NoiseFloor:
    """Adaptive noise-floor estimate in linear power, tracked per channel.

    Uses a low quantile of per-window power rather than a mean, so an
    intermittent strong emitter does not drag the floor up and desensitise the
    gate.  Reset on retune -- the floor is a property of the channel.
    """

    value: float = 1e-9
    alpha: float = 0.05
    initialised: bool = False
    history: list = field(default_factory=list)

    def update(self, power_windows: np.ndarray) -> float:
        if power_windows.size == 0:
            return self.value
        # A quantile means a sort; on 16k windows at 500 blocks/s that alone is
        # a measurable slice of the budget.  The floor is a slowly varying
        # statistic, so estimate it from a decimated view.
        sample = power_windows[::16] if power_windows.size >= 4096 else power_windows
        est = float(np.quantile(sample, 0.15))
        if est <= 0 or not np.isfinite(est):
            return self.value
        if not self.initialised:
            self.value = est
            self.initialised = True
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * est
        return self.value

    def reset(self) -> None:
        self.value = 1e-9
        self.initialised = False
        self.history.clear()

    @property
    def dbfs(self) -> float:
        return 10.0 * np.log10(max(self.value, 1e-15))


def energy_gate(
    iq: np.ndarray,
    sample_rate: float,
    noise: NoiseFloor,
    threshold_db: float = 8.0,
    window_us: float = 4.0,
) -> tuple[np.ndarray, float]:
    """Cheap first-pass gate.  Returns (candidate mask, window power array).

    Squared magnitude, boxcar-averaged over `window_us`, compared against the
    adaptive floor.  Everything downstream is skipped for blocks where this
    returns an all-false mask, and that is what buys the real-time headroom.
    """
    if iq.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    n = max(int(window_us * 1e-6 * sample_rate), 1)
    power = (iq.real.astype(np.float32) ** 2 + iq.imag.astype(np.float32) ** 2)
    win = boxcar(power, n) / n
    floor = noise.update(win)
    return win > floor * (10 ** (threshold_db / 10.0)), win


# --------------------------------------------------------------------------
# preamble + access address correlation
# --------------------------------------------------------------------------

def _aa_bits(access_address: int) -> np.ndarray:
    aa = np.array(
        [
            access_address & 0xFF,
            (access_address >> 8) & 0xFF,
            (access_address >> 16) & 0xFF,
            (access_address >> 24) & 0xFF,
        ],
        dtype=np.uint8,
    )
    preamble = 0xAA if (aa[0] & 1) == 0 else 0x55
    full = np.concatenate([[preamble], aa]).astype(np.uint8)
    return np.unpackbits(full, bitorder="little")


def sync_reference(access_address: int, sps: float) -> np.ndarray:
    """Zero-mean +/-1 template for preamble+AA, upsampled to `sps`.

    Zero-mean so that a constant carrier offset -- which appears as a DC term in
    the instantaneous-frequency signal -- contributes nothing to the correlation
    and cannot move the detected timing.
    """
    bits = _aa_bits(access_address)
    nrz = bits.astype(np.float32) * 2.0 - 1.0
    n = int(round(len(nrz) * sps))
    idx = np.minimum((np.arange(n) / sps).astype(int), len(nrz) - 1)
    ref = nrz[idx]
    return ref - ref.mean()


def correlate_sync(
    freq: np.ndarray, ref: np.ndarray, normalise: bool = True
) -> np.ndarray:
    """Normalised correlation of the frequency signal against the sync template.

    FFT overlap-save via `scipy.signal.fftconvolve`; no Python-level loop over
    lags.  The result is aligned so index k is the correlation of a template
    starting at sample k.
    """
    if freq.size < ref.size:
        return np.zeros(0, dtype=np.float32)
    corr = sps_signal.fftconvolve(freq, ref[::-1], mode="valid").astype(np.float32)
    if not normalise:
        return corr

    # Normalise by the local *variance*, not the local energy.  A carrier offset
    # appears as a constant term in the instantaneous frequency: it contributes
    # nothing to the numerator (the template is zero-mean) but it inflates the
    # energy, so an energy-normalised score falls as the offset grows.  A
    # transmitter 400 kHz off frequency then fails to correlate at all -- and
    # those are exactly the transmitters worth looking at.
    n = len(ref)
    f64 = freq.astype(np.float64)
    s1 = boxcar(f64, n)[: len(corr)]
    s2 = boxcar(f64 * f64, n)[: len(corr)]
    var = np.maximum(s2 - (s1 * s1) / n, 1e-12)
    denom = np.sqrt(var * float(np.sum(ref**2)))
    return (corr / denom).astype(np.float32)


def find_sync_positions(
    corr: np.ndarray, threshold: float = 0.62, min_separation: int = 200
) -> np.ndarray:
    """Local maxima of the correlation above `threshold`, at least `min_separation` apart."""
    if corr.size == 0:
        return np.zeros(0, dtype=np.int64)
    cand = np.flatnonzero(corr > threshold)
    if cand.size == 0:
        return cand
    # keep only the strongest peak within each run of candidates
    picks: list[int] = []
    start = 0
    for i in range(1, len(cand) + 1):
        if i == len(cand) or cand[i] - cand[i - 1] > min_separation:
            group = cand[start:i]
            picks.append(int(group[np.argmax(corr[group])]))
            start = i
    return np.array(picks, dtype=np.int64)


# --------------------------------------------------------------------------
# symbol slicing
# --------------------------------------------------------------------------

@njit(cache=True)
def _slice_symbols_nb(freq, start, sps, nsym, offset):  # pragma: no cover
    out = np.empty(nsym, dtype=np.float32)
    for k in range(nsym):
        idx = int(start + offset + k * sps + 0.5)
        if idx < 0 or idx >= freq.shape[0]:
            out[k] = 0.0
        else:
            out[k] = freq[idx]
    return out


def slice_symbols(
    freq: np.ndarray, start: float, sps: float, nsym: int, offset: float | None = None
) -> np.ndarray:
    """Sample the frequency signal at symbol centres."""
    if offset is None:
        offset = sps / 2.0
    if HAVE_NUMBA:
        return _slice_symbols_nb(freq, float(start), float(sps), int(nsym), float(offset))
    idx = np.round(start + offset + np.arange(nsym) * sps).astype(np.int64)
    idx = np.clip(idx, 0, len(freq) - 1)
    return freq[idx]


def refine_timing(
    freq: np.ndarray, start: int, sps: float, ref_bits: np.ndarray, search: float = 1.0
) -> tuple[float, float]:
    """Refine the symbol sampling phase against the known sync word.

    Returns (best offset from `start` in samples, agreement score 0..1).

    Sync-word agreement alone is not a sharp enough criterion.  Measured on air
    at 8 samples/symbol, agreement stays at a perfect 40/40 across a span of
    nearly six samples, while the eye opening over that same span varies by a
    factor of three -- so picking the first offset that matches the sync word
    can leave the slicer sampling close to a transition.  The sync word then
    decodes perfectly and the payload does not, which reads like a whitening
    bug and is not one.

    So: maximise agreement first, then among the offsets that tie on agreement,
    maximise the worst-case eye opening.  That second term is what makes the
    payload decode.
    """
    nsym = len(ref_bits)
    target = (ref_bits.astype(np.float32) * 2.0 - 1.0)[None, :]

    # All candidate offsets are evaluated in one gather rather than in a Python
    # loop.  The loop version, calling a compiled slicer once per candidate, cost
    # 9.8 ms per detection -- five times the entire real-time budget for a block
    # -- because it paid dispatch overhead 33 times over for 40 samples of work.
    offs = sps / 2.0 + np.linspace(-search, search, 33) * sps
    idx = np.rint(start + offs[:, None] + np.arange(nsym)[None, :] * sps).astype(np.int64)
    np.clip(idx, 0, len(freq) - 1, out=idx)
    sym = freq[idx]  # (n_offsets, nsym)

    centred = sym - sym.mean(axis=1, keepdims=True)
    scores = np.mean(np.sign(centred) == target, axis=1)
    eyes = np.min(np.abs(centred), axis=1)

    best_score = scores.max()
    tied = np.flatnonzero(scores >= best_score - 1e-9)
    pick = tied[np.argmax(eyes[tied])]
    return float(offs[pick]), float(best_score)


# --------------------------------------------------------------------------
# de-whitening and CRC (compiled where it pays)
# --------------------------------------------------------------------------

def bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    """LSB-first bit array to bytes; length truncated to a whole number of bytes."""
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n].astype(np.uint8), bitorder="little")


# --------------------------------------------------------------------------
# PDU parsing
# --------------------------------------------------------------------------

ADV_PDU_TYPES = {
    0x00: "ADV_IND",
    0x01: "ADV_DIRECT_IND",
    0x02: "ADV_NONCONN_IND",
    0x03: "SCAN_REQ",
    0x04: "SCAN_RSP",
    0x05: "CONNECT_IND",
    0x06: "ADV_SCAN_IND",
    0x07: "ADV_EXT_IND",
    0x08: "AUX_CONNECT_RSP",
}

AD_TYPES = {
    0x01: "Flags",
    0x02: "Incomplete 16-bit UUIDs",
    0x03: "Complete 16-bit UUIDs",
    0x04: "Incomplete 32-bit UUIDs",
    0x05: "Complete 32-bit UUIDs",
    0x06: "Incomplete 128-bit UUIDs",
    0x07: "Complete 128-bit UUIDs",
    0x08: "Shortened Local Name",
    0x09: "Complete Local Name",
    0x0A: "Tx Power Level",
    0x0D: "Class of Device",
    0x10: "Device ID",
    0x12: "Peripheral Connection Interval Range",
    0x14: "16-bit Service Solicitation",
    0x16: "Service Data (16-bit)",
    0x19: "Appearance",
    0x1A: "Advertising Interval",
    0x1B: "LE Device Address",
    0x20: "Service Data (32-bit)",
    0x21: "Service Data (128-bit)",
    0x24: "URI",
    0x2D: "Mesh Message",
    0x3D: "3D Information Data",
    0xFF: "Manufacturer Specific",
}

COMPANY_IDS = {
    0x0006: "Microsoft",
    0x004C: "Apple",
    0x0075: "Samsung",
    0x00E0: "Google",
    0x0087: "Garmin",
    0x0157: "Huawei",
    0x0171: "Amazon",
    0x0499: "Ruuvi",
    0x05A7: "Sonos",
    0x038F: "Xiaomi",
}


def format_address(addr: bytes) -> str:
    """BLE addresses are transmitted least-significant byte first."""
    return ":".join(f"{b:02X}" for b in reversed(bytes(addr)))


def address_kind(tx_add_random: bool, addr: bytes) -> str:
    """Classify the address per Vol 6 B 1.3.2 using its two most significant bits."""
    if not tx_add_random:
        return "public"
    if len(addr) != 6:
        return "random"
    top = addr[5] >> 6
    return {0b11: "random static", 0b01: "resolvable private", 0b00: "non-resolvable private"}.get(
        top, "random (reserved)"
    )


def parse_ad_structures(payload: bytes) -> list[dict]:
    """Walk the length/type/value list of an advertising payload."""
    out: list[dict] = []
    i = 0
    data = bytes(payload)
    while i < len(data):
        ln = data[i]
        if ln == 0:
            break
        if i + 1 + ln > len(data):
            out.append(
                {
                    "type": None,
                    "name": "Truncated AD structure",
                    "raw": data[i:],
                    "value": None,
                    "truncated": True,
                }
            )
            break
        ad_type = data[i + 1]
        value = data[i + 2 : i + 1 + ln]
        entry = {
            "type": ad_type,
            "name": AD_TYPES.get(ad_type, f"Unknown (0x{ad_type:02X})"),
            "raw": value,
            "value": _decode_ad_value(ad_type, value),
            "truncated": False,
        }
        out.append(entry)
        i += 1 + ln
    return out


def _decode_ad_value(ad_type: int, value: bytes):
    try:
        if ad_type in (0x08, 0x09):
            return value.decode("utf-8", "replace")
        if ad_type == 0x01 and value:
            flags = value[0]
            names = []
            for bit, label in (
                (0, "LE Limited Discoverable"),
                (1, "LE General Discoverable"),
                (2, "BR/EDR Not Supported"),
                (3, "Simultaneous LE+BR/EDR (Controller)"),
                (4, "Simultaneous LE+BR/EDR (Host)"),
            ):
                if flags & (1 << bit):
                    names.append(label)
            return f"0x{flags:02X}" + (" | " + ", ".join(names) if names else "")
        if ad_type == 0x0A and value:
            return f"{int.from_bytes(value[:1], 'little', signed=True)} dBm"
        if ad_type == 0x19 and len(value) >= 2:
            return f"0x{int.from_bytes(value[:2], 'little'):04X}"
        if ad_type == 0x1A and len(value) >= 2:
            return f"{int.from_bytes(value[:2], 'little') * 0.625:.1f} ms"
        if ad_type in (0x02, 0x03, 0x14) and len(value) >= 2:
            uuids = [
                f"0x{int.from_bytes(value[i:i+2], 'little'):04X}"
                for i in range(0, len(value) - 1, 2)
            ]
            return ", ".join(uuids)
        if ad_type == 0x16 and len(value) >= 2:
            uuid = int.from_bytes(value[:2], "little")
            return f"UUID 0x{uuid:04X}, {value[2:].hex()}"
        if ad_type == 0xFF and len(value) >= 2:
            cid = int.from_bytes(value[:2], "little")
            who = COMPANY_IDS.get(cid, f"0x{cid:04X}")
            return f"{who}: {value[2:].hex()}"
    except Exception:
        pass
    return value.hex() if value else ""


@dataclass
class DecodedPDU:
    """A parsed advertising PDU."""

    ok: bool
    pdu_type: int = 0
    pdu_name: str = ""
    tx_add_random: bool = False
    rx_add_random: bool = False
    chsel: bool = False
    length: int = 0
    adva: bytes = b""
    adva_str: str = ""
    adva_kind: str = ""
    target_a: bytes = b""
    payload: bytes = b""
    ad_structures: list = field(default_factory=list)
    crc_received: int = 0
    crc_computed: int = 0
    crc_ok: bool = False
    raw: bytes = b""
    info: str = ""
    error: str = ""


def parse_pdu(pdu_and_crc: bytes, crc_init: int, is_advertising: bool = True) -> DecodedPDU:
    """Parse a de-whitened (header || payload || CRC) byte string."""
    data = bytes(pdu_and_crc)
    if len(data) < 5:
        return DecodedPDU(ok=False, error="runt PDU", raw=data)

    h0, h1 = data[0], data[1]
    pdu_type = h0 & 0x0F
    length = h1 & 0xFF
    body_len = 2 + length
    if body_len + 3 > len(data):
        return DecodedPDU(
            ok=False,
            error=f"length field {length} exceeds captured bytes",
            pdu_type=pdu_type,
            length=length,
            raw=data,
        )

    body = data[:body_len]
    crc_bytes = data[body_len : body_len + 3]
    crc_rx = crc24_from_air(crc_bytes)
    crc_calc = crc24(body, crc_init)

    out = DecodedPDU(
        ok=True,
        pdu_type=pdu_type,
        pdu_name=ADV_PDU_TYPES.get(pdu_type, f"RFU(0x{pdu_type:02X})"),
        tx_add_random=bool(h0 & 0x40),
        rx_add_random=bool(h0 & 0x80),
        chsel=bool(h0 & 0x20),
        length=length,
        crc_received=crc_rx,
        crc_computed=crc_calc,
        crc_ok=(crc_rx == crc_calc),
        raw=data[: body_len + 3],
    )

    payload = body[2:]
    out.payload = payload

    # Address layout depends on the PDU type (Vol 6 B 2.3).
    if pdu_type in (0x00, 0x02, 0x06, 0x04, 0x03, 0x05, 0x01) and len(payload) >= 6:
        out.adva = payload[:6]
        out.adva_str = format_address(out.adva)
        out.adva_kind = address_kind(out.tx_add_random, out.adva)
        rest = payload[6:]
        if pdu_type in (0x01, 0x03, 0x05) and len(rest) >= 6:
            out.target_a = rest[:6]
            rest = rest[6:]
        if pdu_type in (0x00, 0x02, 0x06, 0x04):
            out.ad_structures = parse_ad_structures(rest)
    elif pdu_type == 0x07:
        out.info = "extended advertising header (not parsed on 1M legacy path)"

    if out.ad_structures:
        name = next(
            (a["value"] for a in out.ad_structures if a["type"] in (0x08, 0x09)), None
        )
        if name:
            out.info = str(name)
        else:
            out.info = out.ad_structures[0]["name"]
    elif not out.info:
        out.info = out.pdu_name

    return out


# --------------------------------------------------------------------------
# the demodulator
# --------------------------------------------------------------------------

@dataclass
class Detection:
    """One successfully sliced burst, before feature extraction."""

    sync_index: int  # sample index of the first preamble symbol
    corr_peak: float
    sym_offset: float
    sync_score: float
    bits: np.ndarray
    air_bytes: np.ndarray
    pdu: DecodedPDU
    freq_symbols: np.ndarray  # frequency at each symbol centre, Hz
    slice_start: int  # start of the retained IQ window, in the buffer processed
    slice_end: int
    payload_start_sym: int  # first symbol after preamble+AA
    n_symbols: int
    # Filled in by `process_stream`: the retained samples themselves, cut from
    # the buffer the detector actually used, plus their absolute index.
    iq_slice: np.ndarray | None = None
    abs_slice_start: int = 0


class Demodulator:
    """Turns a block of IQ into decoded packets for one channel plan."""

    PREAMBLE_AA_BITS = 8 + 32
    MAX_PDU_BITS = (2 + 255 + 3) * 8

    def __init__(
        self,
        sample_rate: float,
        access_address: int = ADV_ACCESS_ADDRESS,
        channel: int = 37,
        crc_init: int = 0x555555,
        corr_threshold: float = 0.62,
        pre_window_us: float = 50.0,
        post_window_us: float = 20.0,
        channel_filter_hz: float = 1.1e6,
        max_payload_bytes: int = 37,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.sps = self.sample_rate / 1e6
        self.access_address = access_address
        self.channel = channel
        self.crc_init = crc_init
        self.corr_threshold = corr_threshold
        self.pre_window = int(pre_window_us * 1e-6 * sample_rate)
        self.post_window = int(post_window_us * 1e-6 * sample_rate)
        self.ref = sync_reference(access_address, self.sps)
        self.ref_bits = _aa_bits(access_address)
        self.taps = design_channel_filter(sample_rate, channel_filter_hz)
        self.noise = NoiseFloor()
        self.max_payload_bytes = max_payload_bytes
        # Enough tail to hold one whole maximum-length packet plus its
        # pre-trigger window, so no packet can straddle a boundary unseen.
        self._tail_len = int((5 + max_payload_bytes + 3) * 8 * self.sps) + self.pre_window
        self._tail = np.zeros(0, dtype=np.complex64)
        self._last_emitted = -1
        self._ks_bits = np.unpackbits(
            keystream_for(channel, 2 + 255 + 3), bitorder="little"
        )

    def process_stream(self, iq: np.ndarray, abs_start: int) -> list[tuple[int, Detection]]:
        """Process one block while stitching across the block boundary.

        A packet that begins near the end of a block does not fit inside it, and
        `_decode_at` correctly refuses to decode a truncated one -- so without
        overlap every such packet is simply lost.  At 8 MSPS with 16k blocks and
        a 376 us advertising packet that is a few percent of all traffic, which
        would put the capture ratio outside the acceptance criterion on its own.

        Carrying a tail of the previous block and de-duplicating by absolute
        sample index recovers them.  Returns (absolute sync index, detection).
        """
        if self._tail.size:
            buf = np.concatenate([self._tail, iq])
            buf_abs = abs_start - self._tail.size
        else:
            buf, buf_abs = iq, abs_start

        out: list[tuple[int, Detection]] = []
        for det in self.process(buf):
            abs_sync = buf_abs + det.sync_index
            # A packet visible in both the tail and the previous buffer must be
            # emitted once; the previous pass wins.
            if abs_sync <= self._last_emitted:
                continue
            self._last_emitted = abs_sync
            # Attach the samples here rather than returning indices.  The
            # detection indices are relative to `buf`, which is the carried tail
            # plus this block -- a caller that slices them out of the block alone
            # gets a window shifted by the tail length.  That decodes fine (the
            # bits came from `buf`) while every feature is measured on the wrong
            # samples, which is a genuinely nasty failure mode because it looks
            # like an estimator bug rather than an indexing one.
            det.iq_slice = buf[det.slice_start : det.slice_end].copy()
            det.abs_slice_start = buf_abs + det.slice_start
            out.append((abs_sync, det))

        keep = min(self._tail_len, buf.size)
        self._tail = buf[buf.size - keep :].copy()
        return out

    def reset_stream(self) -> None:
        """Drop carried state.  Called on retune, where continuity is broken."""
        self._tail = np.zeros(0, dtype=np.complex64)
        self._last_emitted = -1
        self.noise.reset()

    def reconfigure(self, channel: int, access_address: int, crc_init: int) -> None:
        """Retune the decoder.  Whitening keystream follows the channel index."""
        self.channel = channel
        self.access_address = access_address
        self.crc_init = crc_init
        self.ref = sync_reference(access_address, self.sps)
        self.ref_bits = _aa_bits(access_address)
        self._ks_bits = np.unpackbits(
            keystream_for(channel, 2 + 255 + 3), bitorder="little"
        )
        self.reset_stream()

    # ------------------------------------------------------------------
    def process(self, iq: np.ndarray, gate: bool = True) -> list[Detection]:
        """Find and decode every packet in one block of IQ."""
        if iq.size < len(self.ref) + 64:
            return []

        if not gate:
            return self._process_span(iq, 0, iq.size)

        # The gate runs on raw magnitude only: no filtering, no arctangent, no
        # FFT.  Everything expensive is then confined to the spans that actually
        # carry energy, so the cost tracks channel occupancy rather than wall
        # time.  Advertising traffic is bursty even on a busy channel, and this
        # is what turns a 1.3x-realtime pipeline into a comfortable one.
        mask, _ = energy_gate(iq, self.sample_rate, self.noise)
        if not mask.any():
            return []

        out: list[Detection] = []
        for lo, hi in self._energetic_spans(mask, iq.size):
            out.extend(self._process_span(iq, lo, hi))
        return out

    def _energetic_spans(self, mask: np.ndarray, n: int) -> list[tuple[int, int]]:
        """Contiguous runs of `mask`, padded and merged.

        Padding is asymmetric on purpose: a sync word is found at the *start* of
        a burst, so the span must extend forward by a whole maximum-length
        packet, and backward only by the pre-trigger window.
        """
        edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
        starts = list(edges[::2] + 1) if not mask[0] else [0] + list(edges[1::2] + 1)
        stops = list(edges[1::2] + 1) if not mask[0] else list(edges[::2] + 1)
        if len(stops) < len(starts):
            stops.append(n)

        # Backward padding must cover the retained pre-trigger window, since the
        # slice handed to the feature extractor starts 50 us before the preamble
        # and the PA ramp there is below the energy threshold by definition.
        #
        # Forward padding only needs a small margin.  It is tempting to pad
        # forward by a whole maximum-length packet so that a preamble detected
        # at the end of a span still has its payload inside -- but the payload
        # carries energy too, so the gate has already included it.  Padding
        # forward by a full packet instead inflates every span by 3400 samples
        # and was costing about a third of the entire real-time budget.
        pad_back = self.pre_window + len(self.ref)
        pad_fwd = int(4 * self.sps) + len(self.ref)
        spans: list[tuple[int, int]] = []
        for s, e in zip(starts, stops):
            lo = max(int(s) - pad_back, 0)
            hi = min(int(e) + pad_fwd, n)
            if spans and lo <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
            else:
                spans.append((lo, hi))
        return spans

    def _process_span(self, iq: np.ndarray, lo: int, hi: int) -> list[Detection]:
        """Filter, demodulate and decode one energetic span of the block."""
        seg = iq[lo:hi]
        if seg.size < len(self.ref) + 64:
            return []

        # Detection and bit decisions run on the channel-filtered signal; the
        # retained slice handed to the feature extractor stays unfiltered, so
        # ramp shape and spectral splatter are measured on what was actually
        # received rather than on the filter's step response.
        filtered = apply_channel_filter(seg, self.taps)
        freq = instantaneous_frequency(filtered, self.sample_rate)
        corr = correlate_sync(freq, self.ref)
        if corr.size == 0:
            return []

        positions = find_sync_positions(
            corr, self.corr_threshold, min_separation=int(self.sps * 8)
        )
        out: list[Detection] = []
        for pos in positions:
            det = self._decode_at(seg, freq, int(pos), float(corr[pos]))
            if det is None:
                continue
            # Re-base indices onto the full block so the caller's absolute
            # sample arithmetic and the retained IQ slice stay correct.
            det.sync_index += lo
            det.slice_start += lo
            det.slice_end += lo
            out.append(det)
        return out

    def _decode_at(
        self, iq: np.ndarray, freq: np.ndarray, pos: int, peak: float
    ) -> Detection | None:
        sps = self.sps
        off, score = refine_timing(freq, pos, sps, self.ref_bits)
        if score < 0.85:
            return None

        # Slice header first so we learn the true length before slicing the rest.
        hdr_sym = self.PREAMBLE_AA_BITS + 16
        need = int((hdr_sym + 2) * sps) + 4
        if pos + need >= len(freq):
            return None

        sym = slice_symbols(freq, pos, sps, hdr_sym, off)
        bits = self._slice_bits(sym)
        hdr = self._dewhiten_bits(bits[self.PREAMBLE_AA_BITS :])
        hdr_bytes = bits_to_bytes(hdr)
        if len(hdr_bytes) < 2:
            return None
        length = int(hdr_bytes[1])
        if length > 255:
            return None

        total_pdu_bits = (2 + length + 3) * 8
        n_sym = self.PREAMBLE_AA_BITS + total_pdu_bits
        end_needed = int(pos + off + n_sym * sps) + 4
        if end_needed >= len(freq):
            return None

        sym = slice_symbols(freq, pos, sps, n_sym, off)
        bits = self._slice_bits(sym)
        body = self._dewhiten_bits(bits[self.PREAMBLE_AA_BITS :])
        air = bits_to_bytes(body)
        pdu = parse_pdu(air.tobytes(), self.crc_init)

        last_sample = int(pos + off + n_sym * sps)
        slice_start = max(pos - self.pre_window, 0)
        slice_end = min(last_sample + self.post_window, len(iq))

        return Detection(
            sync_index=pos,
            corr_peak=peak,
            sym_offset=off,
            sync_score=score,
            bits=bits,
            air_bytes=air,
            pdu=pdu,
            freq_symbols=sym,
            slice_start=slice_start,
            slice_end=slice_end,
            payload_start_sym=self.PREAMBLE_AA_BITS,
            n_symbols=n_sym,
        )

    # ------------------------------------------------------------------
    def _slice_bits(self, sym: np.ndarray) -> np.ndarray:
        """Hard-decide symbols against their own mean.

        The mean of the symbol-centre frequencies over a whole packet is the
        carrier offset; slicing against it makes the decision robust to an
        offset of many hundreds of kHz without a separate AFC loop.
        """
        # Use the sync word, whose average is zero by construction, to set the
        # decision level -- payload data is not guaranteed to be DC balanced.
        n_sync = min(self.PREAMBLE_AA_BITS, len(sym))
        level = float(np.mean(sym[:n_sync])) if n_sync else float(np.mean(sym))
        return (sym > level).astype(np.uint8)

    def _dewhiten_bits(self, bits: np.ndarray) -> np.ndarray:
        ks = self._ks_bits
        if len(bits) > len(ks):
            ks = np.unpackbits(
                keystream_for(self.channel, (len(bits) + 7) // 8), bitorder="little"
            )
            self._ks_bits = ks
        return np.bitwise_xor(bits, ks[: len(bits)])


def bit_error_positions(received: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Indices where two bit arrays differ; used for the CRC-failure histogram."""
    n = min(len(received), len(expected))
    return np.flatnonzero(received[:n] != expected[:n])
