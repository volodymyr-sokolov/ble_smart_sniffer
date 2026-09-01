"""BLE channel map, whitening LFSR and CRC-24.

The channel index is the single source of truth for three things that must never
disagree: the LO frequency, the de-whitening LFSR seed, and the metadata stamped
into each packet record.  `ChannelPlan` derives all three from one integer and
`ChannelPlan.assert_consistent()` re-derives the index back out of the frequency
so a mismatch is caught at startup rather than showing up as a silent 0% CRC
pass rate hours into a capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed access address for all three advertising channels (Core spec Vol 6 B 2.1.2).
ADV_ACCESS_ADDRESS = 0x8E89BED6

# CRC-24 seed for advertising physical channel PDUs (Vol 6 B 3.1.1).
ADV_CRC_INIT = 0x555555

BLE_SYMBOL_RATE = 1_000_000.0  # 1 Msym/s, LE 1M PHY
BLE_DEVIATION = 250_000.0  # nominal +/-250 kHz at modulation index 0.5


# --------------------------------------------------------------------------
# channel <-> frequency
# --------------------------------------------------------------------------

def channel_to_freq(channel: int) -> float:
    """RF centre frequency in Hz for a BLE channel index (0..39)."""
    if channel == 37:
        return 2_402_000_000.0
    if channel == 38:
        return 2_426_000_000.0
    if channel == 39:
        return 2_480_000_000.0
    if 0 <= channel <= 10:
        return 2_404_000_000.0 + 2_000_000.0 * channel
    if 11 <= channel <= 36:
        return 2_428_000_000.0 + 2_000_000.0 * (channel - 11)
    raise ValueError(f"channel index out of range 0..39: {channel}")


def freq_to_channel(freq_hz: float) -> int | None:
    """Inverse of :func:`channel_to_freq`; None when the frequency is off-plan."""
    for ch in range(40):
        if abs(channel_to_freq(ch) - freq_hz) < 1.0:
            return ch
    return None


def is_advertising_channel(channel: int) -> bool:
    return channel in (37, 38, 39)


# --------------------------------------------------------------------------
# whitening
# --------------------------------------------------------------------------

def whitening_keystream(channel: int, nbytes: int) -> np.ndarray:
    """Data-whitening keystream for `channel`, `nbytes` long.

    7-bit LFSR, polynomial x^7 + x^4 + 1 (Vol 6 B 3.2).  Position 0 is preset to
    1 and positions 1..6 take the 6-bit channel index, MSB first.  The output is
    consumed LSB-first per byte because BLE is transmitted LSB-first.
    """
    if not 0 <= channel <= 39:
        raise ValueError(f"channel index out of range 0..39: {channel}")

    reg = [0] * 7
    reg[0] = 1
    for i in range(6):
        reg[i + 1] = (channel >> (5 - i)) & 1

    out = np.zeros(nbytes, dtype=np.uint8)
    for n in range(nbytes):
        byte = 0
        for b in range(8):
            bit = reg[6]
            byte |= bit << b
            # x^7 + x^4 + 1: the output bit shifts into position 0 and is XORed
            # into position 4 -- the x^4 tap.  Verified against live air traffic:
            # XORing into position 3 instead still produces a self-consistent
            # codec (a synthetic packet whitened and de-whitened by it round
            # trips perfectly) but decodes nothing off the air.
            reg[6] = reg[5]
            reg[5] = reg[4]
            reg[4] = reg[3] ^ bit
            reg[3] = reg[2]
            reg[2] = reg[1]
            reg[1] = reg[0]
            reg[0] = bit
        out[n] = byte
    return out


# Keystreams are short and channel-constant; cache them per channel.
_KEYSTREAM_CACHE: dict[int, np.ndarray] = {}
_MAX_PDU_BYTES = 2 + 255 + 3  # header + payload + CRC


def keystream_for(channel: int, nbytes: int) -> np.ndarray:
    ks = _KEYSTREAM_CACHE.get(channel)
    if ks is None or len(ks) < nbytes:
        ks = whitening_keystream(channel, max(nbytes, _MAX_PDU_BYTES))
        _KEYSTREAM_CACHE[channel] = ks
    return ks[:nbytes]


def dewhiten(data: np.ndarray, channel: int) -> np.ndarray:
    """XOR `data` (uint8 array) with the channel keystream.  Self-inverse."""
    data = np.asarray(data, dtype=np.uint8)
    return np.bitwise_xor(data, keystream_for(channel, len(data)))


def whitening_bits(channel: int, nbits: int) -> np.ndarray:
    """Keystream as one bit per element, in transmission (LSB-first) order."""
    ks = keystream_for(channel, (nbits + 7) // 8)
    return np.unpackbits(ks, bitorder="little")[:nbits]


# --------------------------------------------------------------------------
# CRC-24
# --------------------------------------------------------------------------

# Reversed form of x^24+x^10+x^9+x^6+x^4+x^3+x+1: 0x00065B reflected -> 0xDA6000
_CRC_POLY_REV = 0xDA6000


def _build_crc_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.uint32)
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC_POLY_REV if crc & 1 else 0)
        table[byte] = crc & 0xFFFFFF
    return table


_CRC_TABLE = _build_crc_table()


def reflect24(value: int) -> int:
    """Reverse the bit order of a 24-bit word."""
    out = 0
    for i in range(24):
        if value & (1 << i):
            out |= 1 << (23 - i)
    return out


def crc24(data: np.ndarray | bytes, init: int = ADV_CRC_INIT) -> int:
    """BLE CRC-24 over `data`, seeded with `init` as the spec states it.

    The spec describes the CRC with the most significant bit of the shift
    register first, while this implementation is the reflected (LSB-first) form
    that matches BLE's transmission order and allows a byte-at-a-time table.
    The two differ only in the seed's bit order, so the reflection is done here
    rather than pushed onto the caller: `ADV_CRC_INIT` stays 0x555555, exactly
    the number in Vol 6 B 3.1.1 and exactly what an operator passes to
    `--crc-init` for a data channel.

    Confirmed on air: an advertising packet whose CRC verifies with a reflected
    seed of 0xAAAAAA fails with 0x555555 applied directly.
    """
    crc = reflect24(init & 0xFFFFFF)
    for byte in bytes(data):
        crc = (crc >> 8) ^ int(_CRC_TABLE[(crc ^ byte) & 0xFF])
    return crc & 0xFFFFFF


def crc24_bytes(data: np.ndarray | bytes, init: int = ADV_CRC_INIT) -> bytes:
    """CRC as it appears on air: three bytes, least significant first."""
    crc = crc24(data, init)
    return bytes((crc & 0xFF, (crc >> 8) & 0xFF, (crc >> 16) & 0xFF))


def crc24_from_air(three_bytes: bytes) -> int:
    """Reassemble the 24-bit CRC value from its on-air byte order."""
    b = bytes(three_bytes)
    return b[0] | (b[1] << 8) | (b[2] << 16)


# --------------------------------------------------------------------------
# the plan object
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelPlan:
    """Everything derived from one channel selection.

    Build with :meth:`from_args` so the LO frequency, whitening seed and record
    metadata can never be set independently of each other.
    """

    channel: int
    frequency_hz: float
    access_address: int
    crc_init: int
    whitening_channel: int  # the seed actually handed to the de-whitener
    label: str

    @classmethod
    def from_args(
        cls,
        channel: int | None = None,
        freq_hz: float | None = None,
        access_address: int | None = None,
        crc_init: int | None = None,
    ) -> "ChannelPlan":
        if freq_hz is not None:
            derived = freq_to_channel(float(freq_hz))
            if channel is None and derived is None:
                raise ValueError(
                    f"--freq {float(freq_hz)/1e6:.3f} MHz is not a BLE channel "
                    "centre; pass --channel too so the whitening seed is defined"
                )
            ch = channel if channel is not None else derived
            frequency = float(freq_hz)
        else:
            ch = 37 if channel is None else channel
            frequency = channel_to_freq(ch)

        if not 0 <= ch <= 39:
            raise ValueError(f"channel index out of range 0..39: {ch}")

        if is_advertising_channel(ch):
            aa = ADV_ACCESS_ADDRESS if access_address is None else access_address
            ci = ADV_CRC_INIT if crc_init is None else crc_init
        else:
            if access_address is None:
                raise ValueError(
                    f"channel {ch} is a data channel: --access-address is required "
                    "(the advertising address 0x8E89BED6 will not decode it)"
                )
            aa = access_address
            ci = ADV_CRC_INIT if crc_init is None else crc_init

        kind = "adv" if is_advertising_channel(ch) else "data"
        label = f"ch{ch} ({kind}) {frequency/1e6:.3f} MHz"
        plan = cls(
            channel=ch,
            frequency_hz=frequency,
            access_address=aa & 0xFFFFFFFF,
            crc_init=ci & 0xFFFFFF,
            whitening_channel=ch,
            label=label,
        )
        plan.assert_consistent()
        return plan

    def retuned(self, **kw) -> "ChannelPlan":
        """A new plan for a different channel, re-running all the checks."""
        return ChannelPlan.from_args(**kw)

    def assert_consistent(self) -> None:
        """Fail loudly if the LO and the whitening seed have drifted apart.

        A receiver tuned to 2426 MHz while de-whitening with the channel-37 seed
        decodes nothing and reports no error.  This check turns that silent
        failure into a startup exception.
        """
        if self.whitening_channel != self.channel:
            raise AssertionError(
                f"whitening seed (ch{self.whitening_channel}) does not match "
                f"channel index (ch{self.channel})"
            )
        expected = freq_to_channel(self.frequency_hz)
        if expected is None:
            return  # deliberate off-plan override; nothing to cross-check against
        if expected != self.channel:
            raise AssertionError(
                f"LO {self.frequency_hz/1e6:.3f} MHz is channel {expected} but the "
                f"whitening seed is for channel {self.channel}; refusing to run"
            )
        if is_advertising_channel(self.channel):
            if self.access_address != ADV_ACCESS_ADDRESS:
                raise AssertionError(
                    "advertising channel configured with a non-standard access address"
                )
            if self.crc_init != ADV_CRC_INIT:
                raise AssertionError(
                    "advertising channel configured with a non-standard CRC init"
                )

    @property
    def is_advertising(self) -> bool:
        return is_advertising_channel(self.channel)
