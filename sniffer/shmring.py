"""Single-producer / single-consumer shared-memory ring for raw IQ blocks.

This exists because a capture thread and a DSP loop cannot share a Python
process and both meet a 2 ms deadline.  Measured on an i7-1255U: capture alone
sustains 8.00 MSPS; the same capture thread beside the DSP loop in one process
falls to 4-6.5 MSPS, because the two contend for one interpreter lock and the
capture thread misses its window to re-enter `bladerf_sync_rx`.  Separating them
into processes removes the contention entirely -- there is no lock to share.

The ring is lock-free by construction: one writer, one reader, and a monotonic
64-bit write cursor that only the writer advances.  A reader that falls more
than `capacity` blocks behind detects it by comparing cursors and reports the
loss rather than reading torn data.

It doubles as the raw-IQ history the operator dumps to SigMF, so the samples
behind any packet are already resident and no second copy is kept.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

# Per-slot metadata, kept in its own small shared block so the sample buffer
# stays a plain int16 array that numpy can view without any offset arithmetic.
META_DTYPE = np.dtype(
    [
        ("timestamp", np.uint64),  # FPGA sample counter of the first sample
        ("wall_time", np.float64),
        ("n_samples", np.int32),
        ("epoch", np.int32),
        ("gain_db", np.int32),
        ("channel", np.int32),
        ("temperature_c", np.float32),
        ("peak", np.float32),
        ("frequency_hz", np.float64),
        ("calibrated", np.int32),
        # 0 when the stream carries no FPGA timestamp (the MIMO path), so the
        # reader knows not to treat the counter as evidence of continuity.
        ("timestamp_valid", np.int32),
    ]
)


@dataclass
class RingSpec:
    """Everything a child process needs to re-attach to an existing ring."""

    data_name: str
    meta_name: str
    capacity: int
    block_size: int
    n_channels: int
    sample_rate: float


class SharedRing:
    """Fixed-capacity ring of interleaved int16 IQ blocks."""

    def __init__(
        self,
        capacity: int,
        block_size: int,
        n_channels: int = 1,
        sample_rate: float = 8e6,
        create: bool = True,
        spec: RingSpec | None = None,
    ) -> None:
        if spec is not None:
            capacity = spec.capacity
            block_size = spec.block_size
            n_channels = spec.n_channels
            sample_rate = spec.sample_rate

        self.capacity = capacity
        self.block_size = block_size
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.stride = block_size * 2 * n_channels  # int16 elements per slot

        nbytes = self.stride * capacity * 2  # int16 -> 2 bytes
        meta_bytes = META_DTYPE.itemsize * capacity

        if create:
            self._data_shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self._meta_shm = shared_memory.SharedMemory(create=True, size=meta_bytes)
            self._owner = True
        else:
            assert spec is not None
            self._data_shm = shared_memory.SharedMemory(name=spec.data_name)
            self._meta_shm = shared_memory.SharedMemory(name=spec.meta_name)
            self._owner = False

        self.data = np.ndarray(
            (capacity, self.stride), dtype=np.int16, buffer=self._data_shm.buf
        )
        self.meta = np.ndarray((capacity,), dtype=META_DTYPE, buffer=self._meta_shm.buf)

        # Only the writer advances this; the reader only ever reads it.
        self.cursor = mp.Value(ctypes.c_int64, 0, lock=False)

    # ------------------------------------------------------------------
    @property
    def spec(self) -> RingSpec:
        return RingSpec(
            self._data_shm.name,
            self._meta_shm.name,
            self.capacity,
            self.block_size,
            self.n_channels,
            self.sample_rate,
        )

    def slot_view(self, index: int) -> np.ndarray:
        """Writable int16 view of one slot -- `bladerf_sync_rx` writes here."""
        return self.data[index % self.capacity]

    def publish(self, index: int, **meta) -> None:
        """Make slot `index` visible to the reader.

        The metadata is written before the cursor is advanced, so a reader that
        sees the new cursor always sees complete metadata for it.
        """
        m = self.meta[index % self.capacity]
        for k, v in meta.items():
            m[k] = v
        self.cursor.value = index + 1

    def written(self) -> int:
        return int(self.cursor.value)

    def read(self, index: int) -> tuple[np.ndarray, np.void] | None:
        """Return (samples, metadata) for `index`, or None if it has been lapped."""
        w = self.written()
        if index >= w:
            return None
        if w - index > self.capacity:
            return None  # lapped: the writer has overwritten this slot
        slot = index % self.capacity
        m = self.meta[slot]
        n = int(m["n_samples"])
        return self.data[slot, : n * 2 * self.n_channels], m

    def read_verified(self, index: int) -> tuple[np.ndarray, np.void] | None:
        """Copy out slot `index`, then confirm the writer did not lap it.

        A lock-free single-producer ring is only safe if the reader checks
        *after* copying: the writer can overwrite the slot between the freshness
        test and the copy, and the metadata handed back is a live view into
        shared memory, not a snapshot.  Reading a torn slot yields a timestamp
        from a much later block, which then looks like an enormous gap in the
        radio's sample counter -- the receiver gets blamed for the host being
        slow.
        """
        w = self.written()
        if index >= w or w - index > self.capacity:
            return None
        slot = index % self.capacity
        meta = self.meta[slot].copy()
        n = int(meta["n_samples"])
        if n <= 0 or n > self.block_size:
            return None
        data = self.data[slot, : n * 2 * self.n_channels].copy()
        # Re-check: if the writer has come all the way round since we started,
        # what we copied may be a mixture of two blocks.
        if self.written() - index > self.capacity:
            return None
        return data, meta

    def read_samples(self, abs_start: int, count: int) -> np.ndarray | None:
        """Complex64 by absolute sample index, spanning slots if necessary.

        This is what backs the SigMF dump of the samples behind a packet.
        """
        w = self.written()
        if w == 0 or count <= 0:
            return None
        bs = self.block_size
        first_block = max(w - self.capacity, 0)
        start_block = abs_start // bs
        end_block = (abs_start + count - 1) // bs
        if start_block < first_block or end_block >= w:
            return None
        out = np.empty(count, dtype=np.complex64)
        got = 0
        pos = abs_start
        while got < count:
            b = pos // bs
            off = pos - b * bs
            take = min(bs - off, count - got)
            r = self.read(b)
            if r is None:
                return None
            raw, _ = r
            v = raw.reshape(-1, self.n_channels, 2)[off : off + take, 0, :]
            out[got : got + take] = (
                v[:, 0].astype(np.float32) + 1j * v[:, 1].astype(np.float32)
            ) / 2048.0
            got += take
            pos += take
        return out

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Drop the numpy views before releasing the buffers, or the shared
        # memory refuses to close with "cannot close exported pointers exist".
        self.data = None
        self.meta = None
        try:
            self._data_shm.close()
            self._meta_shm.close()
        except Exception:
            pass
        if self._owner:
            for shm in (self._data_shm, self._meta_shm):
                try:
                    shm.unlink()
                except Exception:
                    pass
