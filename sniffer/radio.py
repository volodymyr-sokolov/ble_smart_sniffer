"""bladeRF 2.0 micro receiver setup, calibration and capture thread.

Receive only.  Nothing in this module enables a TX module, configures a TX
stream, or writes to a TX gain stage; `libbladerf` does not even bind the
transmit entry points.

Layout of responsibilities:

* :class:`RadioConfig`   -- everything the operator can choose
* :class:`BladeRF`       -- device handle, tuning, calibration, health telemetry
* :class:`IQCorrector`   -- receiver DC offset and quadrature imbalance removal
* :class:`CaptureThread` -- fills preallocated buffers and pushes them; allocates
  nothing in the steady state and never touches a sample value

`CaptureThread` is the simple single-process capture path, used by the tests and
by standalone tools built on this module.  The GUI application does not use it:
measured on an i7-1255U, a capture *thread* running beside the DSP loop could
not reliably re-enter `bladerf_sync_rx` inside the 2 ms a 16k block allows, and
the sustained rate fell from 8.0 to 4-6.5 MSPS.  `pipeline._capture_entry` runs
the same loop alone in its own process, writing straight into a shared-memory
ring, and holds the full rate.
"""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from ctypes import byref, c_bool, c_char_p, c_float, c_int, c_int16, c_uint, c_uint64
from dataclasses import dataclass, field, replace

import numpy as np

from . import libbladerf as B
from .channels import ChannelPlan

# SC16_Q11 full scale: samples are 12-bit signed, sign-extended into int16.
SC16_FULL_SCALE = 2048.0

# libbladeRF requires the sync-stream buffer size to be a multiple of 1024.
BUFFER_GRANULARITY = 1024


@dataclass
class RadioConfig:
    """Operator-facing receiver settings.  All of these are overridable."""

    plan: ChannelPlan
    sample_rate: float = 8e6
    bandwidth: float = 3e6
    gain_db: int = 40
    agc: bool = False  # AGC off by default: gain must be known per packet
    rx_channels: tuple[int, ...] = (0,)  # RX0 only unless --dual-antenna
    block_size: int = 16384  # samples per sync_rx call
    num_buffers: int = 64
    num_transfers: int = 32
    stream_timeout_ms: int = 1000
    # Blocks of slack between capture and DSP.  256 blocks is ~0.5 s at 8 MSPS
    # (~17 MB) and measured 0.000% drops with 0 overruns on this hardware;
    # 64 dropped 1.7% on the same traffic because packet bursts stall the DSP
    # loop momentarily, and 1024 was worse again -- the working set stops
    # fitting in cache and the DSP itself slows down.
    queue_depth: int = 256
    external_clock: bool = False  # 10 MHz reference via U.FL clock input
    refclk_hz: float = 10e6
    bias_tee: bool = False
    device_id: str | None = None

    @property
    def samples_per_symbol(self) -> float:
        return self.sample_rate / 1e6

    def normalized(self) -> "RadioConfig":
        blk = max(
            BUFFER_GRANULARITY,
            int(round(self.block_size / BUFFER_GRANULARITY)) * BUFFER_GRANULARITY,
        )
        return replace(self, block_size=blk)


@dataclass
class ClockStatus:
    """Reference-clock state, and therefore whether ppm features mean anything."""

    external_selected: bool = False
    pll_enabled: bool = False
    pll_locked: bool = False
    refclk_hz: float = 0.0
    detail: str = "onboard VCTCXO"

    @property
    def calibrated(self) -> bool:
        """True only when an external reference is selected *and* locked."""
        return bool(self.external_selected and self.pll_locked)

    @property
    def label(self) -> str:
        return "GPSDO locked" if self.calibrated else "UNCALIBRATED"


@dataclass
class RadioHealth:
    """Telemetry sampled about once a second alongside the sample stream."""

    rfic_temperature_c: float = float("nan")
    timestamp: float = 0.0
    clock: ClockStatus = field(default_factory=ClockStatus)
    gain_db: int = 0
    clipping: bool = False
    peak_dbfs: float = float("-inf")


class IQCorrector:
    """Receiver DC-offset and quadrature-imbalance correction.

    Estimated from received noise (not from a transmitted tone), by
    Gram-Schmidt orthogonalisation of the I and Q rails.  Applied before any
    feature is computed so that receiver impairments are not later attributed to
    the transmitter -- an uncorrected 1% gain imbalance shows up as a phantom
    per-device modulation-index bias of about the same size as the real
    between-device spread.
    """

    def __init__(self) -> None:
        self.dc = 0.0 + 0.0j
        self.theta = 0.0  # quadrature skew
        self.gain = 1.0  # Q/I amplitude ratio
        self.valid = False
        self._n = 0

    def estimate(self, iq: np.ndarray, adapt: float = 0.1) -> None:
        """Update the estimate from a block of samples.

        `adapt` is the weight given to this block; the estimate is a running
        average so a single noisy block cannot move it far.
        """
        if iq.size < 1024:
            return
        dc = complex(np.mean(iq))
        centred = iq - dc
        i = centred.real
        q = centred.imag
        eii = float(np.mean(i * i))
        if eii <= 0:
            return
        eiq = float(np.mean(i * q))
        theta = eiq / eii
        q1 = q - theta * i
        eqq = float(np.mean(q1 * q1))
        if eqq <= 0:
            return
        gain = float(np.sqrt(eii / eqq))
        if not np.isfinite(gain) or not 0.5 < gain < 2.0:
            return

        w = 1.0 if not self.valid else adapt
        self.dc = (1 - w) * self.dc + w * dc
        self.theta = (1 - w) * self.theta + w * theta
        self.gain = (1 - w) * self.gain + w * gain
        self.valid = True
        self._n += 1

    def apply(self, iq: np.ndarray) -> np.ndarray:
        """Return `iq` with DC offset removed and the Q rail orthonormalised."""
        if not self.valid:
            return iq - complex(np.mean(iq)) if iq.size else iq
        centred = iq - self.dc
        i = centred.real
        q = (centred.imag - self.theta * i) * self.gain
        return (i + 1j * q).astype(np.complex64, copy=False)

    def summary(self) -> dict:
        return {
            "dc_i": float(self.dc.real),
            "dc_q": float(self.dc.imag),
            "quadrature_skew": float(self.theta),
            "gain_imbalance": float(self.gain),
            "valid": self.valid,
            "blocks": self._n,
        }


class BladeRF:
    """Open device handle with the receive chain configured for BLE."""

    def __init__(self, cfg: RadioConfig, log=print) -> None:
        self.cfg = cfg.normalized()
        self.log = log
        self._dev = ctypes.c_void_p()
        self._open = False
        self._enabled: list[int] = []
        self.clock = ClockStatus()
        self.info: dict = {}
        self.actual_sample_rate = 0.0
        self.actual_bandwidth = 0.0
        self._temp_cache = float("nan")
        self._temp_time = 0.0
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- open
    def open(self) -> None:
        L = B.lib()
        ident = self.cfg.device_id.encode() if self.cfg.device_id else None
        B.check(L.bladerf_open(byref(self._dev), ident), "bladerf_open")
        self._open = True

        name = L.bladerf_get_board_name(self._dev)
        board = name.decode() if name else "unknown"
        serial = ctypes.create_string_buffer(34)
        L.bladerf_get_serial(self._dev, serial)

        fw, fpga = B.BladerfVersion(), B.BladerfVersion()
        L.bladerf_fw_version(self._dev, byref(fw))
        L.bladerf_fpga_version(self._dev, byref(fpga))
        speed = L.bladerf_device_speed(self._dev)

        self.info = {
            "board": board,
            "serial": serial.value.decode("ascii", "replace"),
            "fw": str(fw),
            "fpga": str(fpga),
            "libbladerf": B.library_version(),
            "usb_speed": {0: "unknown", 1: "high", 2: "super", 3: "super+"}.get(speed, str(speed)),
        }
        if board != "bladerf2":
            self.log(
                f"warning: board reports as {board!r}; this application targets the "
                "bladeRF 2.0 micro (bladerf2) and some telemetry will be missing"
            )
        if speed < 2:
            self.log(
                "warning: device is not on a SuperSpeed port -- sustaining 8 MSPS "
                "over USB 2.0 is not possible, expect dropped blocks"
            )
        self.log(
            f"opened {board} serial {self.info['serial'][:8]}... "
            f"fw {self.info['fw']} fpga {self.info['fpga']} "
            f"usb {self.info['usb_speed']}-speed"
        )

    # ------------------------------------------------------------- clocking
    def configure_clock(self) -> ClockStatus:
        """Select the reference clock and report whether ppm features are valid.

        With `--external-clock` the U.FL clock input drives an on-board PLL that
        disciplines the sample clock.  If it does not lock we do not fall back
        silently: the returned status stays UNCALIBRATED and every ppm-scale
        feature is tagged accordingly all the way through to the exports.
        """
        L = B.lib()
        st = ClockStatus(refclk_hz=self.cfg.refclk_hz)

        if not self.cfg.external_clock:
            try:
                L.bladerf_set_clock_select(self._dev, B.CLOCK_SELECT_ONBOARD)
                L.bladerf_set_pll_enable(self._dev, False)
            except Exception:
                pass
            st.detail = (
                "onboard VCTCXO, no external reference: carrier-offset numbers "
                "include receiver drift"
            )
            self.clock = st
            return st

        try:
            B.check(
                L.bladerf_set_pll_refclk(self._dev, c_uint64(int(self.cfg.refclk_hz))),
                "bladerf_set_pll_refclk",
            )
            B.check(L.bladerf_set_pll_enable(self._dev, True), "bladerf_set_pll_enable")
            st.pll_enabled = True
            B.check(
                L.bladerf_set_clock_select(self._dev, B.CLOCK_SELECT_EXTERNAL),
                "bladerf_set_clock_select",
            )
            st.external_selected = True
        except B.BladeRFError as exc:
            st.detail = f"external clock requested but setup failed: {exc}"
            self.log(f"warning: {st.detail}")
            self.clock = st
            return st

        # The PLL needs a moment to acquire; poll rather than assume.
        locked = c_bool(False)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if L.bladerf_get_pll_lock_state(self._dev, byref(locked)) == 0 and locked.value:
                break
            time.sleep(0.05)
        st.pll_locked = bool(locked.value)

        if st.pll_locked:
            st.detail = f"external {self.cfg.refclk_hz/1e6:.3f} MHz reference, PLL locked"
            self.log(f"reference clock: {st.detail}")
        else:
            st.detail = (
                f"external {self.cfg.refclk_hz/1e6:.3f} MHz reference selected but PLL "
                "did NOT lock -- check the U.FL clock input; ppm features are UNCALIBRATED"
            )
            self.log(f"warning: {st.detail}")
        self.clock = st
        return st

    # ------------------------------------------------------------ RX config
    def configure_rx(self) -> None:
        L = B.lib()
        cfg = self.cfg
        rate = c_uint()
        bw = c_uint()

        for idx in cfg.rx_channels:
            ch = B.CHANNEL_RX(idx)
            B.check(
                L.bladerf_set_frequency(self._dev, ch, c_uint64(int(cfg.plan.frequency_hz))),
                "bladerf_set_frequency",
            )
            B.check(
                L.bladerf_set_sample_rate(self._dev, ch, c_uint(int(cfg.sample_rate)), byref(rate)),
                "bladerf_set_sample_rate",
            )
            B.check(
                L.bladerf_set_bandwidth(self._dev, ch, c_uint(int(cfg.bandwidth)), byref(bw)),
                "bladerf_set_bandwidth",
            )
            mode = B.BLADERF_GAIN_SLOWATTACK_AGC if cfg.agc else B.BLADERF_GAIN_MGC
            B.check(L.bladerf_set_gain_mode(self._dev, ch, mode), "bladerf_set_gain_mode")
            if not cfg.agc:
                B.check(L.bladerf_set_gain(self._dev, ch, c_int(int(cfg.gain_db))), "bladerf_set_gain")
            try:
                L.bladerf_set_bias_tee(self._dev, ch, bool(cfg.bias_tee))
            except Exception:
                pass

        self.actual_sample_rate = float(rate.value)
        self.actual_bandwidth = float(bw.value)

        # Verify the LO actually landed where the channel plan says it should.
        got = c_uint64()
        B.check(
            L.bladerf_get_frequency(self._dev, B.CHANNEL_RX(cfg.rx_channels[0]), byref(got)),
            "bladerf_get_frequency",
        )
        err = abs(float(got.value) - cfg.plan.frequency_hz)
        if err > 1000.0:
            raise AssertionError(
                f"LO landed at {got.value/1e6:.6f} MHz but channel plan "
                f"{cfg.plan.label} requires {cfg.plan.frequency_hz/1e6:.6f} MHz"
            )
        cfg.plan.assert_consistent()

        self.log(
            f"RX{list(cfg.rx_channels)} @ {got.value/1e6:.3f} MHz "
            f"({cfg.plan.label}), fs={self.actual_sample_rate/1e6:.3f} MSPS, "
            f"bw={self.actual_bandwidth/1e6:.3f} MHz, "
            f"gain={'AGC' if cfg.agc else str(cfg.gain_db)+' dB'}"
        )

    #: True when the stream carries FPGA timestamps.  See `configure_stream`.
    timestamps_available = True

    def configure_stream(self) -> None:
        L = B.lib()
        cfg = self.cfg
        layout = B.BLADERF_RX_X2 if len(cfg.rx_channels) > 1 else B.BLADERF_RX_X1

        # The metadata format is broken for MIMO on this libbladeRF (2.4.1):
        # with 32768 samples requested it returns after delivering 508 -- one
        # 512-sample sub-block less its 4 samples of header -- and reports a
        # sample rate two orders of magnitude above what the ADC can produce.
        # The plain format streams both channels correctly at full rate, so
        # dual-antenna captures use it and give up the FPGA timestamp; loss is
        # then reported from the USB overrun flag alone, and the status bar
        # says the timestamp account is unavailable.
        self.timestamps_available = len(cfg.rx_channels) == 1
        fmt = (
            B.BLADERF_FORMAT_SC16_Q11_META
            if self.timestamps_available
            else B.BLADERF_FORMAT_SC16_Q11
        )
        B.check(
            L.bladerf_sync_config(
                self._dev,
                layout,
                fmt,
                c_uint(cfg.num_buffers),
                c_uint(cfg.block_size),
                c_uint(cfg.num_transfers),
                c_uint(cfg.stream_timeout_ms),
            ),
            "bladerf_sync_config",
        )

    def enable(self, on: bool = True) -> None:
        L = B.lib()
        for idx in self.cfg.rx_channels:
            ch = B.CHANNEL_RX(idx)
            B.check(L.bladerf_enable_module(self._dev, ch, on), "bladerf_enable_module")
        self._enabled = list(self.cfg.rx_channels) if on else []

    def start(self) -> None:
        self.open()
        self.configure_clock()
        self.configure_rx()
        self.configure_stream()
        self.enable(True)

    # ------------------------------------------------------------- retuning
    def retune(self, plan: ChannelPlan) -> None:
        """Move the LO.  Caller is responsible for flushing the pipeline."""
        plan.assert_consistent()
        L = B.lib()
        with self._lock:
            for idx in self.cfg.rx_channels:
                B.check(
                    L.bladerf_set_frequency(
                        self._dev, B.CHANNEL_RX(idx), c_uint64(int(plan.frequency_hz))
                    ),
                    "bladerf_set_frequency",
                )
            got = c_uint64()
            L.bladerf_get_frequency(self._dev, B.CHANNEL_RX(self.cfg.rx_channels[0]), byref(got))
            if abs(float(got.value) - plan.frequency_hz) > 1000.0:
                raise AssertionError(
                    f"retune to {plan.label} landed at {got.value/1e6:.6f} MHz"
                )
            self.cfg = replace(self.cfg, plan=plan)
        self.log(f"retuned to {plan.label}")

    def set_gain(self, gain_db: int) -> None:
        L = B.lib()
        with self._lock:
            for idx in self.cfg.rx_channels:
                B.check(
                    L.bladerf_set_gain(self._dev, B.CHANNEL_RX(idx), c_int(int(gain_db))),
                    "bladerf_set_gain",
                )
            self.cfg = replace(self.cfg, gain_db=int(gain_db))

    def gain_range(self, idx: int = 0) -> tuple[float, float, float]:
        rng = ctypes.POINTER(B.BladerfRange)()
        L = B.lib()
        if L.bladerf_get_gain_range(self._dev, B.CHANNEL_RX(idx), byref(rng)) != 0:
            return (0.0, 60.0, 1.0)
        return rng.contents.as_tuple()

    # ------------------------------------------------------------ telemetry
    def rfic_temperature(self, max_age: float = 1.0) -> float:
        """RFIC die temperature in C, sampled at most once per `max_age` seconds."""
        now = time.monotonic()
        if now - self._temp_time < max_age:
            return self._temp_cache
        val = c_float()
        try:
            if B.lib().bladerf_get_rfic_temperature(self._dev, byref(val)) == 0:
                self._temp_cache = float(val.value)
            else:
                self._temp_cache = float("nan")
        except Exception:
            self._temp_cache = float("nan")
        self._temp_time = now
        return self._temp_cache

    def pll_locked(self) -> bool:
        if not self.cfg.external_clock:
            return False
        locked = c_bool(False)
        try:
            if B.lib().bladerf_get_pll_lock_state(self._dev, byref(locked)) == 0:
                self.clock.pll_locked = bool(locked.value)
        except Exception:
            pass
        return self.clock.pll_locked

    def timestamp(self) -> int:
        ts = c_uint64()
        try:
            B.lib().bladerf_get_timestamp(self._dev, B.BLADERF_RX, byref(ts))
        except Exception:
            return 0
        return int(ts.value)

    # ----------------------------------------------------------- shutdown
    def close(self) -> None:
        if not self._open:
            return
        try:
            self.enable(False)
        except Exception:
            pass
        try:
            B.lib().bladerf_close(self._dev)
        finally:
            self._open = False
            self._dev = ctypes.c_void_p()

    def __enter__(self) -> "BladeRF":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class SampleBlock:
    """One block of received IQ handed from the capture thread to the DSP loop.

    Carried as raw interleaved int16 exactly as the FPGA delivered it.  The
    conversion to complex64 happens in the consumer via :meth:`as_complex`, so
    the capture thread never spends GIL time on arithmetic.
    """

    raw: np.ndarray  # interleaved int16: I,Q[,I,Q per extra channel]
    n_channels: int
    timestamp: int  # FPGA sample counter of the first sample
    wall_time: float
    gain_db: int
    temperature_c: float
    channel: int
    frequency_hz: float
    calibrated: bool
    peak: float  # peak |s| in this block, 0..1+
    seq: int
    epoch: int  # increments on every retune; marks a pipeline discontinuity

    @property
    def n_samples(self) -> int:
        return self.raw.size // (2 * self.n_channels)

    def as_complex(self, channel: int = 0) -> np.ndarray:
        """Scaled complex64 view of one RX channel.

        SC16_Q11 is 12-bit signed left-aligned in int16; dividing by 2048 puts
        full scale at |s| = 1 so every downstream level is expressible in dBFS.
        """
        scale = np.float32(1.0 / SC16_FULL_SCALE)
        if self.n_channels == 1:
            v = self.raw.reshape(-1, 2)
        else:
            # RX_X2 interleaves channels sample-by-sample: I0 Q0 I1 Q1 ...
            v = self.raw.reshape(-1, self.n_channels, 2)[:, channel, :]
        out = np.empty(v.shape[0], dtype=np.complex64)
        out.real = v[:, 0]
        out.imag = v[:, 1]
        return out * scale


class CaptureThread(threading.Thread):
    """Fills preallocated buffers from `bladerf_sync_rx` and pushes them on.

    Steady-state allocation is confined to one complex64 view per block, which
    the DSP stage consumes and releases; the int16 staging buffers are allocated
    once up front and reused forever.  Nothing here inspects sample values
    beyond a vectorised peak for the clipping indicator.
    """

    def __init__(
        self,
        radio: BladeRF,
        out_queue: "queue.Queue[SampleBlock]",
        stats: "CaptureStats",
        pool_size: int = 8,
    ) -> None:
        super().__init__(name="bladerf-capture", daemon=True)
        self.radio = radio
        self.out = out_queue
        self.stats = stats
        self._stop = threading.Event()
        self._flush = threading.Event()
        self.epoch = 0

        n = radio.cfg.block_size
        nch = len(radio.cfg.rx_channels)
        # Preallocated int16 staging buffers, round-robin so the DSP stage can
        # still be reading buffer k while we fill k+1.
        self._pool = [np.empty(n * 2 * nch, dtype=np.int16) for _ in range(pool_size)]
        self._pool_idx = 0
        self._meta = B.BladerfMetadata()

    def request_flush(self, epoch: int) -> None:
        """Drop everything in flight; called on retune."""
        self.epoch = epoch
        self._flush.set()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            from .pipeline import raise_thread_priority

            raise_thread_priority()
        except Exception:
            pass
        L = B.lib()
        cfg = self.radio.cfg
        n = cfg.block_size
        nch = len(cfg.rx_channels)
        timeout = c_uint(cfg.stream_timeout_ms)
        seq = 0
        scale = 1.0 / SC16_FULL_SCALE
        temp = float("nan")
        last_temp = 0.0

        while not self._stop.is_set():
            buf = self._pool[self._pool_idx]
            self._pool_idx = (self._pool_idx + 1) % len(self._pool)

            self._meta.flags = B.BLADERF_META_FLAG_RX_NOW
            self._meta.status = 0
            self._meta.actual_count = 0
            rc = L.bladerf_sync_rx(
                self.radio._dev,
                buf.ctypes.data_as(ctypes.c_void_p),
                c_uint(n),
                byref(self._meta),
                timeout,
            )
            if rc < 0:
                if self._stop.is_set():
                    break
                self.stats.errors += 1
                self.stats.last_error = B.strerror(rc)
                # A timeout is recoverable; anything else is worth backing off on.
                time.sleep(0.01)
                continue

            if self._meta.status & B.BLADERF_META_STATUS_OVERRUN:
                self.stats.overruns += 1

            if self._flush.is_set():
                self._flush.clear()
                self.stats.flushed += 1
                continue

            count = int(self._meta.actual_count) or n
            count = min(count, n)

            now = time.monotonic()
            if now - last_temp >= 1.0:
                temp = self.radio.rfic_temperature(max_age=0.0)
                last_temp = now

            # Everything this thread does holds the GIL, and every microsecond
            # it holds is a microsecond the DSP stage is not running.  So the
            # int16 -> complex64 conversion is deferred to the consumer, and the
            # only work here is one memcpy out of the reuse pool plus a peak
            # taken on the raw integers -- no float conversion, no allocation
            # beyond the copy the consumer needs to own anyway.
            raw = buf[: count * 2 * nch]
            peak = float(np.abs(raw).max()) / SC16_FULL_SCALE if raw.size else 0.0

            block = SampleBlock(
                raw=raw.copy(),
                n_channels=nch,
                timestamp=int(self._meta.timestamp),
                wall_time=time.time(),
                gain_db=cfg.gain_db,
                temperature_c=temp,
                channel=cfg.plan.channel,
                frequency_hz=cfg.plan.frequency_hz,
                calibrated=self.radio.clock.calibrated,
                peak=peak,
                seq=seq,
                epoch=self.epoch,
            )
            seq += 1
            self.stats.blocks += 1
            self.stats.samples += count

            # Bounded queue, drop-oldest: falling behind must be visible, never
            # silent, and must not turn into unbounded memory growth.
            try:
                self.out.put_nowait(block)
            except queue.Full:
                try:
                    self.out.get_nowait()
                    self.stats.dropped_blocks += 1
                except queue.Empty:
                    pass
                try:
                    self.out.put_nowait(block)
                except queue.Full:
                    self.stats.dropped_blocks += 1


@dataclass
class CaptureStats:
    blocks: int = 0
    samples: int = 0
    dropped_blocks: int = 0
    overruns: int = 0
    errors: int = 0
    flushed: int = 0
    last_error: str = ""

    def as_dict(self) -> dict:
        return {
            "blocks": self.blocks,
            "samples": self.samples,
            "dropped_blocks": self.dropped_blocks,
            "overruns": self.overruns,
            "errors": self.errors,
            "drop_rate": (self.dropped_blocks / self.blocks) if self.blocks else 0.0,
            "last_error": self.last_error,
        }


def dbfs(x: float) -> float:
    """Amplitude (0..1 of full scale) to dBFS."""
    return 20.0 * np.log10(max(x, 1e-12))
