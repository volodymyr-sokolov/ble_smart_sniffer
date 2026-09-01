"""Process orchestration, backpressure accounting and the raw-IQ ring buffer.

Three stages as specified, with the split placed where it actually buys
something:

    capture thread  --(bounded queue)-->  DSP loop      : same process
    DSP loop        --(bounded queue)-->  GUI           : separate process

The capture thread and the DSP loop share a process on purpose.
`bladerf_sync_rx` spends its time inside libusb with the GIL released, so a
thread there costs almost nothing and avoids copying 32 MB/s across a process
boundary.  The GUI is what must never be blocked by DSP, and that is the
boundary a process is spent on.

Every queue is bounded and every drop is counted.  A sniffer that quietly falls
behind is worse than one that captures less, so `PipelineStats` carries separate
counters for USB overruns, capture-queue drops and GUI-queue drops, and the
status bar shows all three.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .analysis import InterferenceMonitor, LiveAnalyzer
from .channels import ChannelPlan
from .dsp import Demodulator, bits_to_bytes
from .features import extract_features
from .packet import PacketRecord, make_event
from .radio import BladeRF, CaptureStats, IQCorrector, RadioConfig, SC16_FULL_SCALE
from .shmring import RingSpec, SharedRing


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

@dataclass
class PipelineStats:
    blocks: int = 0
    samples: int = 0
    usb_overruns: int = 0
    capture_drops: int = 0  # capture -> DSP queue
    gui_drops: int = 0  # DSP -> GUI queue
    radio_errors: int = 0
    lost_samples: int = 0  # never captured: gaps in the FPGA timestamp sequence
    gap_events: int = 0
    skipped_samples: int = 0  # captured, but the DSP fell behind and lapped them
    skip_events: int = 0
    packets: int = 0
    crc_ok: int = 0
    events: int = 0
    features_degraded: int = 0
    packets_per_s: float = 0.0
    dsp_load: float = 0.0  # fraction of real time spent in the DSP loop
    noise_floor_dbfs: float = float("nan")
    last_rssi_dbfs: float = float("nan")
    temperature_c: float = float("nan")
    peak_dbfs: float = float("nan")
    clipping: bool = False
    calibrated: bool = False
    clock_detail: str = ""
    channel: int = 37
    frequency_hz: float = 2.402e9
    gain_db: int = 0
    sample_rate: float = 8e6
    running: bool = False
    epoch: int = 0
    ring_seconds: float = 0.0
    interference: dict = field(default_factory=dict)
    message: str = ""

    @property
    def crc_rate(self) -> float:
        return self.crc_ok / self.packets if self.packets else 0.0

    @property
    def drop_rate(self) -> float:
        return (self.capture_drops / self.blocks) if self.blocks else 0.0



@dataclass
class DetectionMsg:
    """What the DSP stage hands to the feature stage.

    Deliberately compact and free of numpy object arrays so it pickles quickly:
    at a few hundred packets a second this crosses a process boundary, and the
    IQ slice is already the bulk of it.
    """

    number: int
    timestamp_us: float
    wall_time: float
    abs_sync: int
    abs_slice_start: int
    sync_in_slice: int
    sym_offset: float
    n_symbols: int
    epoch: int
    channel: int
    frequency_hz: float
    access_address: int
    gain_db: int
    temperature_c: float
    calibrated: bool
    noise_floor_dbfs: float
    sync_score: float
    corr_peak: float
    full_features: bool
    keep_iq: bool
    bits: "np.ndarray"
    sym_freq: "np.ndarray"
    iq: "np.ndarray"
    iq_second: "np.ndarray | None"
    sample_rate: float
    # decoded PDU, flattened
    pdu_type: int
    pdu_name: str
    adva: str
    adva_bytes: bytes
    adva_kind: str
    tx_add_random: bool
    rx_add_random: bool
    length: int
    payload: bytes
    raw_bytes: bytes
    ad_structures: list
    info: str
    crc_received: int
    crc_computed: int
    crc_ok: bool


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

@dataclass
class Command:
    kind: str
    payload: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------

class SnifferWorker:
    """Capture + DSP + features + analysis.  Runs in its own process."""

    def __init__(
        self,
        cfg: RadioConfig,
        ring,
        out_q,
        cmd_q,
        stat_q,
        cap_ctrl_q,
        enroll: tuple[str, ...] = (),
        keep_iq: bool = True,
        rssi_cal_db: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.ring = ring
        self.out_q = out_q
        self.cmd_q = cmd_q
        self.stat_q = stat_q
        self.cap_ctrl_q = cap_ctrl_q
        self.enroll = enroll
        self.keep_iq = keep_iq
        self.rssi_cal_db = rssi_cal_db

        self.stats = PipelineStats(
            channel=cfg.plan.channel,
            frequency_hz=cfg.plan.frequency_hz,
            gain_db=cfg.gain_db,
            sample_rate=cfg.sample_rate,
        )
        self._stop = threading.Event()
        self._epoch = 0
        self._number = 0
        self._abs_samples = 0
        self._t0 = 0.0

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Stage 2: read the ring, demodulate, hand detections to stage 3."""
        raise_process_priority()
        pin = pin_to_performance_cores(stage="dsp")
        self._log(f"dsp process affinity: {pin}")
        try:
            self._loop()
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.stats.message = f"{type(exc).__name__}: {exc}"
            self._log("dsp worker failed: " + self.stats.message)
            self._log(traceback.format_exc())
        finally:
            self.stats.running = False
            self._push_stats(force=True)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        cfg = self.cfg
        ring = self.ring
        demod = Demodulator(
            cfg.sample_rate,
            access_address=cfg.plan.access_address,
            channel=cfg.plan.channel,
            crc_init=cfg.plan.crc_init,
        )
        corrector = IQCorrector()
        interference = InterferenceMonitor(cfg.sample_rate)
        self.stats.ring_seconds = ring.capacity * ring.block_size / cfg.sample_rate
        self.stats.running = True

        read_idx = 0
        last_timestamp = -1
        last_status = 0.0
        last_pkt_count = 0
        last_pkt_time = time.monotonic()
        dsp_time = 0.0
        window_start = last_pkt_time
        rep = interference.last_report
        batch: list = []
        nblocks = 0
        last_cmd_poll = 0.0

        while not self._stop.is_set():
            now_cmd = time.monotonic()
            if now_cmd - last_cmd_poll >= 0.02:
                last_cmd_poll = now_cmd
                self._drain_commands(demod)

            written = ring.written()
            if read_idx >= written:
                time.sleep(0.0005)
                if time.monotonic() - last_status >= 0.25:
                    last_status = time.monotonic()
                    self._push_stats()
                continue

            # Falling more than the ring behind is real block loss.  Skip to the
            # oldest slot that is still intact and count what was missed, rather
            # than reading a slot the capture process is overwriting.
            if written - read_idx > ring.capacity:
                # These samples reached the ring intact; this stage could not
                # keep up and lapped them.  That is a different failure from a
                # sample the radio never delivered, and conflating the two would
                # misreport a slow host as a bad receiver -- so they are counted
                # separately and both are shown.
                lost_blocks = written - read_idx - ring.capacity
                self.stats.capture_drops += lost_blocks
                self.stats.skipped_samples += lost_blocks * ring.block_size
                self.stats.skip_events += 1
                read_idx = written - ring.capacity
                demod.reset_stream()
                last_timestamp = -1  # the discontinuity is ours, not the radio's

            got = ring.read_verified(read_idx)
            if got is None:
                # The writer lapped this slot between the check above and this
                # read.  That is our own lateness, not a radio gap, so it counts
                # as skipped and the timestamp reference is invalidated -- if it
                # were left stale the next good block would look like a
                # hardware discontinuity of however long we were behind.
                self.stats.skipped_samples += ring.block_size
                self.stats.capture_drops += 1
                last_timestamp = -1
                read_idx += 1
                continue
            raw, meta = got
            read_idx += 1
            nblocks += 1

            # Continuity check against the FPGA sample counter.  This is the
            # only trustworthy account of what was missed: the capture queue
            # can report zero drops and libbladeRF zero overruns while samples
            # are still absent, because a late `sync_rx` loses them upstream of
            # both counters.  A gap here is proof, not inference.
            ts = int(meta["timestamp"])
            n_samp = int(meta["n_samples"])
            if not int(meta["timestamp_valid"]):
                # No hardware timestamp on this stream; skip the continuity
                # check rather than report a fabricated 0% loss.
                last_timestamp = -1
            # `last_timestamp` is the sample index one past the previous block,
            # so a contiguous stream satisfies ts == last_timestamp exactly and
            # any excess is the number of samples that never arrived.  The
            # earlier form subtracted this block's length as well, which quietly
            # forgave every single-block gap.
            if last_timestamp >= 0 and ts > last_timestamp:
                self.stats.lost_samples += ts - last_timestamp
                self.stats.gap_events += 1
                demod.reset_stream()
            last_timestamp = ts + n_samp

            if int(meta["epoch"]) != self._epoch:
                self._abs_samples += int(meta["n_samples"])
                continue

            t_start = time.perf_counter()

            n_ch = ring.n_channels
            frames = raw.reshape(-1, n_ch, 2)
            v = frames[:, 0, :]
            iq = np.empty(v.shape[0], dtype=np.complex64)
            iq.real = v[:, 0]
            iq.imag = v[:, 1]
            iq *= np.float32(1.0 / 2048.0)
            iq2 = None
            if n_ch > 1:
                w = frames[:, 1, :]
                iq2 = np.empty(w.shape[0], dtype=np.complex64)
                iq2.real = w[:, 0]
                iq2.imag = w[:, 1]
                iq2 *= np.float32(1.0 / 2048.0)

            if nblocks % 8 == 0:
                corrector.estimate(iq)
            iq = corrector.apply(iq)

            dets = demod.process_stream(iq, self._abs_samples)
            self._abs_samples += int(meta["n_samples"])

            if nblocks % 16 == 0:
                rep = interference.observe_block(iq, demod.noise.dbfs)

            backlog = (ring.written() - read_idx) / max(ring.capacity, 1)
            want_full = backlog < 0.5

            for abs_sync, det in dets:
                msg = self._build_message(det, abs_sync, meta, demod, want_full, iq2)
                if msg is None:
                    continue
                if not msg.full_features:
                    self.stats.features_degraded += 1
                self.stats.packets += 1
                if msg.crc_ok:
                    self.stats.crc_ok += 1
                batch.append(msg)

            dsp_time += time.perf_counter() - t_start
            self.stats.blocks = nblocks
            self.stats.samples += int(meta["n_samples"])
            self.stats.peak_dbfs = 20 * np.log10(max(float(meta["peak"]), 1e-9))
            self.stats.clipping = float(meta["peak"]) > 0.98
            self.stats.temperature_c = float(meta["temperature_c"])
            self.stats.noise_floor_dbfs = demod.noise.dbfs
            self.stats.channel = int(meta["channel"])
            self.stats.frequency_hz = float(meta["frequency_hz"])
            self.stats.gain_db = int(meta["gain_db"])
            self.stats.calibrated = bool(meta["calibrated"])

            now = time.monotonic()
            if batch and (len(batch) >= 32 or now - last_status > 0.05):
                self._send(batch)
                batch = []
            if now - window_start >= 1.0:
                self.stats.dsp_load = dsp_time / (now - window_start)
                dsp_time = 0.0
                window_start = now
            if now - last_pkt_time >= 1.0:
                self.stats.packets_per_s = (
                    (self.stats.packets - last_pkt_count) / (now - last_pkt_time)
                )
                last_pkt_count = self.stats.packets
                last_pkt_time = now
            if now - last_status >= 0.25:
                self.stats.interference = rep.as_dict()
                self.stats.epoch = self._epoch
                self._push_stats()
                last_status = now

        if batch:
            self._send(batch)

    def _build_message(self, det, abs_sync, meta, demod, full: bool, iq2=None):
        """Package one detection for the feature process.  No feature work here."""
        sl = det.iq_slice
        if sl is None or sl.size < 64:
            return None
        # With --dual-antenna the same sample range from RX1 travels with the
        # packet so the feature stage can measure the inter-antenna phase.  Both
        # channels share one LO, so the pair is coherent and the phase
        # difference is meaningful; they are not two BLE channels.
        second = None
        if iq2 is not None:
            lo = det.slice_start
            hi = min(det.slice_end, iq2.size)
            if hi - lo >= 64:
                second = iq2[lo:hi].copy()
        pdu = det.pdu
        self._number += 1
        return DetectionMsg(
            number=self._number,
            timestamp_us=(abs_sync / demod.sample_rate) * 1e6,
            wall_time=float(meta["wall_time"]),
            abs_sync=abs_sync,
            abs_slice_start=det.abs_slice_start,
            sync_in_slice=det.sync_index - det.slice_start,
            sym_offset=det.sym_offset,
            n_symbols=det.n_symbols,
            epoch=self._epoch,
            channel=int(meta["channel"]),
            frequency_hz=float(meta["frequency_hz"]),
            access_address=self.cfg.plan.access_address,
            gain_db=int(meta["gain_db"]),
            temperature_c=float(meta["temperature_c"]),
            calibrated=bool(meta["calibrated"]),
            noise_floor_dbfs=demod.noise.dbfs,
            sync_score=det.sync_score,
            corr_peak=det.corr_peak,
            full_features=full,
            keep_iq=self.keep_iq,
            bits=det.bits,
            sym_freq=det.freq_symbols,
            iq=sl,
            iq_second=second,
            sample_rate=demod.sample_rate,
            pdu_type=pdu.pdu_type,
            pdu_name=pdu.pdu_name,
            adva=pdu.adva_str,
            adva_bytes=pdu.adva,
            adva_kind=pdu.adva_kind,
            tx_add_random=pdu.tx_add_random,
            rx_add_random=pdu.rx_add_random,
            length=pdu.length,
            payload=pdu.payload,
            raw_bytes=pdu.raw,
            ad_structures=pdu.ad_structures,
            info=pdu.info,
            crc_received=pdu.crc_received,
            crc_computed=pdu.crc_computed,
            crc_ok=pdu.crc_ok,
        )

    # ------------------------------------------------------------------
    def _send(self, batch: list) -> None:
        try:
            self.out_q.put_nowait(batch)
        except queue.Full:
            # Drop-oldest, and say so.  The alternative -- blocking here --
            # would stall the DSP loop and turn a GUI hiccup into sample loss.
            try:
                self.out_q.get_nowait()
                self.stats.gui_drops += 1
            except queue.Empty:
                pass
            try:
                self.out_q.put_nowait(batch)
            except queue.Full:
                self.stats.gui_drops += len(batch)

    def _push_stats(self, force: bool = False) -> None:
        try:
            self.stat_q.put_nowait(asdict(self.stats))
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        try:
            self.stat_q.put_nowait({"log": str(msg)})
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _drain_commands(self, demod) -> None:
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return
            try:
                self._handle(cmd, demod)
            except Exception as exc:
                self._log(f"command {cmd.kind} failed: {exc}")

    def _handle(self, cmd: Command, demod) -> None:
        if cmd.kind == "stop":
            self._stop.set()
            try:
                self.cap_ctrl_q.put_nowait(Command("stop"))
            except Exception:
                pass
            return

        if cmd.kind == "retune":
            plan = ChannelPlan.from_args(
                channel=cmd.payload.get("channel"),
                freq_hz=cmd.payload.get("freq_hz"),
                access_address=cmd.payload.get("access_address"),
                crc_init=cmd.payload.get("crc_init"),
            )
            self._epoch += 1
            # The radio is retuned by the capture process, which owns the device
            # handle; this side reconfigures the decoder.  Everything
            # channel-specific moves together -- whitening seed, sync template,
            # noise floor and stream continuity -- because a baseline carried
            # across a retune is a baseline for the wrong channel.
            try:
                self.cap_ctrl_q.put_nowait(Command("retune", dict(cmd.payload)))
            except Exception:
                pass
            demod.reconfigure(plan.channel, plan.access_address, plan.crc_init)
            self.cfg = replace(self.cfg, plan=plan)
            self._number += 1
            ev = make_event(
                self._number,
                (self._abs_samples / demod.sample_rate) * 1e6,
                plan.channel,
                "channel change",
                f"retuned to {plan.label} -- baselines and noise floor reset",
                self._epoch,
            )
            self._send([ev])
            self.stats.events += 1
            self.stats.channel = plan.channel
            self.stats.frequency_hz = plan.frequency_hz
            self._log(f"retuned: {plan.label}")
            return

        if cmd.kind == "gain":
            try:
                self.cap_ctrl_q.put_nowait(Command("gain", dict(cmd.payload)))
            except Exception:
                pass
            self.stats.gain_db = int(cmd.payload["gain_db"])
            return

        if cmd.kind == "clear":
            self.stats.packets = 0
            self.stats.crc_ok = 0
            self.stats.events = 0
            self._number = 0
            return




# --------------------------------------------------------------------------
# scheduling helpers
# --------------------------------------------------------------------------

def pin_to_performance_cores(mask=None, stage: str = "") -> str:
    """Pin this process to the fastest CPUs available.

    Hybrid laptop CPUs (Intel 12th gen and later) expose performance and
    efficiency cores as one flat set, and the scheduler is free to park a
    background process on the efficiency cores.  For a capture loop with a 2 ms
    deadline that is the difference between keeping up and not: measured on an
    i7-1255U, pinning the pipeline to the four performance-core threads raised
    the sustained rate from about 3.9 to 6.6 MSPS before the capture stage was
    given its own process.

    Set SNIFFER_CPU_MASK to override (a hex affinity mask), or 0 to disable.
    """
    # Per-stage override, e.g. SNIFFER_CPU_MASK_CAPTURE=0xF.  With the three
    # stages in separate processes, pinning them all to the same performance
    # cores makes them compete for exactly the cores they were pinned to; only
    # the capture stage has a hard deadline, so by default only it is pinned.
    env = os.environ.get(f"SNIFFER_CPU_MASK_{stage.upper()}") if stage else None
    if env is None:
        env = os.environ.get("SNIFFER_CPU_MASK")
    if env is not None:
        try:
            mask = int(env, 0)
        except ValueError:
            mask = None
        if mask == 0:
            return "not pinned (disabled)"
    if mask is None:
        # Capture and DSP both carry hard deadlines -- one has to re-enter
        # `sync_rx` every 2 ms, the other has to drain the ring before it laps --
        # so both need performance cores.  They get *different* ones: pinning
        # both to the same pair made them compete for exactly the cores they
        # were pinned to, which is worse than not pinning at all on a busy run.
        # Feature extraction and the GUI have no deadline and are left free.
        n = os.cpu_count() or 4
        if n >= 8:
            # Intel hybrid parts enumerate the performance cores first, two
            # logical CPUs each: capture takes the first physical core, the DSP
            # the second.
            if stage == "capture":
                mask = 0x3
            elif stage == "dsp":
                mask = 0xC
            else:
                return "not pinned"
        else:
            if stage not in ("capture", "dsp"):
                return "not pinned"
            mask = (1 << min(4, n)) - 1
    try:
        if sys.platform == "win32":
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k32.SetProcessAffinityMask.restype = ctypes.c_int
            if k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
                return f"pinned to mask 0x{mask:X}"
            return "not pinned (SetProcessAffinityMask failed)"
        os.sched_setaffinity(0, {i for i in range(64) if mask >> i & 1})
        return f"pinned to mask 0x{mask:X}"
    except Exception as exc:
        return f"not pinned ({exc})"


def raise_process_priority() -> str:
    """Ask the OS to treat this process as latency-sensitive."""
    try:
        if sys.platform == "win32":
            import ctypes

            ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            k32 = ctypes.windll.kernel32
            # GetCurrentProcess returns a HANDLE.  Without an explicit restype
            # ctypes truncates it to a 32-bit int and SetPriorityClass then
            # fails on an invalid handle -- silently, since nothing checks.
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass.restype = ctypes.c_int
            if k32.SetPriorityClass(k32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS):
                return "above-normal"
            return "default (SetPriorityClass failed)"
        os.nice(-5)
        return "nice -5"
    except Exception as exc:
        return f"default ({exc})"


def raise_thread_priority() -> str:
    """Give the calling thread priority over others sharing its process."""
    try:
        if sys.platform == "win32":
            import ctypes

            THREAD_PRIORITY_HIGHEST = 2
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            k32.SetThreadPriority.restype = ctypes.c_int
            if k32.SetThreadPriority(k32.GetCurrentThread(), THREAD_PRIORITY_HIGHEST):
                return "highest"
    except Exception:
        pass
    return "default"


# --------------------------------------------------------------------------
# capture process
# --------------------------------------------------------------------------

def _capture_entry(cfg, spec_dict, cursor, ctrl_q, stat_q):
    """Stage 1, alone in its own process: fill the ring and nothing else.

    Nothing here allocates, converts or inspects samples beyond a peak taken on
    the raw integers.  `bladerf_sync_rx` writes straight into the shared-memory
    slot, so a block is never copied at all on this side.

    It is a process rather than a thread for a measured reason: sharing an
    interpreter with the DSP loop cost 20-50% of the sample rate on an i7-1255U,
    because the capture thread could not reliably re-enter `sync_rx` inside the
    2 ms a 16k block allows.
    """
    import ctypes as _c
    from ctypes import byref, c_uint

    from . import libbladerf as B
    from .shmring import RingSpec, SharedRing

    prio = raise_process_priority()
    pin = pin_to_performance_cores(stage="capture")
    try:
        sys.setswitchinterval(float(os.environ.get("SNIFFER_SWITCH_INTERVAL", "0.0002")))
    except Exception:
        pass

    def log(msg):
        try:
            stat_q.put_nowait({"log": f"[capture] {msg}"})
        except Exception:
            pass

    ring = SharedRing(0, 0, create=False, spec=RingSpec(**spec_dict))
    ring.cursor = cursor
    radio = None
    stats = {"blocks": 0, "overruns": 0, "errors": 0, "last_error": ""}
    try:
        # A bladeRF that was not closed cleanly -- because a previous run was
        # terminated rather than asked to stop -- refuses the next open with
        # NIOS II I/O errors for a moment.  One retry after a short pause
        # clears it, and turns "restart does nothing" into "restart works".
        radio = None
        for attempt in (1, 2, 3):
            try:
                radio = BladeRF(cfg, log=log)
                radio.start()
                break
            except Exception as exc:
                if radio is not None:
                    try:
                        radio.close()
                    except Exception:
                        pass
                    radio = None
                if attempt == 3:
                    raise
                log(f"open attempt {attempt} failed ({exc}); retrying")
                time.sleep(0.6 * attempt)
        log(f"priority {prio}, affinity {pin}")
        try:
            stat_q.put_nowait({
                "capture_ready": True,
                "calibrated": radio.clock.calibrated,
                "clock_detail": radio.clock.detail,
                "info": radio.info,
            })
        except Exception:
            pass

        L = B.lib()
        n = cfg.block_size
        nch = len(cfg.rx_channels)
        timeout = c_uint(cfg.stream_timeout_ms)
        meta = B.BladerfMetadata()
        use_meta = radio.timestamps_available
        meta_ptr = byref(meta) if use_meta else None
        if not use_meta:
            log(
                "dual-antenna stream: FPGA timestamps unavailable (the metadata "
                "format is broken for MIMO in this libbladeRF), so sample loss "
                "is reported from USB overruns only"
            )
        sample_clock = 0
        idx = 0
        epoch = 0
        temp = float("nan")
        last_temp = 0.0
        last_stat = 0.0

        while True:
            # Commands are polled between blocks, not inside the deadline.
            if time.monotonic() - last_stat >= 0.2:
                last_stat = time.monotonic()
                try:
                    while True:
                        cmd = ctrl_q.get_nowait()
                        if cmd is None or cmd.kind == "stop":
                            raise StopIteration
                        if cmd.kind == "retune":
                            plan = ChannelPlan.from_args(**cmd.payload)
                            radio.retune(plan)
                            epoch += 1
                        elif cmd.kind == "gain":
                            radio.set_gain(int(cmd.payload["gain_db"]))
                except queue.Empty:
                    pass
                except StopIteration:
                    break
                stats["temperature"] = temp
                try:
                    stat_q.put_nowait({"capture": dict(stats)})
                except Exception:
                    pass

            slot = ring.slot_view(idx)
            meta.flags = B.BLADERF_META_FLAG_RX_NOW
            meta.status = 0
            meta.actual_count = 0
            # In a multi-channel layout `num_samples` is the TOTAL number of
            # interleaved samples across channels, not the count per channel:
            # "2048 samples for two channels will generate 4096 total samples".
            # Asking for the per-channel figure leaves the driver waiting for a
            # buffer that never fills, which surfaces as a sync_rx timeout.
            rc = L.bladerf_sync_rx(
                radio._dev,
                slot.ctypes.data_as(_c.c_void_p),
                c_uint(n * nch),
                meta_ptr,
                timeout,
            )
            if rc < 0:
                stats["errors"] += 1
                stats["last_error"] = B.strerror(rc)
                time.sleep(0.005)
                continue
            if meta.status & B.BLADERF_META_STATUS_OVERRUN:
                stats["overruns"] += 1

            # Without the metadata format there is no actual_count to read; a
            # successful sync_rx has filled the whole request.
            count = min(int(meta.actual_count) or n, n) if use_meta else n
            now = time.monotonic()
            if now - last_temp >= 1.0:
                temp = radio.rfic_temperature(max_age=0.0)
                last_temp = now

            # Clipping detection does not need every sample, and np.abs on the
            # whole block allocates a 64 KB temporary inside the one loop that
            # has a hard deadline.  A strided view over one sample in eight
            # still catches sustained clipping, which is what the indicator is
            # for; a single isolated clipped sample is not actionable.
            view = slot[: count * 2 * nch : 8]
            peak = max(int(view.max()), -int(view.min())) / SC16_FULL_SCALE

            if use_meta:
                stamp = int(meta.timestamp)
            else:
                # A running count, so downstream arithmetic still works; the
                # `timestamp_valid` flag tells the reader not to mistake it for
                # evidence that nothing was dropped.
                stamp = sample_clock
            sample_clock += count

            ring.publish(
                idx,
                timestamp=stamp,
                timestamp_valid=1 if use_meta else 0,
                wall_time=time.time(),
                n_samples=count,
                epoch=epoch,
                gain_db=radio.cfg.gain_db,
                channel=radio.cfg.plan.channel,
                temperature_c=temp,
                peak=peak,
                frequency_hz=radio.cfg.plan.frequency_hz,
                calibrated=int(radio.clock.calibrated),
            )
            idx += 1
            stats["blocks"] = idx
    except Exception as exc:
        log(f"capture failed: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
    finally:
        if radio is not None:
            try:
                radio.close()
            except Exception:
                pass
        try:
            stat_q.put_nowait({"capture_stopped": True, "capture": dict(stats)})
        except Exception:
            pass


# --------------------------------------------------------------------------
# feature / analysis process
# --------------------------------------------------------------------------

class FeatureWorker:
    """Feature extraction, live analysis and record assembly, in its own process.

    Splitting this off the DSP loop is not premature optimisation.  Measured on
    an i7-1255U, feature extraction cost about 0.7 ms per 2.048 ms block at
    typical advertising rates -- a third of the entire real-time budget -- and
    keeping it in the capture process pushed the sustained sample rate down to
    roughly 4 MSPS because the capture thread could no longer re-enter
    `bladerf_sync_rx` in time.  Feature work is per-packet and order-preserving
    within a single consumer, so it moves cleanly to a second process.
    """

    def __init__(self, in_q, out_q, stat_q, enroll=(), rssi_cal_db=None,
                 antenna_cal_rad: float = 0.0):
        self.in_q = in_q
        self.out_q = out_q
        self.stat_q = stat_q
        self.analyzer = LiveAnalyzer(enroll_addresses=enroll)
        self.rssi_cal_db = rssi_cal_db
        # Per-channel phase offset between the two RX chains.  They share an LO
        # but not a signal path, and their fixed offset is tens of degrees, so
        # an AoA reported without removing it is meaningless.
        self.antenna_cal_rad = antenna_cal_rad
        self.dropped = 0
        self.processed = 0

    def run(self) -> None:
        raise_process_priority()
        pin_to_performance_cores(stage="features")
        batch_out: list = []
        last_flush = time.monotonic()
        while True:
            try:
                batch = self.in_q.get(timeout=0.2)
            except queue.Empty:
                if batch_out:
                    self._send(batch_out)
                    batch_out = []
                continue
            except (EOFError, OSError):
                break
            if batch is None:
                break

            for msg in batch:
                rec = self._to_record(msg)
                if rec is None:
                    continue
                batch_out.append(rec)
                for alert in rec.alerts:
                    batch_out.append(
                        make_event(rec.number, rec.timestamp_us, rec.channel,
                                   "expert", alert, rec.epoch)
                    )

            now = time.monotonic()
            if batch_out and (len(batch_out) >= 32 or now - last_flush > 0.05):
                self._send(batch_out)
                batch_out = []
                last_flush = now

    def _to_record(self, msg):
        if isinstance(msg, PacketRecord):
            return msg  # already-formed event row passed straight through
        try:
            feats = extract_features(
                msg.iq,
                msg.sample_rate,
                sync_offset=msg.sync_in_slice,
                sym_offset=msg.sym_offset,
                bits=msg.bits,
                sym_freq=msg.sym_freq,
                gain_db=msg.gain_db,
                calibrated=msg.calibrated,
                carrier_hz=msg.frequency_hz,
                rssi_cal_db=self.rssi_cal_db,
                full=msg.full_features,
                iq_second=getattr(msg, "iq_second", None),
                antenna_cal_rad=self.antenna_cal_rad,
            )
        except Exception:
            return None

        rec = PacketRecord(
            number=msg.number,
            timestamp_us=msg.timestamp_us,
            wall_time=msg.wall_time,
            sample_index=msg.abs_sync,
            epoch=msg.epoch,
            channel=msg.channel,
            frequency_hz=msg.frequency_hz,
            access_address=msg.access_address,
            pdu_type=msg.pdu_type,
            pdu_name=msg.pdu_name,
            adva=msg.adva,
            adva_bytes=msg.adva_bytes,
            adva_kind=msg.adva_kind,
            tx_add_random=msg.tx_add_random,
            rx_add_random=msg.rx_add_random,
            length=msg.length,
            payload=msg.payload,
            raw_bytes=msg.raw_bytes,
            ad_structures=msg.ad_structures,
            info=msg.info,
            crc_received=msg.crc_received,
            crc_computed=msg.crc_computed,
            crc_ok=msg.crc_ok,
            rssi_dbfs=feats.value("rssi_dbfs"),
            rssi_dbm=feats.value("rssi_dbm"),
            snr_db=feats.value("snr_db"),
            noise_floor_dbfs=msg.noise_floor_dbfs,
            gain_db=msg.gain_db,
            temperature_c=msg.temperature_c,
            calibrated=msg.calibrated,
            sync_score=msg.sync_score,
            corr_peak=msg.corr_peak,
            features=feats,
            iq=msg.iq if msg.keep_iq else None,
            iq_sample_offset=msg.abs_slice_start,
            sync_offset_in_slice=msg.sync_in_slice,
            sym_offset=msg.sym_offset,
            n_symbols=msg.n_symbols,
            n_antennas=2 if getattr(msg, "iq_second", None) is not None else 1,
        )
        self.analyzer.observe(rec)
        self.processed += 1
        return rec

    def _send(self, batch) -> None:
        try:
            self.out_q.put_nowait(batch)
        except queue.Full:
            try:
                self.out_q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.out_q.put_nowait(batch)
            except queue.Full:
                self.dropped += len(batch)
        try:
            self.stat_q.put_nowait({"feature_dropped": self.dropped,
                                    "feature_processed": self.processed})
        except Exception:
            pass


def _feature_entry(in_q, out_q, stat_q, enroll, rssi_cal_db, antenna_cal_rad=0.0):
    try:
        sys.setswitchinterval(float(os.environ.get("SNIFFER_SWITCH_INTERVAL", "0.0002")))
    except Exception:
        pass
    FeatureWorker(in_q, out_q, stat_q, enroll, rssi_cal_db, antenna_cal_rad).run()


def _worker_entry(cfg, spec_dict, cursor, out_q, cmd_q, stat_q, cap_ctrl_q,
                  enroll, keep_iq, rssi_cal_db):
    """Stage-2 process entry point.  Importable at module level for spawn."""
    try:
        # The capture thread must be handed the GIL promptly: at 8 MSPS with 16k
        # blocks it needs to re-enter bladerf_sync_rx every 2 ms, and CPython's
        # default 5 ms switch interval (or anything close to the block period)
        # lets the DSP loop hold the interpreter through a whole block time.
        # Measured on this hardware, 5 ms and 2 ms both starve capture down to
        # roughly 3.3 MSPS with no USB overrun reported -- the samples are simply
        # never asked for.
        sys.setswitchinterval(float(os.environ.get("SNIFFER_SWITCH_INTERVAL", "0.0002")))
    except Exception:
        pass
    ring = SharedRing(0, 0, create=False, spec=RingSpec(**spec_dict))
    ring.cursor = cursor
    SnifferWorker(
        cfg, ring, out_q, cmd_q, stat_q, cap_ctrl_q, enroll, keep_iq, rssi_cal_db
    ).run()


class SnifferPipeline:
    """GUI-side handle on the three worker processes.

        capture process  --(shared-memory ring)-->  DSP process
        DSP process      --(bounded queue)------->  feature process
        feature process  --(bounded queue)------->  GUI (this process)

    Each boundary is placed where it was measured to be needed, not where it
    looked tidy.  Capture is separated from DSP because sharing an interpreter
    lock with the DSP loop cost 20-50% of the sample rate on an i7-1255U;
    features are separated from DSP because they cost about a third of the
    per-block budget and would otherwise push the DSP loop past real time.
    """

    def __init__(
        self,
        cfg: RadioConfig,
        enroll: tuple = (),
        gui_queue_depth: int = 256,
        ring_seconds: float = 2.0,
        keep_iq: bool = True,
        rssi_cal_db=None,
        antenna_cal_rad: float = 0.0,
        in_process: bool = False,
    ) -> None:
        self.cfg = cfg.normalized()
        self.enroll = enroll
        self.ring_seconds = ring_seconds
        self.keep_iq = keep_iq
        self.rssi_cal_db = rssi_cal_db
        self.antenna_cal_rad = antenna_cal_rad

        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self.out_q = ctx.Queue(maxsize=gui_queue_depth)
        self.feat_q = ctx.Queue(maxsize=gui_queue_depth)
        self.cmd_q = ctx.Queue(maxsize=64)
        self.cap_ctrl_q = ctx.Queue(maxsize=64)
        self.stat_q = ctx.Queue(maxsize=512)

        capacity = max(
            int(self.ring_seconds * self.cfg.sample_rate / self.cfg.block_size), 8
        )
        self.ring = SharedRing(
            capacity,
            self.cfg.block_size,
            len(self.cfg.rx_channels),
            self.cfg.sample_rate,
            create=True,
        )

        self._cap_proc = None
        self._dsp_proc = None
        self._feat_proc = None
        self.stats = PipelineStats(
            channel=self.cfg.plan.channel,
            frequency_hz=self.cfg.plan.frequency_hz,
            gain_db=self.cfg.gain_db,
            sample_rate=self.cfg.sample_rate,
            ring_seconds=capacity * self.cfg.block_size / self.cfg.sample_rate,
        )
        self.log_lines = []
        self.device_info = {}
        self.feature_dropped = 0
        self.feature_processed = 0
        self.capture_stats = {}
        self.ready = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        spec = asdict(self.ring.spec)
        # Stage 3 first, then stage 2, then stage 1: every consumer is already
        # draining before its producer is allowed to start.
        self._feat_proc = self._ctx.Process(
            target=_feature_entry,
            args=(self.feat_q, self.out_q, self.stat_q, self.enroll,
                  self.rssi_cal_db, self.antenna_cal_rad),
            daemon=True, name="sniffer-features",
        )
        self._feat_proc.start()

        self._dsp_proc = self._ctx.Process(
            target=_worker_entry,
            args=(self.cfg, spec, self.ring.cursor, self.feat_q, self.cmd_q,
                  self.stat_q, self.cap_ctrl_q, self.enroll, self.keep_iq,
                  self.rssi_cal_db),
            daemon=True, name="sniffer-dsp",
        )
        self._dsp_proc.start()

        self._cap_proc = self._ctx.Process(
            target=_capture_entry,
            args=(self.cfg, spec, self.ring.cursor, self.cap_ctrl_q, self.stat_q),
            daemon=True, name="sniffer-capture",
        )
        self._cap_proc.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal every stage, then join them against one shared deadline.

        Joining each process against its own full timeout meant a shutdown
        could take three times as long as intended, and the caller is the GUI
        thread -- measured at 1 to 4 seconds of a frozen window per stop.  One
        deadline for the whole teardown bounds it, and the stages are told to
        stop before any of them is waited on so they shut down in parallel.
        """
        for q, msg in (
            (self.cap_ctrl_q, Command("stop")),
            (self.cmd_q, Command("stop")),
        ):
            try:
                q.put_nowait(msg)
            except Exception:
                pass
        try:
            self.feat_q.put_nowait(None)
        except Exception:
            pass

        deadline = time.monotonic() + timeout
        for proc in (self._cap_proc, self._dsp_proc, self._feat_proc):
            self._join(proc, max(deadline - time.monotonic(), 0.05))
        try:
            self.ring.close()
        except Exception:
            pass

    @staticmethod
    def _join(proc, timeout: float) -> None:
        """Join a child, tolerating one that was never started.

        `start()` can fail partway -- the device is busy, or the caller never
        started the pipeline at all -- and `Process.join()` asserts rather than
        returning in that case.  Shutdown has to be safe from any state, or a
        failed start leaves an exception on the way out that hides the real one.
        """
        if proc is None:
            return
        try:
            if proc.pid is None:
                return
            proc.join(timeout=timeout)
            if proc.is_alive():
                proc.terminate()
        except (AssertionError, ValueError, OSError):
            pass

    @property
    def alive(self) -> bool:
        def running(p):
            try:
                return p is not None and p.pid is not None and p.is_alive()
            except (AssertionError, ValueError):
                return False

        return running(self._dsp_proc) and running(self._cap_proc)

    # ---- commands ----------------------------------------------------
    def send(self, kind: str, **payload) -> None:
        try:
            self.cmd_q.put_nowait(Command(kind, payload))
        except Exception:
            pass

    def retune(self, channel=None, freq_hz=None, access_address=None, crc_init=None):
        self.send("retune", channel=channel, freq_hz=freq_hz,
                  access_address=access_address, crc_init=crc_init)

    def set_gain(self, gain_db: int) -> None:
        self.send("gain", gain_db=int(gain_db))

    def clear(self) -> None:
        self.send("clear")

    # ---- raw IQ ------------------------------------------------------
    def read_iq(self, abs_start: int, count: int):
        """Samples straight from the shared ring, for the SigMF dump.

        The GUI process reads this directly: the ring is shared memory, so no
        copy crosses a queue and a packet's samples stay available for as long
        as the ring holds them.
        """
        try:
            return self.ring.read_samples(int(abs_start), int(count))
        except Exception:
            return None

    # ---- draining ----------------------------------------------------
    def drain(self, max_batches: int = 64) -> list:
        out = []
        for _ in range(max_batches):
            try:
                out.extend(self.out_q.get_nowait())
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
        return out

    def poll_stats(self):
        while True:
            try:
                d = self.stat_q.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
            if "log" in d:
                self.log_lines.append(d["log"])
                if len(self.log_lines) > 500:
                    del self.log_lines[:-500]
                continue
            if "capture_ready" in d:
                self.ready = True
                self.device_info = d.get("info", {})
                self.stats.calibrated = bool(d.get("calibrated"))
                self.stats.clock_detail = str(d.get("clock_detail", ""))
                continue
            if "capture" in d:
                self.capture_stats = d["capture"]
                self.stats.usb_overruns = int(d["capture"].get("overruns", 0))
                self.stats.radio_errors = int(d["capture"].get("errors", 0))
                continue
            if "capture_stopped" in d:
                self.stats.running = False
                continue
            if "feature_dropped" in d:
                self.feature_dropped = d["feature_dropped"]
                self.feature_processed = d["feature_processed"]
                continue
            keep_cal = self.stats.calibrated
            keep_detail = self.stats.clock_detail
            ovr, errs = self.stats.usb_overruns, self.stats.radio_errors
            self.stats = PipelineStats(**d)
            self.stats.calibrated = self.stats.calibrated or keep_cal
            self.stats.clock_detail = self.stats.clock_detail or keep_detail
            self.stats.usb_overruns = max(ovr, self.stats.usb_overruns)
            self.stats.radio_errors = max(errs, self.stats.radio_errors)
        self.stats.gui_drops = self.feature_dropped
        return self.stats
