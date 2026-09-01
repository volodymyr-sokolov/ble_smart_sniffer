"""The packet record: what the DSP stage produces and everything else consumes.

Deliberately a plain dataclass of primitives plus a few small arrays, because it
crosses a process boundary on a multiprocessing queue and has to pickle cheaply.
The retained IQ slice is the largest field and is the reason `iq` is optional --
the GUI does not need it for the table, only for the per-packet plots and the
SigMF export.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .features import PacketFeatures


@dataclass
class PacketRecord:
    """One decoded (or attempted) BLE packet with all its metadata."""

    # --- identity / ordering ------------------------------------------
    number: int = 0
    timestamp_us: float = 0.0  # monotonic, microseconds since capture start
    wall_time: float = 0.0  # absolute epoch seconds
    sample_index: int = 0  # absolute sample offset into the session
    epoch: int = 0  # bumped on retune; marks a timeline discontinuity

    # --- channel ------------------------------------------------------
    channel: int = 37
    frequency_hz: float = 2.402e9
    access_address: int = 0x8E89BED6

    # --- link layer ---------------------------------------------------
    pdu_type: int = 0
    pdu_name: str = ""
    adva: str = ""
    adva_bytes: bytes = b""
    adva_kind: str = ""
    tx_add_random: bool = False
    rx_add_random: bool = False
    length: int = 0
    payload: bytes = b""
    raw_bytes: bytes = b""
    ad_structures: list = field(default_factory=list)
    info: str = ""

    # --- CRC ----------------------------------------------------------
    crc_received: int = 0
    crc_computed: int = 0
    crc_ok: bool = False

    # --- radio --------------------------------------------------------
    rssi_dbfs: float = float("nan")
    rssi_dbm: float = float("nan")
    snr_db: float = float("nan")
    noise_floor_dbfs: float = float("nan")
    gain_db: int = 0
    temperature_c: float = float("nan")
    calibrated: bool = False
    sync_score: float = 0.0
    corr_peak: float = 0.0

    # --- features -----------------------------------------------------
    features: PacketFeatures | None = None

    # --- live analysis ------------------------------------------------
    cluster_id: int = -1
    anomaly_score: float = float("nan")
    anomaly_contributions: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)

    # --- raw samples (optional; large) --------------------------------
    iq: np.ndarray | None = None
    iq_sample_offset: int = 0
    sync_offset_in_slice: int = 0
    # Sub-sample symbol phase found by timing recovery.  The eye diagram folds
    # on this; folding on the integer sync index alone smears the overlay.
    sym_offset: float = float("nan")
    n_symbols: int = 0  # preamble + access address + PDU + CRC, in symbols
    n_antennas: int = 1

    # --- expert-info rows ---------------------------------------------
    is_event: bool = False  # True for interference / discontinuity markers
    event_kind: str = ""
    event_text: str = ""

    # ------------------------------------------------------------------
    def feature(self, key: str) -> float:
        if self.features is None:
            return float("nan")
        return self.features.value(key)

    @property
    def cfo_ppm(self) -> float:
        return self.feature("cfo_ppm")

    @property
    def modulation_index(self) -> float:
        return self.feature("modulation_index")

    def short_info(self) -> str:
        if self.is_event:
            return self.event_text
        return self.info or self.pdu_name

    def as_row(self) -> dict:
        """Flat dictionary for CSV / Parquet export."""
        row = {
            "number": self.number,
            "timestamp_us": self.timestamp_us,
            "wall_time": self.wall_time,
            "sample_index": self.sample_index,
            "epoch": self.epoch,
            "channel": self.channel,
            "frequency_hz": self.frequency_hz,
            "access_address": f"0x{self.access_address:08X}",
            "pdu_type": self.pdu_type,
            "pdu_name": self.pdu_name,
            "adva": self.adva,
            "adva_kind": self.adva_kind,
            "tx_add": "random" if self.tx_add_random else "public",
            "length": self.length,
            "crc_ok": self.crc_ok,
            "crc_received": f"0x{self.crc_received:06X}",
            "crc_computed": f"0x{self.crc_computed:06X}",
            "rssi_dbfs": self.rssi_dbfs,
            "rssi_dbm": self.rssi_dbm,
            "snr_db": self.snr_db,
            "noise_floor_dbfs": self.noise_floor_dbfs,
            "gain_db": self.gain_db,
            "temperature_c": self.temperature_c,
            "calibrated": self.calibrated,
            "cluster_id": self.cluster_id,
            "anomaly_score": self.anomaly_score,
            "n_antennas": self.n_antennas,
            "aoa_deg": self.feature("aoa_deg"),
            "antenna_phase_deg": self.feature("antenna_phase_deg"),
            "alerts": ";".join(self.alerts),
            "info": self.info,
            "payload_hex": self.payload.hex(),
        }
        if self.features is not None:
            row.update(self.features.as_row())
        return row


def make_event(
    number: int,
    timestamp_us: float,
    channel: int,
    kind: str,
    text: str,
    epoch: int = 0,
) -> PacketRecord:
    """An inline expert-info row, Wireshark style."""
    return PacketRecord(
        number=number,
        timestamp_us=timestamp_us,
        wall_time=time.time(),
        channel=channel,
        epoch=epoch,
        is_event=True,
        event_kind=kind,
        event_text=text,
        info=text,
        pdu_name=kind,
    )
