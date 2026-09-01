"""Packet-list model and detail/hex views.

The table is a `QAbstractTableModel` over a fixed-size deque, never a
`QTableWidget`.  A widget-based table allocates three QTableWidgetItem objects
per cell; at fourteen columns and a few hundred packets a second that is tens of
thousands of QObjects per second and the GUI stops responding well before the
capture does.  The model here owns plain records and renders on demand, so the
cost is proportional to the visible rows, not to the capture length.
"""

from __future__ import annotations

import math
from collections import deque

from PyQt6.QtCore import QAbstractItemModel, QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPalette
from PyQt6.QtWidgets import QStyle, QStyledItemDelegate

# Row colouring.  Chosen to stay legible on both light and dark palettes.
COLOR_CRC_FAIL = QColor(140, 144, 150)
COLOR_OUT_OF_BASELINE = QColor(92, 68, 16)
COLOR_TWO_CLUSTERS = QColor(104, 32, 34)
COLOR_EVENT = QColor(38, 62, 96)
COLOR_EVENT_TEXT = QColor(150, 195, 255)
COLOR_ENROLLING = QColor(190, 240, 200)

# (title, default width, right-aligned, size-to-contents)
#
# The fixed-content columns are sized to their widest possible value rather than
# to a guess: an absolute time is always 8 characters, a channel at most 2, a
# length at most 3, CRC is "FAIL" or "OK", and an address is always 17.
COLUMNS = [
    ("#", 64, True, False),
    ("Time (us)", 104, True, False),
    ("Abs time", 78, False, True),
    ("Ch", 34, True, True),
    ("PDU type", 132, False, False),
    ("AdvA", 138, False, True),
    ("TxAdd", 132, False, False),
    ("Len", 42, True, True),
    ("RSSI (dBm)", 84, True, False),
    ("CFO (ppm)", 80, True, False),
    ("Mod idx", 70, True, False),
    ("CRC", 46, False, True),
    ("Cluster", 62, True, False),
    ("Anomaly", 72, True, False),
    # Only meaningful with --dual-antenna; blank on a single-antenna capture.
    ("AoA (deg)", 76, True, False),
    ("d-phi (deg)", 84, True, False),
    ("Info", 320, False, False),
]

RIGHT_ALIGNED = {i for i, c in enumerate(COLUMNS) if c[2]}
SIZE_TO_CONTENTS = {i for i, c in enumerate(COLUMNS) if c[3]}

#: Widest value each size-to-contents column can ever hold, used to reserve the
#: right width before any packet has arrived.
COLUMN_SAMPLES = {
    2: "00:00:00",
    3: "39",
    5: "AA:BB:CC:DD:EE:FF",
    7: "255",
    11: "FAIL",
}


def _fmt(v: float, spec: str = "{:.2f}") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return spec.format(v)


class PacketTableModel(QAbstractTableModel):
    """Fixed-capacity packet list with live filtering."""

    def __init__(self, capacity: int = 20000, parent=None) -> None:
        super().__init__(parent)
        self._all: deque = deque(maxlen=capacity)
        self._view: list = []
        self._filter = None
        self._filter_text = ""
        self.capacity = capacity
        self._rssi_in_dbm = False
        self._calibrated = False

    # ---- Qt interface -------------------------------------------------
    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._view)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            if section == 8:
                return "RSSI (dBm)" if self._rssi_in_dbm else "RSSI (dBFS)"
            if section == 9:
                return "CFO (ppm)" if self._calibrated else "CFO (ppm) *"
            return COLUMNS[section][0]
        if role == Qt.ItemDataRole.FontRole:
            # Headers stay bold whatever the selection does; Qt un-bolds the
            # header of the current column by default, which reads as a glitch.
            f = QFont()
            f.setBold(True)
            return f
        if role == Qt.ItemDataRole.ToolTipRole:
            if section in (14, 15):
                return (
                    "Angle of arrival and inter-antenna phase difference.\n"
                    "Only produced with --dual-antenna (RX1 as a second "
                    "coherent antenna), and only meaningful after the fixed "
                    "per-channel phase offset has been calibrated out."
                )
            if section == 8 and not self._rssi_in_dbm:
                return (
                    "No dBm calibration table is loaded (--rssi-cal), so this "
                    "column shows dBFS: receiver-referred, gain dependent."
                )
            if section == 9 and not self._calibrated:
                return (
                    "* UNCALIBRATED: no disciplined reference is locked, so "
                    "these values include receiver drift. They are comparable "
                    "within this session but not against a stored baseline."
                )
            return COLUMNS[section][0]
        return None

    def set_units(self, rssi_in_dbm: bool, calibrated: bool) -> None:
        """Let the headers, not every cell, carry the units and caveats."""
        if (rssi_in_dbm, calibrated) == (self._rssi_in_dbm, self._calibrated):
            return
        self._rssi_in_dbm = rssi_in_dbm
        self._calibrated = calibrated
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 8, 9)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        rec = self._view[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._cell(rec, col)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in RIGHT_ALIGNED:
                return int(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            # A selected row is painted with the highlight colour, and the
            # per-row foregrounds below are chosen against the *unselected*
            # background -- grey on blue is unreadable.  Qt does not tell the
            # model what is selected, so the view supplies a delegate that
            # overrides this for the current row; see SelectionAwareDelegate.
            if rec.is_event:
                return QBrush(COLOR_EVENT_TEXT)
            if not rec.crc_ok:
                return QBrush(COLOR_CRC_FAIL)
            return None
        if role == Qt.ItemDataRole.BackgroundRole:
            return self._background(rec)
        if role == Qt.ItemDataRole.FontRole and rec.is_event:
            f = QFont()
            f.setItalic(True)
            return f
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(rec)
        if role == Qt.ItemDataRole.UserRole:
            return rec
        return None

    # ---- rendering ----------------------------------------------------
    def _cell(self, rec, col: int):
        if rec.is_event:
            if col == 0:
                return str(rec.number)
            if col == 1:
                return _fmt(rec.timestamp_us, "{:.0f}")
            if col == 3:
                return str(rec.channel)
            if col == 4:
                return rec.event_kind
            if col == 16:
                return rec.event_text
            return ""

        if col == 0:
            return str(rec.number)
        if col == 1:
            return _fmt(rec.timestamp_us, "{:.0f}")
        if col == 2:
            import time as _t

            return _t.strftime("%H:%M:%S", _t.localtime(rec.wall_time)) if rec.wall_time else ""
        if col == 3:
            return str(rec.channel)
        if col == 4:
            return rec.pdu_name
        if col == 5:
            return rec.adva
        if col == 6:
            return rec.adva_kind
        if col == 7:
            return str(rec.length)
        if col == 8:
            # dBm where a calibration table exists, dBFS otherwise.  Which one
            # is in the column is stated in the header, not repeated on every
            # row -- see PacketTableModel.set_units.
            if math.isfinite(rec.rssi_dbm):
                return _fmt(rec.rssi_dbm, "{:.1f}")
            return _fmt(rec.rssi_dbfs, "{:.1f}")
        if col == 9:
            return _fmt(rec.cfo_ppm)
        if col == 10:
            return _fmt(rec.modulation_index, "{:.3f}")
        if col == 11:
            return "OK" if rec.crc_ok else "FAIL"
        if col == 12:
            return str(rec.cluster_id) if rec.cluster_id >= 0 else ""
        if col == 13:
            return _fmt(rec.anomaly_score)
        if col == 14:
            return _fmt(rec.feature("aoa_deg"), "{:.1f}")
        if col == 15:
            return _fmt(rec.feature("antenna_phase_deg"), "{:.1f}")
        if col == 16:
            return rec.short_info()
        return ""

    def _background(self, rec):
        if rec.is_event:
            return QBrush(COLOR_EVENT)
        for a in rec.alerts:
            if "two separated feature clusters" in a:
                return QBrush(COLOR_TWO_CLUSTERS)
        if rec.alerts:
            return QBrush(COLOR_OUT_OF_BASELINE)
        if math.isfinite(rec.anomaly_score) and rec.anomaly_score > 3.0:
            return QBrush(COLOR_OUT_OF_BASELINE)
        return None

    def _tooltip(self, rec) -> str:
        if rec.is_event:
            return rec.event_text
        bits = [f"{rec.pdu_name} from {rec.adva or '(no address)'}"]
        if rec.alerts:
            bits.extend("! " + a for a in rec.alerts)
        if not rec.calibrated:
            bits.append("ppm values UNCALIBRATED (no disciplined reference locked)")
        if math.isnan(rec.rssi_dbm):
            bits.append("RSSI shown in dBFS ('F'): no dBm calibration table loaded")
        return "\n".join(bits)

    # ---- mutation -----------------------------------------------------
    def append(self, records: list) -> int:
        """Add a batch.  Returns how many passed the filter and became visible."""
        if not records:
            return 0
        keep = [r for r in records if self._filter is None or self._filter(r)]
        # The deque discards from the left once full, so the visible list has to
        # be rebuilt when that happens; otherwise the two drift apart.
        overflow = max(len(self._all) + len(records) - self.capacity, 0)
        self._all.extend(records)
        if overflow:
            self.beginResetModel()
            self._rebuild()
            self.endResetModel()
            return len(keep)
        if keep:
            start = len(self._view)
            self.beginInsertRows(QModelIndex(), start, start + len(keep) - 1)
            self._view.extend(keep)
            self.endInsertRows()
        return len(keep)

    def clear(self) -> None:
        self.beginResetModel()
        self._all.clear()
        self._view = []
        self.endResetModel()

    def set_filter(self, predicate, text: str = "") -> None:
        self._filter = predicate
        self._filter_text = text
        self.beginResetModel()
        self._rebuild()
        self.endResetModel()

    def _rebuild(self) -> None:
        if self._filter is None:
            self._view = list(self._all)
        else:
            self._view = [r for r in self._all if self._filter(r)]

    def record(self, row: int):
        if 0 <= row < len(self._view):
            return self._view[row]
        return None

    @property
    def total(self) -> int:
        return len(self._all)

    @property
    def shown(self) -> int:
        return len(self._view)

    def all_records(self) -> list:
        return list(self._all)

    def visible_records(self) -> list:
        return list(self._view)


# --------------------------------------------------------------------------
# detail tree
# --------------------------------------------------------------------------

class DetailNode:
    __slots__ = ("name", "value", "extra", "children", "parent", "row", "byte_range")

    def __init__(self, name, value="", extra="", byte_range=None, parent=None):
        self.name = name
        self.value = value
        self.extra = extra
        self.byte_range = byte_range  # (start, length) into raw_bytes
        self.children: list[DetailNode] = []
        self.parent = parent
        self.row = 0

    def add(self, node: "DetailNode") -> "DetailNode":
        node.parent = self
        node.row = len(self.children)
        self.children.append(node)
        return node

    def child(self, name, value="", extra="", byte_range=None) -> "DetailNode":
        return self.add(DetailNode(name, value, extra, byte_range))


class SelectionAwareDelegate(QStyledItemDelegate):
    """Force readable text on the selected row.

    The model paints CRC failures grey and expert rows blue against the normal
    background.  Qt keeps those foregrounds when the row is selected, so grey
    text lands on the selection highlight and becomes unreadable.  The model
    cannot fix this -- it is not told what is selected -- so the delegate
    replaces the foreground for selected rows with the palette's
    highlighted-text colour.
    """

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if option.state & QStyle.StateFlag.State_Selected:
            colour = option.palette.color(
                QPalette.ColorGroup.Active, QPalette.ColorRole.HighlightedText
            )
            option.palette.setColor(QPalette.ColorRole.Text, colour)
            option.palette.setColor(QPalette.ColorRole.WindowText, colour)
            option.palette.setColor(QPalette.ColorRole.HighlightedText, colour)


class DetailTreeModel(QAbstractItemModel):
    """Expandable per-packet detail, Wireshark style.

    Every physical-layer row shows `measured | spec limit | position within the
    enrolled baseline`, because a number on its own does not tell an operator
    whether it is unusual for this device or merely unusual in general.
    """

    HEADERS = ("Field", "Value", "Spec / baseline")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root = DetailNode("root")

    def set_record(self, rec, baseline=None) -> None:
        self.beginResetModel()
        self.root = DetailNode("root")
        if rec is not None:
            self._build(rec, baseline)
        self.endResetModel()

    # ---- Qt interface -------------------------------------------------
    def index(self, row, column, parent=QModelIndex()):
        node = parent.internalPointer() if parent.isValid() else self.root
        if row < 0 or row >= len(node.children):
            return QModelIndex()
        return self.createIndex(row, column, node.children[row])

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        p = node.parent
        if p is None or p is self.root:
            return QModelIndex()
        return self.createIndex(p.row, 0, p)

    def rowCount(self, parent=QModelIndex()):
        node = parent.internalPointer() if parent.isValid() else self.root
        return len(node.children)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        if role == Qt.ItemDataRole.FontRole:
            f = QFont()
            f.setBold(True)
            return f
        return None

    def widest_field(self) -> str:
        """Longest label in the Field column, for sizing it to its contents."""
        widest = self.HEADERS[0]

        def walk(node, depth=0):
            nonlocal widest
            for c in node.children:
                text = "    " * depth + c.name
                if len(text) > len(widest):
                    widest = text
                walk(c, depth + 1)

        walk(self.root)
        return widest

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node = index.internalPointer()
        if role == Qt.ItemDataRole.DisplayRole:
            return (node.name, str(node.value), str(node.extra))[index.column()]
        if role == Qt.ItemDataRole.UserRole:
            return node.byte_range
        if role == Qt.ItemDataRole.ForegroundRole and "UNCALIBRATED" in str(node.value):
            return QBrush(QColor(200, 120, 0))
        return None

    # ---- construction -------------------------------------------------
    def _build(self, rec, baseline) -> None:
        r = self.root

        if rec.is_event:
            n = r.child("Expert info", rec.event_kind)
            n.child("Message", rec.event_text)
            n.child("Time (us)", f"{rec.timestamp_us:.0f}")
            n.child("Channel", rec.channel)
            return

        frame = r.child("Frame", f"#{rec.number}")
        import time as _t

        frame.child("Capture time", _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(rec.wall_time)))
        frame.child("Monotonic time (us)", f"{rec.timestamp_us:.1f}")
        frame.child("Channel", rec.channel)
        frame.child("Frequency (MHz)", f"{rec.frequency_hz/1e6:.3f}")
        frame.child("Access address", f"0x{rec.access_address:08X}")
        frame.child("Sample offset in ring", rec.iq_sample_offset)
        frame.child("Retained samples", 0 if rec.iq is None else len(rec.iq))
        frame.child("Epoch", rec.epoch, "increments on retune")

        radio = r.child("Radio", f"{rec.rssi_dbfs:.1f} dBFS")
        radio.child("RSSI (dBFS)", f"{rec.rssi_dbfs:.2f}")
        radio.child(
            "RSSI (dBm)",
            "n/a" if math.isnan(rec.rssi_dbm) else f"{rec.rssi_dbm:.1f}",
            "" if not math.isnan(rec.rssi_dbm) else "no calibration table loaded",
        )
        radio.child("SNR (dB)", _fmt(rec.snr_db))
        radio.child("Noise floor (dBFS)", _fmt(rec.noise_floor_dbfs))
        radio.child("Receiver gain (dB)", rec.gain_db)
        radio.child("RFIC temperature (C)", _fmt(rec.temperature_c, "{:.1f}"))
        radio.child(
            "Reference",
            "GPSDO locked" if rec.calibrated else "UNCALIBRATED",
            "ppm-scale features valid" if rec.calibrated
            else "ppm-scale features include receiver drift",
        )
        radio.child("Sync correlation", f"{rec.corr_peak:.3f}")
        radio.child("Sync agreement", f"{rec.sync_score:.3f}")

        if rec.features is not None:
            self._phy(r, rec, baseline)

        ll = r.child("Link layer", rec.pdu_name)
        ll.child("PDU type", f"{rec.pdu_name} (0x{rec.pdu_type:02X})", byte_range=(0, 1))
        ll.child("TxAdd", "random" if rec.tx_add_random else "public", byte_range=(0, 1))
        ll.child("RxAdd", "random" if rec.rx_add_random else "public", byte_range=(0, 1))
        ll.child("Length", rec.length, byte_range=(1, 1))
        if rec.adva:
            ll.child("AdvA", rec.adva, rec.adva_kind, byte_range=(2, 6))

        if rec.ad_structures:
            ad = r.child("Advertising data", f"{len(rec.ad_structures)} structures")
            off = 8
            for s in rec.ad_structures:
                ln = len(s["raw"]) + 2
                node = ad.child(
                    s["name"],
                    s["value"] if s["value"] is not None else "",
                    f"type 0x{s['type']:02X}" if s["type"] is not None else "truncated",
                    byte_range=(off, ln),
                )
                node.child("Raw", bytes(s["raw"]).hex(), byte_range=(off + 2, ln - 2))
                off += ln

        crc = r.child("CRC", "OK" if rec.crc_ok else "FAIL")
        crc.child("Received", f"0x{rec.crc_received:06X}", byte_range=(2 + rec.length, 3))
        crc.child("Computed", f"0x{rec.crc_computed:06X}")
        crc.child("Status", "pass" if rec.crc_ok else "fail")

        if rec.alerts:
            al = r.child("Expert info", f"{len(rec.alerts)} alerts")
            for a in rec.alerts:
                al.child("Alert", a)

    def _phy(self, root, rec, baseline) -> None:
        feats = rec.features
        phy = root.child(
            "PHY features",
            "full" if feats.full else "reduced (load shedding)",
        )
        groups = {
            "Carrier and oscillator": [
                "cfo_hz", "cfo_ppm", "cfo_preamble_hz", "cfo_payload_hz",
                "drift_hz", "drift_rate", "lo_leakage_dbc", "image_rejection_db",
                "phase_noise_10kHz", "phase_noise_50kHz",
                "phase_noise_100kHz", "phase_noise_500kHz",
            ],
            "Modulation quality": [
                "modulation_index", "dev_one_hz", "dev_zero_hz", "dev_asymmetry",
                "transition_asymmetry", "effective_bt", "freq_error_rms",
                "freq_error_peak", "symbol_clock_ppm", "symbol_jitter_ps",
                "eye_opening", "residual_isi",
            ],
            "Envelope and transient": [
                "rise_time_us", "overshoot", "fall_time_us", "splatter_db",
            ],
            "Amplitude and multipath": [
                "rssi_dbfs", "rssi_dbm", "snr_db", "delay_spread_us",
                "antenna_phase_deg", "aoa_deg",
            ],
        }
        stats = baseline.stats if baseline is not None else None
        keys = list(baseline.keys) if baseline is not None else []

        for group, names in groups.items():
            present = [n for n in names if n in feats.measurements]
            if not present:
                continue
            g = phy.child(group)
            for name in present:
                m = feats.measurements[name]
                sigma = ""
                if stats is not None and name in keys:
                    i = keys.index(name)
                    sd = stats.std[i]
                    if sd > 0 and math.isfinite(m.value):
                        sigma = f" | {(m.value - stats.mean[i]) / sd:+.1f} sigma"
                spec = m.spec_text()
                if m.in_spec is False:
                    spec += "  OUT OF SPEC"
                g.child(name, m.format(), spec + sigma)

        if rec.anomaly_contributions:
            an = root.child(
                "Anomaly", f"{rec.anomaly_score:.2f} sigma (RMS over features)"
            )
            for k, v in sorted(
                rec.anomaly_contributions.items(), key=lambda kv: -abs(kv[1])
            ):
                an.child(k, f"{v:+.2f} sigma")
