"""The Wireshark-layout main window.

Three stacked panes (packet list / detail tree / hex dump) plus a tabbed plot
dock, a control bar and a status bar.

The GUI never sees a packet one at a time.  A single timer drains the pipeline
queue in batches at a fixed rate and repaints once, because a busy channel
produces hundreds of packets a second and a repaint per packet is the fastest
way to make the window stop responding while the capture itself is fine.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import deque
import threading
import time

import numpy as np
from PyQt6.QtCore import QModelIndex, QSettings, QTimer, Qt
from PyQt6.QtGui import QAction, QColor, QFont, QPalette, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableView,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .. import export as X
from ..calibration import (
    CalibrationHistory,
    calibrate_device,
    calibrate_from_samples,
)
from ..channels import ChannelPlan, channel_to_freq
from ..libbladerf import BladeRFError, list_devices
from ..pipeline import SnifferPipeline
from ..radio import RadioConfig
from .dialogs import CalibrationDialog, device_error
from .filters import FilterError, compile_filter
from .model import (
    COLUMN_SAMPLES,
    COLUMNS,
    SIZE_TO_CONTENTS,
    DetailTreeModel,
    PacketTableModel,
    SelectionAwareDelegate,
)
from .plots import PlotDock

GUI_HZ = 25  # 40 ms; inside the 20-30 Hz the design calls for

ORG, APP = "ble-sniffer", "bladerf-ble-sniffer"

_ARROW_CACHE = None

# The plot dock's own background, reused for the whole window so the packet
# table does not sit on a bright panel next to a dark chart.
BG = QColor(28, 30, 34)
BG_ALT = QColor(34, 37, 42)
FG = QColor(222, 226, 232)

STYLE = """
/* --- buttons ------------------------------------------------------- */
QPushButton, QToolButton {{
    border:1px solid #4a4f57; border-radius:4px; padding:5px 14px;
    font-weight:600; color:{fg}; background:#3a3f47;
}}
QPushButton:hover, QToolButton:hover {{ background:#454b55; }}
QPushButton:disabled, QToolButton:disabled {{ color:#7d838c; background:#333740; }}
QPushButton#start {{ background:#1f7a34; color:#fff; border-color:#2c9a45; }}
QPushButton#start:hover {{ background:#249140; }}
QPushButton#stop  {{ background:#7d2028; color:#fff; border-color:#9c2b34; }}
QPushButton#stop:hover  {{ background:#96262f; }}
QPushButton#clear {{ background:#7d2028; color:#fff; border-color:#9c2b34; }}
QPushButton#clear:hover {{ background:#96262f; }}
QPushButton#calib {{ background:#1d5c96; color:#fff; border-color:#2b76ba; }}
QPushButton#calib:hover {{ background:#236fb4; }}

/* The dropdown arrow sinks into the bottom corner because QToolButton puts its
   menu indicator in a corner sub-control unless it is told otherwise.  This
   applies to every menu button, not just Export -- Save inherited the same
   problem when it was added. */
QToolButton {{ padding-right:26px; }}
QToolButton::menu-indicator {{
    subcontrol-origin:padding; subcontrol-position:center right;
    right:8px; width:10px; height:10px;
    image:url("{down}");
}}

/* --- text entry ---------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background:#1b1d21; color:{fg}; border:1px solid #4a4f57;
    border-radius:3px; padding:3px 6px; selection-background-color:#26609e;
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color:#3f88c5; }}
QLineEdit:read-only {{ background:#232629; color:#c2c7ce; }}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    background:#26282c; color:#7d838c;
}}
QComboBox::drop-down {{ border:0; width:18px; }}
QComboBox QAbstractItemView {{
    background:#1b1d21; color:{fg}; selection-background-color:#26609e;
    border:1px solid #4a4f57;
}}
/* Qt draws no arrow of its own once the spin box is styled, and the default
   was invisible against the dark button.  These are CSS triangles: a zero-size
   box whose borders form the arrowhead. */
QSpinBox::up-button, QSpinBox::down-button {{
    background:#4a515c; width:16px; border-left:1px solid #5b626c;
}}
QSpinBox::up-button {{ subcontrol-origin:border; subcontrol-position:top right; }}
QSpinBox::down-button {{ subcontrol-origin:border; subcontrol-position:bottom right; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background:#5c6572; }}
QSpinBox::up-arrow   {{ image:url("{up}");   width:10px; height:10px; }}
QSpinBox::down-arrow {{ image:url("{down}"); width:10px; height:10px; }}
QSpinBox::up-arrow:disabled   {{ image:url("{up_off}"); }}
QSpinBox::down-arrow:disabled {{ image:url("{down_off}"); }}

/* --- menus --------------------------------------------------------- */
/* Popup menus take the desktop palette, not the window's, so on a dark theme
   they came up as dark text on a dark ground. */
QMenu {{
    background:#2b2f36; color:{fg}; border:1px solid #5b626c; padding:4px;
}}
QMenu::item {{ padding:5px 22px 5px 16px; border-radius:3px; }}
QMenu::item:selected {{ background:#1d5c96; color:#ffffff; }}
QMenu::item:disabled {{ color:#7d838c; }}
QMenu::separator {{ height:1px; background:#4a4f57; margin:4px 8px; }}

/* --- checkboxes ---------------------------------------------------- */
/* An unchecked box drew as a dark square on a dark ground and vanished. */
QCheckBox {{ color:{fg}; spacing:6px; }}
QCheckBox::indicator {{
    width:14px; height:14px; border:1px solid #7b828c;
    border-radius:3px; background:#1b1d21;
}}
QCheckBox::indicator:hover {{ border-color:#9aa3ae; }}
QCheckBox::indicator:checked {{ background:#2b76ba; border-color:#4b95d8; }}
QCheckBox::indicator:disabled {{ border-color:#4a4f57; background:#26282c; }}

/* --- scrollbars ---------------------------------------------------- */
QScrollBar:vertical {{ background:#232629; width:13px; margin:0; }}
QScrollBar:horizontal {{ background:#232629; height:13px; margin:0; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background:#5b626c; border-radius:6px; min-height:28px; min-width:28px;
}}
QScrollBar::handle:hover {{ background:#767e8a; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}

/* --- tabs ---------------------------------------------------------- */
/* Unselected tabs sat at almost the ground colour and were unreadable. */
QTabBar::tab {{
    background:#31353d; color:#b9c0c9; border:1px solid #454b55;
    border-bottom:0; padding:5px 14px; margin-right:2px;
    border-top-left-radius:4px; border-top-right-radius:4px;
}}
QTabBar::tab:hover {{ background:#3d434c; color:#e6eaf0; }}
QTabBar::tab:selected {{ background:#1d5c96; color:#ffffff; border-color:#2b76ba; }}
QTabWidget::pane {{ border:1px solid #454b55; top:-1px; }}

/* --- tables -------------------------------------------------------- */
QHeaderView::section {{
    background:#31353d; color:{fg}; border:0;
    border-right:1px solid #454b55; border-bottom:1px solid #454b55;
    padding:4px 6px;
}}
QTableView, QTreeView {{ gridline-color:#3a3f47; }}
QSplitter::handle {{ background:#3a3f47; }}
QSplitter::handle:hover {{ background:#4f5661; }}
QToolTip {{ background:#34383f; color:{fg}; border:1px solid #5b626c; }}
QSlider::groove:vertical {{ background:#2a2d33; width:5px; border-radius:2px; }}
QSlider::handle:vertical {{
    background:#7f8792; height:16px; margin:0 -6px; border-radius:4px;
}}
QSlider::handle:vertical:hover {{ background:#9aa3ae; }}
"""



def _arrow_icons() -> dict:
    """Render up/down triangles to PNGs and return their paths.

    A stylesheet-styled QSpinBox draws no arrow of its own, and the border
    trick that works for a plain widget is ignored for `::up-arrow` --
    Qt reserves the sub-control box and paints nothing into it, so the arrows
    came out as bare rectangles.  Real images are the only reliable route.
    """
    global _ARROW_CACHE
    if _ARROW_CACHE is not None:
        return _ARROW_CACHE

    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QPainter, QPixmap, QPolygonF

    out = {}
    directory = tempfile.mkdtemp(prefix="ble-sniffer-icons-")
    for name, points in (
        ("up", [(1.0, 6.5), (9.0, 6.5), (5.0, 2.0)]),
        ("down", [(1.0, 3.5), (9.0, 3.5), (5.0, 8.0)]),
        ("up_off", [(1.0, 6.5), (9.0, 6.5), (5.0, 2.0)]),
        ("down_off", [(1.0, 3.5), (9.0, 3.5), (5.0, 8.0)]),
    ):
        pm = QPixmap(10, 10)
        pm.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(120, 128, 138) if name.endswith("_off")
                         else QColor(232, 236, 242))
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        painter.end()
        path = os.path.join(directory, f"{name}.png")
        pm.save(path)
        out[name] = path.replace("\\", "/")
    _ARROW_CACHE = out
    return out


class HexView(QPlainTextEdit):
    """Offset / bytes / ASCII dump with two-way highlighting against the tree."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(9)
        self.setFont(f)
        self._data = b""
        self._line_starts: list[int] = []

    def set_data(self, data: bytes) -> None:
        self._data = bytes(data or b"")
        lines = []
        self._line_starts = []
        pos = 0
        for off in range(0, len(self._data), 16):
            chunk = self._data[off : off + 16]
            hexpart = " ".join(f"{b:02x}" for b in chunk)
            hexpart += "   " * (16 - len(chunk))
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            line = f"{off:04x}  {hexpart}  {ascii_part}"
            self._line_starts.append(pos)
            pos += len(line) + 1
            lines.append(line)
        self.setPlainText("\n".join(lines))
        self.highlight(None)

    def highlight(self, byte_range) -> None:
        """Select the characters corresponding to a byte range."""
        from PyQt6.QtWidgets import QTextEdit

        self.setExtraSelections([])
        if not byte_range or not self._data:
            return
        start, length = byte_range
        sels = []
        for i in range(start, min(start + length, len(self._data))):
            row, col = divmod(i, 16)
            if row >= len(self._line_starts):
                break
            base = self._line_starts[row] + 6 + col * 3
            cur = self.textCursor()
            cur.setPosition(base)
            cur.setPosition(base + 2, QTextCursor.MoveMode.KeepAnchor)
            sel = QTextEdit.ExtraSelection()
            sel.cursor = cur
            sel.format.setBackground(QColor(255, 220, 120))
            sel.format.setForeground(QColor(20, 20, 20))
            sels.append(sel)
        self.setExtraSelections(sels)
        if sels:
            # Scroll to the first highlighted byte without leaving a *selection*
            # there: setTextCursor with an anchored cursor paints the first
            # octet in the palette's selection colour, so it came out blue while
            # every other octet was yellow.  A collapsed cursor scrolls just the
            # same and leaves the extra-selection colour alone.
            scroll = QTextCursor(sels[0].cursor)
            scroll.clearSelection()
            self.setTextCursor(scroll)
            self.ensureCursorVisible()


class MainWindow(QMainWindow):
    #: Seconds of raw IQ held in the shared ring.  This is the pipeline's shock
    #: absorber: a transient stall in the DSP stage is only lost if it lasts
    #: longer than this, so a larger ring converts "samples skipped" into
    #: "processed a moment late".  At 8 MSPS it costs 32 MB per second.
    RING_SECONDS = 4.0

    def __init__(self, cfg: RadioConfig, enroll=(), autostart: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.enroll = enroll
        self.pipeline: SnifferPipeline | None = None
        self.t0 = time.time()
        self._autoscroll = True
        self._starting = False
        self._device_warned = False
        # Teardown runs on a worker thread; these track it so Start can refuse
        # to run while the previous capture is still releasing the device.
        self._stopping = None
        self._stop_thread = None
        self._stop_done = True
        # Rolling window of inter-antenna phase differences, used by the array
        # calibration.  Populated on every tick that carries dual-antenna
        # packets, so it has to exist before the timer starts.
        self._antenna_pairs: deque = deque(maxlen=2000)
        # Explicitly file-backed.  QSettings defaults to the registry on
        # Windows, where QSettings.setPath has no effect -- so tests could not
        # isolate themselves and would read and write the real user's layout.
        self.settings = QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, ORG, APP
        )
        self.calibration = CalibrationHistory()

        self.setWindowTitle("BLE single-channel sniffer -- bladeRF 2.0 micro")
        self._apply_palette()

        self._build_controls()
        self._build_panes()
        self._build_dock()
        self._build_statusbar()
        self._restore_layout()
        self._update_buttons()
        self._refresh_device_state()

        esc = QAction(self)
        esc.setShortcut("Esc")
        esc.triggered.connect(self.clear_selection)
        self.addAction(esc)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(int(1000 / GUI_HZ))

        if autostart:
            QTimer.singleShot(80, self.start_capture)

    # ------------------------------------------------------------------
    def _apply_palette(self) -> None:
        """A dark window so the table sits on the same ground as the charts."""
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, BG)
        pal.setColor(QPalette.ColorRole.Base, BG)
        pal.setColor(QPalette.ColorRole.AlternateBase, BG_ALT)
        pal.setColor(QPalette.ColorRole.Button, QColor(48, 52, 59))
        pal.setColor(QPalette.ColorRole.Text, FG)
        pal.setColor(QPalette.ColorRole.WindowText, FG)
        pal.setColor(QPalette.ColorRole.ButtonText, FG)
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(52, 56, 64))
        pal.setColor(QPalette.ColorRole.ToolTipText, FG)
        pal.setColor(QPalette.ColorRole.Highlight, QColor(38, 96, 158))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(pal)
        app = QApplication.instance()
        if app is not None:
            app.setPalette(pal)
        self.setStyleSheet(STYLE.format(fg=FG.name(), **_arrow_icons()))

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 5, 6, 5)
        row.setSpacing(8)

        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("start")
        self.btn_start.clicked.connect(self.start_capture)
        row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stop")
        self.btn_stop.clicked.connect(self.stop_capture)
        row.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setObjectName("clear")
        self.btn_clear.clicked.connect(self.clear_all)
        row.addWidget(self.btn_clear)

        self.btn_calib = QPushButton("Calibrate")
        self.btn_calib.setObjectName("calib")
        self.btn_calib.setToolTip(
            "Measure the receiver's DC offset, IQ imbalance and noise floor.\n"
            "Runs from the live stream while capturing, or opens the device "
            "briefly when stopped.  Every run is kept in the history."
        )
        self.btn_calib.clicked.connect(self.run_calibration)
        row.addWidget(self.btn_calib)

        self.btn_history = QPushButton("History...")
        self.btn_history.setToolTip("Show every recorded calibration run")
        self.btn_history.clicked.connect(self.show_calibration_history)
        row.addWidget(self.btn_history)

        row.addSpacing(12)
        row.addWidget(QLabel("Channel:"))
        self.chan_box = QComboBox()
        for ch in (37, 38, 39):
            self.chan_box.addItem(f"{ch} (adv)", ch)
        for ch in range(37):
            self.chan_box.addItem(f"{ch} (data)", ch)
        self.chan_box.addItem("custom frequency", -1)
        idx = self.chan_box.findData(self.cfg.plan.channel)
        if idx >= 0:
            self.chan_box.setCurrentIndex(idx)
        self.chan_box.activated.connect(self._on_channel_change)
        row.addWidget(self.chan_box)

        self.freq_edit = QLineEdit()
        self.freq_edit.setFixedWidth(76)
        self.freq_edit.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.freq_edit.setToolTip(
            "Centre frequency in MHz.\n"
            "A fixed channel fills this in read-only; choose "
            "'custom frequency' to type one."
        )
        self.freq_edit.setText("{0:.3f}".format(self.cfg.plan.frequency_hz / 1e6))
        self.freq_edit.setReadOnly(self.chan_box.currentData() != -1)
        self.freq_edit.returnPressed.connect(self._on_custom_freq)
        row.addWidget(self.freq_edit)
        row.addWidget(QLabel("MHz"))

        row.addSpacing(12)
        row.addWidget(QLabel("Gain:"))
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(-15, 60)
        self.gain_spin.setValue(int(self.cfg.gain_db))
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setFixedWidth(76)
        self.gain_spin.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.gain_spin.valueChanged.connect(self._on_gain_change)
        row.addWidget(self.gain_spin)
        self.gain_hint = QLabel()
        self.gain_hint.setStyleSheet("color:#9aa2ad;")
        row.addWidget(self.gain_hint)
        self._set_gain_range(-15, 60)

        row.addSpacing(12)
        self.auto_scroll = QCheckBox("Auto-scroll")
        self.auto_scroll.setChecked(True)
        self.auto_scroll.stateChanged.connect(
            lambda s: setattr(self, "_autoscroll", bool(s))
        )
        row.addWidget(self.auto_scroll)

        self.show_plots = QCheckBox("Plots")
        self.show_plots.setChecked(True)
        self.show_plots.setToolTip("Show or hide the plot dock")
        self.show_plots.stateChanged.connect(self._on_show_plots)
        row.addWidget(self.show_plots)

        self.dual_antenna = QCheckBox("Dual antenna (AoA)")
        self.dual_antenna.setChecked(len(self.cfg.rx_channels) > 1)
        self.dual_antenna.setToolTip(
            "Enable RX1 as a second coherent antenna for angle-of-arrival and\n"
            "diversity. Both RX channels share one LO, so this does NOT watch\n"
            "two BLE channels at once. Takes effect on the next Start."
        )
        self.dual_antenna.stateChanged.connect(self._on_dual_antenna)
        row.addWidget(self.dual_antenna)

        row.addStretch(1)

        self.export_btn = QToolButton()
        self.export_btn.setObjectName("export")
        self.export_btn.setText("Export")
        self.export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.export_btn)
        for name, slot in (
            ("Current view to CSV...", self.export_csv),
            ("Current view to Parquet...", self.export_parquet),
            ("Session to PCAP...", self.export_pcap),
            ("Selected packet IQ to SigMF...", self.export_sigmf),
        ):
            act = QAction(name, self)
            act.triggered.connect(slot)
            menu.addAction(act)
        self.export_btn.setMenu(menu)
        row.addWidget(self.export_btn)

        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(bar)
        self._control_bar = holder
        self.setMenuWidget(holder)

    def _set_gain_range(self, lo: float, hi: float) -> None:
        self.gain_spin.setRange(int(lo), int(hi))
        self.gain_hint.setText(f"({int(lo)} to {int(hi)} dB)")
        self.gain_spin.setToolTip(
            f"Manual RX gain, {int(lo)} to {int(hi)} dB.\n"
            "Aim for a peak 10-15 dB below full scale; the status bar warns on "
            "clipping and Calibrate suggests a correction."
        )

    # ------------------------------------------------------------------
    def _build_panes(self) -> None:
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 4)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Filter:")
        bar.addWidget(lbl)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            "adva == AA:BB:CC:DD:EE:FF && crc == fail   |   cfo_ppm > 15   |   "
            "anomaly > 0.8   |   pdu_type == ADV_IND && rssi > -60"
        )
        self.filter_edit.returnPressed.connect(self._apply_filter)
        self.filter_edit.textChanged.connect(self._filter_typing)
        # Stretch so the field ends flush with the table's right edge.
        bar.addWidget(self.filter_edit, 1)
        self.filter_status = QLabel("")
        self.filter_status.setMinimumWidth(0)
        self.filter_status.setMaximumWidth(320)
        bar.addWidget(self.filter_status, 0)
        lay.addLayout(bar)

        self.split = QSplitter(Qt.Orientation.Vertical)

        self.table = QTableView()
        self.model = PacketTableModel()
        self.table.setModel(self.model)
        self.table.setItemDelegate(SelectionAwareDelegate(self.table))
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(19)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        header = self.table.horizontalHeader()
        header.setSectionsMovable(True)
        header.setHighlightSections(False)  # keep every header bold, always
        for i, spec in enumerate(COLUMNS):
            self.table.setColumnWidth(i, spec[1])
        self._size_fixed_columns()
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context)
        self.table.selectionModel().currentRowChanged.connect(self._on_select)
        self.table.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.split.addWidget(self.table)

        self.tree = QTreeView()
        self.detail = DetailTreeModel()
        self.tree.setModel(self.detail)
        self.tree.setItemDelegate(SelectionAwareDelegate(self.tree))
        self.tree.header().setHighlightSections(False)
        self.tree.setColumnWidth(0, 320)
        self.tree.setColumnWidth(1, 300)
        self.tree.setAlternatingRowColors(True)
        self.tree.clicked.connect(self._on_tree_click)
        self.split.addWidget(self.tree)

        self.hex = HexView()
        self.split.addWidget(self.hex)

        self.split.setSizes([560, 300, 170])
        lay.addWidget(self.split, 1)
        self.setCentralWidget(central)

    def _size_fixed_columns(self) -> None:
        """Give the fixed-content columns exactly the width their widest value needs."""
        fm = self.table.fontMetrics()
        header_fm = self.table.horizontalHeader().fontMetrics()
        for col in SIZE_TO_CONTENTS:
            sample = COLUMN_SAMPLES.get(col, "")
            title = COLUMNS[col][0]
            width = max(
                fm.horizontalAdvance(sample), header_fm.horizontalAdvance(title)
            )
            self.table.setColumnWidth(col, width + 22)

    def _fit_detail_column(self) -> None:
        """Size the Field column to its longest label, capped so it stays usable."""
        widest = self.detail.widest_field()
        w = self.tree.fontMetrics().horizontalAdvance(widest) + 56
        self.tree.setColumnWidth(0, int(min(max(w, 180), 620)))

    def _build_dock(self) -> None:
        self.dock = QDockWidget("Plots", self)
        self.dock.setObjectName("plotsDock")
        self.dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.plots = PlotDock(self.cfg.sample_rate)
        self.dock.setWidget(self.plots)
        self.dock.setMinimumWidth(520)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.plots.spectrum.set_channel(self.cfg.plan.frequency_hz)
        # Closing the dock with its X used to be irreversible; keep the checkbox
        # in step with it so it can always be brought back.
        self.dock.visibilityChanged.connect(self._on_dock_visibility)

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_device = QLabel()
        self.lbl_state = QLabel("idle")
        self.lbl_fs = QLabel()
        self.lbl_rate = QLabel()
        self.lbl_crc = QLabel()
        self.lbl_noise = QLabel()
        self.lbl_rssi = QLabel()
        self.lbl_drops = QLabel()
        self.lbl_temp = QLabel()
        self.lbl_ref = QLabel()
        self.lbl_cal = QLabel()
        for w in (
            self.lbl_device, self.lbl_state, self.lbl_fs, self.lbl_rate,
            self.lbl_crc, self.lbl_noise, self.lbl_rssi, self.lbl_drops,
            self.lbl_temp, self.lbl_ref, self.lbl_cal,
        ):
            w.setMinimumWidth(80)
            sb.addWidget(w)
        self._update_fs_label()
        self._update_cal_label()

    def _update_cal_label(self) -> None:
        """Last receiver calibration -- a different thing from reference lock.

        The two were previously conflated under one "UNCALIBRATED" indicator,
        which made it look as though pressing Calibrate had failed.  It had
        not: the receiver's DC offset and quadrature error are measurable and
        correctable in software, while an absolute frequency reference can only
        come from a disciplined clock on the U.FL input.  Running Calibrate
        cannot turn ppm-relative into ppm-absolute, so the two are now reported
        separately and named for what they actually are.
        """
        latest = self.calibration.latest()
        if not latest:
            self.lbl_cal.setText("Rx cal: never")
            self.lbl_cal.setStyleSheet("color:#d99a3a;")
            self.lbl_cal.setToolTip(
                "The receiver's DC offset and IQ imbalance have not been "
                "measured this session. Press Calibrate."
            )
            return
        when = str(latest.get("when", ""))[-8:]
        ok = latest.get("ok", True)
        self.lbl_cal.setText(f"Rx cal: {when}" + ("" if ok else "  [!]"))
        self.lbl_cal.setStyleSheet(
            "color:#5cc07a;" if ok else "color:#e06c6c; font-weight:bold;"
        )
        note = (latest.get("notes") or [""])[0]
        self.lbl_cal.setToolTip(
            "Receiver calibrated at {0} ({1}).\n{2}\n\n"
            "This is separate from the frequency reference shown to the "
            "left.".format(latest.get("when"), latest.get("source"), note)
        )

    def _update_fs_label(self) -> None:
        n = len(self.cfg.rx_channels)
        self.lbl_fs.setText(
            f"fs {self.cfg.sample_rate/1e6:.1f} MSPS"
            + (f" x{n} RX" if n > 1 else "")
        )

    # ------------------------------------------------------------------
    # device presence
    # ------------------------------------------------------------------
    def _devices(self) -> list:
        try:
            return list_devices()
        except Exception:
            return []

    def _refresh_device_state(self) -> bool:
        present = bool(self._devices())
        if present:
            self.lbl_device.setText("bladeRF: connected")
            self.lbl_device.setStyleSheet("color:#5cc07a; font-weight:bold;")
            self.lbl_device.setToolTip("")
        else:
            self.lbl_device.setText("bladeRF: NOT CONNECTED")
            self.lbl_device.setStyleSheet("color:#e06c6c; font-weight:bold;")
            self.lbl_device.setToolTip(
                "No bladeRF was found. Check the USB connection and that no "
                "other program holds the device."
            )
        return present

    # ------------------------------------------------------------------
    # capture control
    # ------------------------------------------------------------------
    def start_capture(self) -> None:
        if self.pipeline is not None and self.pipeline.alive:
            return
        if self._stopping is not None:
            # Starting while the previous capture is still releasing the device
            # gives two processes fighting over one bladeRF, and the second open
            # fails.  Wait for the teardown instead.
            self.statusBar().showMessage(
                "still stopping the previous capture; try again in a moment", 4000
            )
            return
        if not self._refresh_device_state():
            device_error(
                self,
                "No bladeRF found",
                "Cannot start: no bladeRF is connected.",
                "Check the USB cable and that no other program is using the "
                "device, then press Start again.",
            )
            return
        try:
            self.pipeline = SnifferPipeline(
                self.cfg, enroll=self.enroll, ring_seconds=self.RING_SECONDS,
                keep_iq=True, antenna_cal_rad=self.antenna_offset_rad(),
            )
            self.pipeline.start()
            self.t0 = time.time()
            self._starting = True
            self._device_warned = False
            self.lbl_state.setText("starting...")
        except (BladeRFError, OSError, ValueError) as exc:
            device_error(self, "Capture failed to start", str(exc))
            self.pipeline = None
        self._update_buttons()

    def stop_capture(self) -> None:
        """Tear the pipeline down without blocking the GUI thread.

        `SnifferPipeline.stop()` waits on three processes and takes one to four
        seconds; called straight from a button handler that is a frozen window,
        which is exactly what it looked like.  The teardown runs on a worker
        thread instead, Start stays disabled until it finishes, and a timer
        picks up the result.
        """
        pipe, self.pipeline = self.pipeline, None
        self._starting = False
        if pipe is None:
            self.lbl_state.setText("stopped")
            self._update_buttons()
            return

        self._stopping = pipe
        self.lbl_state.setText("stopping...")
        self._update_buttons()

        def teardown():
            try:
                pipe.stop()
            except Exception:
                pass
            self._stop_done = True

        self._stop_done = False
        self._stop_thread = threading.Thread(
            target=teardown, name="sniffer-teardown", daemon=True
        )
        self._stop_thread.start()

    def _poll_teardown(self) -> None:
        """Called on the GUI timer while a stop is in flight."""
        if self._stopping is None:
            return
        if self._stop_done and not (
            self._stop_thread and self._stop_thread.is_alive()
        ):
            self._stopping = None
            self._stop_thread = None
            self.lbl_state.setText("stopped")
            self._refresh_device_state()
            self._update_buttons()

    def clear_all(self) -> None:
        self.model.clear()
        self.plots.clear()
        self.plots.set_focus(None)
        self.detail.set_record(None)
        self.hex.set_data(b"")
        if self.pipeline is not None:
            self.pipeline.clear()
        self._update_buttons()

    def _update_buttons(self) -> None:
        stopping = self._stopping is not None
        running = self.pipeline is not None and (
            self.pipeline.alive or self._starting
        )
        self.btn_start.setEnabled(not stopping)
        # Start and Stop swap rather than sitting side by side greyed out, so
        # the current state is legible at a glance.
        self.btn_start.setVisible(not running)
        self.btn_stop.setVisible(running)
        self.btn_clear.setEnabled(self.model.total > 0 or running)
        # Calibrate is always available: it works from the live ring while
        # capturing and opens the device briefly when stopped.
        self.btn_calib.setVisible(True)
        self.btn_calib.setEnabled(True)

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------
    def run_calibration(self) -> None:
        if not self._refresh_device_state() and self.pipeline is None:
            device_error(
                self, "No bladeRF found",
                "Cannot calibrate: no bladeRF is connected.",
            )
            return
        self.btn_calib.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self._calibrate()
        except (BladeRFError, OSError, ValueError) as exc:
            QApplication.restoreOverrideCursor()
            self.btn_calib.setEnabled(True)
            device_error(self, "Calibration failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.btn_calib.setEnabled(True)
        if result is None:
            device_error(
                self, "Calibration failed",
                "Not enough samples were available to calibrate.",
                "If a capture is running, wait a moment for the ring to fill.",
            )
            return
        self.calibration.add(result)
        # A measured array offset is worth keeping: without subtracting it the
        # AoA column is the array's own wiring, not a bearing.  It is applied
        # when the pipeline is next started, because the stream layout and the
        # feature stage are fixed once running.
        if result.antenna_packets and math.isfinite(result.antenna_phase_offset_deg):
            self.settings.setValue(
                "antenna/phase_offset_deg", float(result.antenna_phase_offset_deg)
            )
            self.settings.sync()
            self.statusBar().showMessage(
                f"antenna phase offset {result.antenna_phase_offset_deg:+.1f} deg "
                "stored; it applies on the next Start", 9000
            )
        self._update_cal_label()
        self.statusBar().showMessage(
            "calibration: " + (result.notes[0] if result.notes else "done"), 8000
        )
        CalibrationDialog(self.calibration, self).exec()

    def _calibrate(self):
        """From the live ring when capturing, otherwise from a short capture."""
        if self.pipeline is not None and self.pipeline.alive:
            written = self.pipeline.ring.written()
            block = self.pipeline.cfg.block_size
            want = min(int(self.cfg.sample_rate * 0.25), written * block)
            if want < 8192:
                return None
            start = max(written * block - want, 0)
            iq = self.pipeline.read_iq(start, want)
            if iq is None or iq.size < 8192:
                return None
            st = self.pipeline.stats
            return calibrate_from_samples(
                iq, self.cfg.sample_rate, source="live ring",
                channel=st.channel, frequency_hz=st.frequency_hz,
                gain_db=st.gain_db, temperature_c=st.temperature_c,
                calibrated_reference=st.calibrated, clock_detail=st.clock_detail,
                antenna_pairs=list(self._antenna_pairs),
            )
        return calibrate_device(self.cfg)

    def antenna_offset_rad(self) -> float:
        """The stored per-array phase offset, in radians."""
        try:
            return math.radians(
                float(self.settings.value("antenna/phase_offset_deg", 0.0))
            )
        except (TypeError, ValueError):
            return 0.0

    def show_calibration_history(self) -> None:
        CalibrationDialog(self.calibration, self).exec()

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------
    def _on_channel_change(self, _idx: int) -> None:
        ch = self.chan_box.currentData()
        # A fixed channel owns the frequency: the field shows it and is
        # read-only, so it cannot drift out of step with the selector.  Only
        # "custom frequency" hands editing over to the operator.
        custom = ch == -1
        self.freq_edit.setReadOnly(not custom)
        if custom:
            self.freq_edit.setFocus()
            self.freq_edit.selectAll()
            return
        # Show the channel's own frequency rather than leaving the field blank.
        self.freq_edit.setText(f"{channel_to_freq(ch)/1e6:.3f}")
        try:
            plan = ChannelPlan.from_args(
                channel=ch,
                access_address=None if ch in (37, 38, 39) else self.cfg.plan.access_address,
                crc_init=None if ch in (37, 38, 39) else self.cfg.plan.crc_init,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Channel", str(exc))
            return
        self._retune(plan)

    def _on_custom_freq(self) -> None:
        if self.freq_edit.isReadOnly():
            return
        try:
            hz = float(self.freq_edit.text()) * 1e6
        except ValueError:
            QMessageBox.warning(
                self, "Frequency",
                "{0!r} is not a number.".format(self.freq_edit.text()),
            )
            return
        try:
            plan = ChannelPlan.from_args(freq_hz=hz)
        except ValueError as exc:
            QMessageBox.warning(self, "Frequency", str(exc))
            return
        self._retune(plan)

    def _retune(self, plan: ChannelPlan) -> None:
        from dataclasses import replace

        self.cfg = replace(self.cfg, plan=plan)
        self.plots.spectrum.set_channel(plan.frequency_hz)
        self.freq_edit.setText(f"{plan.frequency_hz/1e6:.3f}")
        if self.pipeline is not None:
            self.pipeline.retune(
                channel=plan.channel,
                access_address=plan.access_address,
                crc_init=plan.crc_init,
            )

    def _on_gain_change(self, v: int) -> None:
        from dataclasses import replace

        self.cfg = replace(self.cfg, gain_db=int(v))
        if self.pipeline is not None:
            self.pipeline.set_gain(int(v))

    def _on_dual_antenna(self, _state) -> None:
        from dataclasses import replace

        channels = (0, 1) if self.dual_antenna.isChecked() else (0,)
        if channels == tuple(self.cfg.rx_channels):
            return
        self.cfg = replace(self.cfg, rx_channels=channels)
        self._update_fs_label()
        if self.pipeline is not None and self.pipeline.alive:
            self.statusBar().showMessage(
                "dual-antenna change takes effect on the next Start "
                "(the stream layout is fixed while running)", 8000
            )

    def _on_show_plots(self, _state) -> None:
        self.dock.setVisible(self.show_plots.isChecked())

    def _on_dock_visibility(self, visible: bool) -> None:
        if self.show_plots.isChecked() != visible:
            self.show_plots.blockSignals(True)
            self.show_plots.setChecked(visible)
            self.show_plots.blockSignals(False)

    # ------------------------------------------------------------------
    def _filter_typing(self, text: str) -> None:
        try:
            compile_filter(text)
            self.filter_edit.setStyleSheet("")
            self.filter_status.setText("")
        except FilterError as exc:
            self.filter_edit.setStyleSheet("background-color:#5c2b2b; color:#fff;")
            self.filter_status.setText(str(exc)[:70])

    def _apply_filter(self) -> None:
        text = self.filter_edit.text()
        try:
            pred = compile_filter(text)
        except FilterError as exc:
            self.filter_edit.setStyleSheet("background-color:#5c2b2b; color:#fff;")
            self.filter_status.setText(str(exc)[:70])
            return
        self.filter_edit.setStyleSheet("")
        self.model.set_filter(pred, text)
        self.filter_status.setText(f"{self.model.shown}/{self.model.total}")

    def _on_scroll(self, value: int) -> None:
        sb = self.table.verticalScrollBar()
        at_bottom = value >= sb.maximum() - 2
        if not at_bottom and self._autoscroll:
            self._autoscroll = False
            self.auto_scroll.setChecked(False)
        elif at_bottom and not self._autoscroll and self.auto_scroll.isChecked():
            self._autoscroll = True

    def _on_select(self, current, _prev) -> None:
        rec = self.model.record(current.row())
        self.detail.set_record(rec, None)
        self.tree.expandToDepth(0)
        self._fit_detail_column()
        self.hex.set_data(rec.raw_bytes if rec is not None else b"")
        self.plots.packet.show_packet(rec)
        addr = rec.adva if (rec is not None and not rec.is_event and rec.adva) else None
        self.plots.set_focus(addr)

    def _on_table_context(self, pos) -> None:
        """Right-click offers a way back out of a selection.

        Selecting a row pins the per-device plots to that address and there was
        no way to unpin it; a table with no empty space below the rows cannot be
        deselected by clicking away either.
        """
        self.selection_menu().exec(self.table.viewport().mapToGlobal(pos))

    def selection_menu(self) -> QMenu:
        """Built separately from being shown, so it can be inspected."""
        menu = QMenu(self.table)
        act = QAction("Clear selection", menu)
        act.setEnabled(self.table.currentIndex().isValid())
        act.triggered.connect(self.clear_selection)
        menu.addAction(act)
        return menu

    def clear_selection(self) -> None:
        self.table.clearSelection()
        self.table.setCurrentIndex(QModelIndex())
        self.detail.set_record(None)
        self.hex.set_data(b"")
        self.plots.packet.show_packet(None)
        self.plots.set_focus(None)

    def _on_tree_click(self, index) -> None:
        self.hex.highlight(index.data(Qt.ItemDataRole.UserRole))

    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        self._poll_teardown()
        if self.pipeline is None:
            return

        # A device unplugged mid-capture kills the worker processes; say so once,
        # in a modal, rather than letting the window sit there looking alive.
        if self._starting and self.pipeline.ready:
            self._starting = False
            self._update_buttons()
        if not self.pipeline.alive and not self._starting:
            if not self._device_warned:
                self._device_warned = True
                self._refresh_device_state()
                msg = self.pipeline.stats.message or ""
                self.stop_capture()
                device_error(
                    self,
                    "Capture stopped",
                    "The capture stopped unexpectedly. The bladeRF may have "
                    "been disconnected.",
                    msg or "\n".join(self.pipeline.log_lines[-3:])
                    if self.pipeline else msg,
                )
            return

        stats = self.pipeline.poll_stats()
        self.model.set_units(
            rssi_in_dbm=self.pipeline.rssi_cal_db is not None,
            calibrated=stats.calibrated,
        )

        records = self.pipeline.drain()
        if records:
            for r in records:
                if r.is_event or not r.crc_ok or r.n_antennas < 2:
                    continue
                d = r.feature("antenna_phase_deg")
                if d is not None and math.isfinite(d):
                    self._antenna_pairs.append((float(d), float(r.rssi_dbfs)))
            added = self.model.append(records)
            self.plots.add_records(records)
            if added and self._autoscroll:
                self.table.scrollToBottom()
            if self.model._filter is not None:
                self.filter_status.setText(f"{self.model.shown}/{self.model.total}")
            self._update_buttons()

        if self.dock.isVisible():
            w = self.pipeline.ring.written()
            if w > 1:
                got = self.pipeline.ring.read(w - 2)
                if got is not None:
                    raw, _ = got
                    v = raw.reshape(-1, self.pipeline.ring.n_channels, 2)[:2048, 0, :]
                    iq = (
                        v[:, 0].astype(np.float32) + 1j * v[:, 1].astype(np.float32)
                    ) / 2048.0
                    self.plots.spectrum.update_spectrum(iq.astype(np.complex64))
            self.plots.refresh()
            self.plots.interference.update_stats(stats, time.time() - self.t0)

        self._update_status(stats)

    def _update_status(self, s) -> None:
        self.lbl_state.setText("running" if s.running else "starting...")
        self.lbl_rate.setText(f"{s.packets_per_s:.0f} pkt/s")
        self.lbl_crc.setText(f"CRC {s.crc_rate*100:.0f}%")
        self.lbl_noise.setText(f"noise {s.noise_floor_dbfs:.1f} dBFS")
        if s.clipping:
            self.lbl_rssi.setText("CLIPPING")
            self.lbl_rssi.setStyleSheet("color:#e06c6c; font-weight:bold;")
        else:
            self.lbl_rssi.setStyleSheet("")
            self.lbl_rssi.setText(
                "last RSSI -" if not math.isfinite(s.last_rssi_dbfs)
                else f"last {s.last_rssi_dbfs:.0f} dBFS"
            )
        total = max(s.samples + s.lost_samples + s.skipped_samples, 1)
        lost_pct = 100 * s.lost_samples / total
        skip_pct = 100 * s.skipped_samples / total
        self.lbl_drops.setText(
            f"lost {lost_pct:.3f}% / skipped {skip_pct:.2f}% / ovr {s.usb_overruns}"
        )
        self.lbl_drops.setToolTip(
            "lost: samples the radio never delivered "
            "(gaps in the FPGA timestamp sequence)\n"
            "skipped: samples captured but lapped in the ring, because this "
            "host could not process them in time\n"
            "ovr: USB overruns reported by libbladeRF"
        )
        self.lbl_drops.setStyleSheet(
            "color:#e06c6c;" if (lost_pct > 0.1 or skip_pct > 1.0) else ""
        )
        self.lbl_temp.setText(
            "-" if not math.isfinite(s.temperature_c) else f"{s.temperature_c:.1f} C"
        )
        if s.calibrated:
            self.lbl_ref.setText("Ref: locked")
            self.lbl_ref.setStyleSheet("color:#5cc07a; font-weight:bold;")
            self.lbl_ref.setToolTip(
                s.clock_detail + "\n\nppm-scale features are absolute."
            )
        else:
            self.lbl_ref.setText("Ref: internal")
            self.lbl_ref.setStyleSheet("color:#d99a3a; font-weight:bold;")
            self.lbl_ref.setToolTip(
                (s.clock_detail or "onboard VCTCXO, no external reference")
                + "\n\nppm-scale features (carrier offset, symbol clock) are "
                "comparable within this session but not against a stored "
                "baseline. Only a disciplined 10 MHz reference on the U.FL "
                "clock input changes that -- the Calibrate button measures "
                "receiver impairments, which is a different thing."
            )

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------
    def _records_for_export(self) -> list:
        return self.model.visible_records() or self.model.all_records()

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "capture.csv", "CSV (*.csv)")
        if path:
            n = X.write_csv(path, self._records_for_export())
            self.statusBar().showMessage(f"wrote {n} rows to {path}", 5000)

    def export_parquet(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Parquet", "capture.parquet", "Parquet (*.parquet)"
        )
        if not path:
            return
        try:
            n = X.write_parquet(path, self._records_for_export())
        except RuntimeError as exc:
            QMessageBox.warning(self, "Parquet", str(exc))
            return
        self.statusBar().showMessage(f"wrote {n} rows to {path}", 5000)

    def export_pcap(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PCAP", "capture.pcap", "PCAP (*.pcap)"
        )
        if path:
            n = X.write_pcap(path, self._records_for_export())
            self.statusBar().showMessage(
                f"wrote {n} packets to {path} "
                "(LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR)", 6000
            )

    def export_sigmf(self) -> None:
        rec = self.model.record(self.table.currentIndex().row())
        if rec is None or rec.is_event:
            QMessageBox.information(self, "SigMF", "Select a packet first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export packet IQ", f"packet_{rec.number}", "SigMF (*.sigmf-meta)"
        )
        if not path:
            return
        iq, offset_in = None, rec.sync_offset_in_slice
        if self.pipeline is not None:
            pad = int(200e-6 * self.cfg.sample_rate)
            start = max(rec.iq_sample_offset - pad, 0)
            count = (0 if rec.iq is None else len(rec.iq)) + 2 * pad
            iq = self.pipeline.read_iq(start, count)
            if iq is not None:
                offset_in = rec.sync_offset_in_slice + (rec.iq_sample_offset - start)
        if iq is None:
            iq, offset_in = rec.iq, rec.sync_offset_in_slice
        if iq is None:
            QMessageBox.warning(self, "SigMF", "No samples retained for this packet.")
            return
        ann = [{
            "core:sample_start": int(offset_in),
            "core:sample_count": int(len(iq) - offset_in),
            "core:label": f"{rec.pdu_name} {rec.adva}",
            "core:description": f"packet #{rec.number}, CRC {'OK' if rec.crc_ok else 'FAIL'}",
        }]
        data, meta = X.write_sigmf(
            path, iq, self.cfg.sample_rate, rec.frequency_hz,
            description=f"BLE packet #{rec.number} on channel {rec.channel}",
            channel=rec.channel, gain_db=rec.gain_db, calibrated=rec.calibrated,
            annotations=ann,
        )
        self.statusBar().showMessage(f"wrote {data} and {meta}", 6000)

    # ------------------------------------------------------------------
    # layout persistence
    # ------------------------------------------------------------------
    def _restore_layout(self) -> None:
        s = self.settings
        geo = s.value("window/geometry")
        state = s.value("window/state")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)
        split_state = s.value("panes/splitter_state")
        if split_state is not None:
            self.split.restoreState(split_state)
        for i in range(len(COLUMNS)):
            w = s.value(f"columns/packet/{i}")
            if w:
                try:
                    self.table.setColumnWidth(i, int(w))
                except (TypeError, ValueError):
                    pass
        for i in range(3):
            w = s.value(f"columns/detail/{i}")
            if w:
                try:
                    self.tree.setColumnWidth(i, int(w))
                except (TypeError, ValueError):
                    pass
        vis = s.value("panes/plots_visible")
        if vis is not None:
            show = str(vis).lower() in ("true", "1")
            self.show_plots.setChecked(show)
            self.dock.setVisible(show)

    def _save_layout(self) -> None:
        s = self.settings
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/state", self.saveState())
        s.setValue("panes/splitter_state", self.split.saveState())
        s.setValue("panes/plots_visible", self.dock.isVisible())
        for i in range(len(COLUMNS)):
            s.setValue(f"columns/packet/{i}", self.table.columnWidth(i))
        for i in range(3):
            s.setValue(f"columns/detail/{i}", self.tree.columnWidth(i))
        s.sync()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self._save_layout()
        # On the way out we do want to wait: the device has to be released
        # before the process exits, or the next run finds it wedged.
        pipe, self.pipeline = self.pipeline, None
        if pipe is not None:
            pipe.stop()
        if self._stop_thread is not None:
            self._stop_thread.join(timeout=4.0)
        super().closeEvent(event)


def run_gui(cfg: RadioConfig, enroll=(), autostart: bool = True) -> int:
    app = QApplication.instance() or QApplication([])
    win = MainWindow(cfg, enroll=enroll, autostart=autostart)
    # Full screen on first run; a saved geometry from a previous session wins.
    if win.settings.value("window/geometry") is None:
        win.showMaximized()
    else:
        win.show()
    return app.exec()
