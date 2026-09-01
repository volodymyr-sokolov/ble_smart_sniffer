"""pyqtgraph plot dock: spectrum, waterfall, CFO, feature scatter, RSSI, per-packet.

All plots are fed from batches on the GUI timer, never per packet.  Series are
capped and use `setData` on existing curves rather than re-adding items, because
a plot that reallocates its scene graph at 30 Hz is the second most common way
to make a capture GUI unresponsive (the first being a widget-based table).

Two rules that came out of looking at the plots on real traffic:

* **Only CRC-verified packets become series.** A CRC failure yields a garbage
  address and garbage features.  Feeding those in produced 186 "devices" from
  451 packets, each contributing a couple of stray points.
* **Never join points across a gap.** A series with two samples ten seconds
  apart drew a straight line diagonally across the whole plot, which is what
  made the time-series charts unreadable.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import exporters

from ..channels import channel_to_freq
from ..features import FEATURE_VECTOR_KEYS
from .chartinfo import as_html

pg.setConfigOptions(antialias=False, useOpenGL=False, imageAxisOrder="row-major")

# A palette that stays distinguishable for a dozen advertisers.
SERIES_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
]

# Colours are handed out in order of first appearance and remembered for the
# life of the process.  Hashing the address instead looked simpler but is wrong
# twice over: Python's string hash is randomised per process, so a device
# changes colour between runs, and unrelated addresses collide on one colour.
_COLOR_ASSIGNMENT: dict[str, tuple] = {}


def color_for(key: str) -> tuple:
    c = _COLOR_ASSIGNMENT.get(key)
    if c is None:
        c = SERIES_COLORS[len(_COLOR_ASSIGNMENT) % len(SERIES_COLORS)]
        _COLOR_ASSIGNMENT[key] = c
    return c


def reset_colors() -> None:
    _COLOR_ASSIGNMENT.clear()


def short_addr(addr: str) -> str:
    """Legends need to fit; the last three octets identify a device in practice."""
    return addr[-8:] if len(addr) > 8 else addr


class ChartInfoDialog(QDialog):
    """Modal reference note for one chart."""

    def __init__(self, key: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chart reference")
        self.setModal(True)
        self.resize(720, 620)
        lay = QVBoxLayout(self)
        view = QTextBrowser()
        view.setOpenExternalLinks(False)
        view.setHtml(as_html(key))
        f = QFont()
        f.setPointSize(10)
        view.setFont(f)
        lay.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        lay.addWidget(buttons)


class SaveMenu(QToolButton):
    """Save the chart this button belongs to as PNG, SVG or CSV.

    The three formats answer different questions: PNG for a report, SVG for a
    figure that will be resized or edited, CSV for the numbers behind the
    picture.  A chart that can only be screenshotted forces the operator to
    re-derive data the tool already has.
    """

    def __init__(self, owner: QWidget, name: str, rows_fn) -> None:
        super().__init__(owner)
        self.owner = owner
        self.name = name
        self.rows_fn = rows_fn
        self.setText("Save")
        self.setObjectName("save")
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setMaximumWidth(84)
        self.setToolTip("Save this chart as an image, a vector figure or its data")
        menu = QMenu(self)
        for label, kind in (
            ("Image (PNG)...", "png"),
            ("Vector (SVG)...", "svg"),
            ("Data (CSV)...", "csv"),
        ):
            act = QAction(label, self)
            act.triggered.connect(lambda _c=False, k=kind: self.save(k))
            menu.addAction(act)
        self.setMenu(menu)

    # ------------------------------------------------------------------
    def _plot_item(self):
        """The first PlotItem in the owning tab, for the vector export."""
        for child in self.owner.findChildren(pg.PlotWidget):
            return child.getPlotItem()
        return None

    def save(self, kind: str) -> None:
        filters = {
            "png": "PNG image (*.png)",
            "svg": "SVG figure (*.svg)",
            "csv": "CSV data (*.csv)",
        }[kind]
        path, _ = QFileDialog.getSaveFileName(
            self.owner, f"Save {self.name} as {kind.upper()}",
            f"{self.name}.{kind}", filters,
        )
        if not path:
            return
        if not path.lower().endswith("." + kind):
            path += "." + kind
        try:
            if kind == "png":
                self.owner.grab().save(path)
            elif kind == "svg":
                item = self._plot_item()
                if item is None:
                    raise RuntimeError("this chart has no vector plot to export")
                exporters.SVGExporter(item).export(path)
            else:
                self._write_csv(path)
        except Exception as exc:
            QMessageBox.warning(self.owner, "Save failed", str(exc))
            return
        w = self.owner.window()
        if hasattr(w, "statusBar"):
            w.statusBar().showMessage(f"wrote {path}", 5000)

    def _write_csv(self, path: str) -> None:
        header, rows = self.rows_fn()
        if not rows:
            raise RuntimeError("this chart has no data yet")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)


class InfoBar(QHBoxLayout):
    """A row with an on-chart title, a Save menu and the reference modal."""

    def __init__(self, owner: QWidget, key: str, title: str, rows_fn=None) -> None:
        super().__init__()
        label = QLabel(f"<b>{title}</b>")
        self.addWidget(label)
        self.addStretch(1)
        if rows_fn is not None:
            self.addWidget(SaveMenu(owner, key, rows_fn))
        btn = QPushButton("Info")
        btn.setToolTip("Formulas, how to read this chart, and its limits")
        btn.setMaximumWidth(78)
        btn.clicked.connect(lambda: ChartInfoDialog(key, owner).exec())
        self.addWidget(btn)


# --------------------------------------------------------------------------
# spectrum + waterfall
# --------------------------------------------------------------------------

class SpectrumTab(QWidget):
    """Live spectrum and waterfall with the BLE channel mask overlaid."""

    PEAK_DECAY_DB = 0.35  # per frame
    DEFAULT_RANGE_DB = 45

    def __init__(self, sample_rate: float, parent=None) -> None:
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.centre = 2.402e9
        lay = QVBoxLayout(self)
        lay.addLayout(InfoBar(self, "spectrum", "Spectrum and waterfall",
                              self._csv_rows))

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Power (dBFS)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setYRange(-110, -10)
        self.plot.addLegend(offset=(-10, 10))
        self.curve = self.plot.plot(pen=pg.mkPen((80, 170, 255), width=1), name="live")
        self.peak_curve = self.plot.plot(
            pen=pg.mkPen((255, 140, 60), width=1), name="peak hold"
        )
        lay.addWidget(self.plot, 3)

        # A bare PlotWidget with an ImageItem, not an ImageView.  ImageView adds
        # its own histogram/levels panel down the right-hand side and positions
        # the image through pos/scale arguments that no longer apply cleanly --
        # the result was an empty plot with the data squeezed into the side
        # panel, which is what "rotated 90 degrees on the left" was.
        self.wf_plot = pg.PlotWidget()
        self.wf_plot.setLabel("bottom", "Frequency", units="Hz")
        self.wf_plot.setLabel("left", "Time (s ago)")
        self.wf_plot.invertY(True)
        self.wf_plot.setMouseEnabled(x=True, y=False)
        # Reserve room for the tick text; the default axis width clipped "0"
        # in half at the top of the scale.
        self.wf_plot.getAxis("left").setWidth(58)
        self.wf_image = pg.ImageItem()
        cmap = pg.colormap.get("viridis")
        if cmap is not None:
            self.wf_image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.wf_plot.addItem(self.wf_image)
        # The two plots are stacked and show the same frequency span, so their
        # x axes are linked: panning or zooming either moves both, and a feature
        # in the waterfall sits directly under its peak in the spectrum.
        # Without this they auto-range independently and the same frequency
        # lands at different horizontal positions in the two panes.
        self.wf_plot.setXLink(self.plot)

        # The useful dynamic range of a waterfall depends entirely on the
        # environment: a fixed -110..-20 dBFS window washes out a quiet channel
        # and saturates a busy one.  The top of the scale tracks the data and
        # the slider sets how many dB below it are shown, which is the control
        # an operator actually wants.
        wf_row = QHBoxLayout()
        wf_row.setContentsMargins(0, 0, 0, 0)
        wf_row.addWidget(self.wf_plot, 1)

        side = QVBoxLayout()
        side.setContentsMargins(0, 4, 2, 4)
        cap = QLabel("Contrast")
        cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        side.addWidget(cap)
        self.contrast = QSlider(Qt.Orientation.Vertical)
        self.contrast.setRange(6, 80)          # dB shown below the peak
        self.contrast.setValue(self.DEFAULT_RANGE_DB)
        self.contrast.setInvertedAppearance(True)  # up = more contrast
        self.contrast.setToolTip(
            "Dynamic range of the waterfall, in dB below the current peak.\n"
            "Drag up for more contrast (a narrower window)."
        )
        self.contrast.valueChanged.connect(self._on_contrast)
        side.addWidget(self.contrast, 1)
        self.contrast_label = QLabel()
        self.contrast_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        side.addWidget(self.contrast_label)
        wf_row.addLayout(side)

        wf_holder = QWidget()
        wf_holder.setLayout(wf_row)
        lay.addWidget(wf_holder, 2)

        self.history_rows = 200
        self._rows_filled = 0
        self._wf_top = -20.0
        self._hist: np.ndarray | None = None
        self._peak: np.ndarray | None = None
        self._mask_items: list = []
        self._frame_dt = 1.0 / 25.0

    def _on_contrast(self, _value: int) -> None:
        self._apply_levels()

    def _apply_levels(self) -> None:
        span = float(self.contrast.value())
        self.contrast_label.setText("{0:.0f} dB".format(span))
        self.wf_image.setLevels((self._wf_top - span, self._wf_top))

    def _csv_rows(self):
        f, live = self.curve.getData()
        if f is None or not len(f):
            return ["frequency_hz", "power_dbfs", "peak_hold_dbfs"], []
        peak = self._peak if self._peak is not None else live
        return (
            ["frequency_hz", "power_dbfs", "peak_hold_dbfs"],
            [[float(a), float(b), float(c)] for a, b, c in zip(f, live, peak)],
        )

    def set_channel(self, centre_hz: float) -> None:
        self.centre = centre_hz
        self._peak = None
        self._hist = None
        self._rows_filled = 0
        self._draw_mask()

    def _draw_mask(self) -> None:
        """Overlay the BLE channel plan so out-of-band interference is obvious."""
        for it in self._mask_items:
            self.plot.removeItem(it)
        self._mask_items = []
        half = self.sample_rate / 2
        for ch in range(40):
            f = channel_to_freq(ch)
            if abs(f - self.centre) > half:
                continue
            region = pg.LinearRegionItem(
                values=(f - 1e6, f + 1e6),
                brush=pg.mkBrush(90, 200, 120, 22),
                pen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(-10)
            self.plot.addItem(region)
            self._mask_items.append(region)
            txt = pg.TextItem(f"ch{ch}", color=(130, 210, 150), anchor=(0.5, 0))
            txt.setPos(f, -12)
            self.plot.addItem(txt)
            self._mask_items.append(txt)

    def update_spectrum(self, iq: np.ndarray) -> None:
        if iq is None or iq.size < 512:
            return
        n = 1024
        x = iq[:n].astype(np.complex128) * np.hanning(n)
        spec = np.fft.fftshift(np.abs(np.fft.fft(x)) ** 2)
        db = 10 * np.log10(np.maximum(spec / (n * n), 1e-14))
        freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / self.sample_rate)) + self.centre
        self.curve.setData(freqs, db)

        # Peak hold decays DOWNWARD by a fixed number of dB per frame.  Scaling
        # a dB value by 0.999 -- the obvious-looking "decay" -- moves a negative
        # number toward zero, so the trace climbed off the top of the chart.
        if self._peak is None or self._peak.shape != db.shape:
            self._peak = db.copy()
        else:
            np.maximum(self._peak - self.PEAK_DECAY_DB, db, out=self._peak)
        self.peak_curve.setData(freqs, self._peak)

        if self._hist is None or self._hist.shape[1] != n:
            # Prime the history with the first frame rather than with the floor
            # value, so a fresh waterfall does not show a solid block of
            # "silence" that was never actually observed.
            self._hist = np.tile(db, (self.history_rows, 1))
            self._rows_filled = 0
        self._hist = np.roll(self._hist, 1, axis=0)
        self._hist[0] = db
        self._rows_filled = min(self._rows_filled + 1, self.history_rows)

        # row-major: image[row, col] -> row is time, col is frequency, and the
        # rectangle maps columns onto real frequency and rows onto seconds ago.
        rows = max(self._rows_filled, 1)
        visible = self._hist[:rows]
        # Anchor the top of the colour scale just above the loudest thing on
        # screen, so the slider's window always contains the signal.
        top = float(np.percentile(visible, 99.9))
        self._wf_top = top if not np.isfinite(self._wf_top) else (
            0.9 * self._wf_top + 0.1 * top
        )
        self._apply_levels()
        self.wf_image.setImage(visible, autoLevels=False)
        span = freqs[-1] - freqs[0]
        height = rows * self._frame_dt
        self.wf_image.setRect(
            pg.QtCore.QRectF(float(freqs[0]), 0.0, float(span), float(height))
        )
        # Set the range on the spectrum only; the waterfall follows through the
        # x-axis link established in __init__.
        self.plot.setXRange(float(freqs[0]), float(freqs[-1]), padding=0)
        # A little padding, not none: with the range flush to the edge the
        # tick at 0 sits exactly on the boundary and its label is drawn half
        # outside the plot.
        self.wf_plot.setYRange(0.0, max(height, self._frame_dt), padding=0.02)


# --------------------------------------------------------------------------
# per-address time series
# --------------------------------------------------------------------------

class SeriesTab(QWidget):
    """A time series with one coloured curve per advertising address.

    Which addresses get drawn is decided at refresh time, not when a packet
    arrives.  Deciding at insertion time -- keeping the first N addresses seen
    -- looks equivalent and is not: advertising addresses rotate constantly, so
    the first twelve are mostly one-shot rotations, and the device the operator
    then selects has no series at all and the chart comes up blank.
    """

    #: Points further apart than this are not joined; the pen lifts instead.
    MAX_JOIN_S = 5.0
    #: How many addresses to retain samples for.
    MAX_TRACKED = 96
    #: How many to draw at once when no device is selected.
    MAX_DRAWN = 8

    def __init__(self, key: str, title: str, ylabel: str, getter,
                 units: str = "", maxlen: int = 600, parent=None):
        super().__init__(parent)
        self.getter = getter
        self.maxlen = maxlen
        self.focus: str | None = None

        lay = QVBoxLayout(self)
        bar = InfoBar(self, key, title, self._csv_rows)
        self.only_selected = QCheckBox("only selected device")
        self.only_selected.setChecked(True)
        self.only_selected.setToolTip(
            "Show only the device of the packet selected in the list. "
            "With nothing selected, the busiest devices are shown."
        )
        self.only_selected.stateChanged.connect(self.refresh)
        bar.insertWidget(1, self.only_selected)
        lay.addLayout(bar)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", ylabel, units=units)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.legend = self.plot.addLegend(offset=(-10, 10), labelTextSize="7pt")
        lay.addWidget(self.plot)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #888;")
        lay.addWidget(self.status)

        self.data: dict[str, tuple[deque, deque]] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.curves: dict[str, object] = {}

    # ------------------------------------------------------------------
    def _csv_rows(self):
        rows = []
        for key in self.drawn_keys():
            t, y = self.data[key]
            rows.extend([key, float(a), float(b)] for a, b in zip(t, y))
        return ["adva", "time_s", "value"], rows

    def set_focus(self, address: str | None) -> None:
        if address != self.focus:
            self.focus = address
            self.refresh()

    def add(self, rec) -> None:
        # A CRC failure means the address bytes are unreliable, so the "device"
        # it would create is fictional.  Feeding failures in produced 186
        # "devices" from 451 packets, each contributing a couple of stray points.
        if rec.is_event or not rec.crc_ok or not rec.adva:
            return
        v = self.getter(rec)
        if v is None or not math.isfinite(v):
            return
        key = rec.adva
        if key not in self.data:
            if len(self.data) >= self.MAX_TRACKED:
                self._evict()
            self.data[key] = (deque(maxlen=self.maxlen), deque(maxlen=self.maxlen))
        self.counts[key] += 1
        t, y = self.data[key]
        t.append(rec.timestamp_us / 1e6)
        y.append(v)

    def _evict(self) -> None:
        """Drop the address with the fewest samples to make room."""
        victim = min(self.data, key=lambda k: self.counts.get(k, 0))
        self.data.pop(victim, None)
        self.counts.pop(victim, None)
        curve = self.curves.pop(victim, None)
        if curve is not None:
            self.plot.removeItem(curve)

    def drawn_keys(self) -> list[str]:
        """Which addresses should be visible right now.

        Falls back to the busiest devices when the focused address has nothing
        to show -- which happens whenever the selected row is a CRC failure,
        since its address bytes are not trustworthy and never became a series.
        A chart that silently goes blank reads as a broken chart.
        """
        if self.only_selected.isChecked() and self.focus and self.focus in self.data:
            return [self.focus]
        return sorted(self.data, key=lambda k: -self.counts.get(k, 0))[: self.MAX_DRAWN]

    def status_text(self) -> str:
        if self.only_selected.isChecked() and self.focus:
            if self.focus in self.data:
                return f"showing {short_addr(self.focus)} only"
            return (
                f"{short_addr(self.focus)} has no verified packets - "
                f"showing the {len(self.drawn_keys())} busiest devices"
            )
        n = len(self.data)
        shown = len(self.drawn_keys())
        return f"showing {shown} of {n} devices (busiest first)"

    def refresh(self) -> None:
        wanted = self.drawn_keys()
        for key in list(self.curves):
            if key not in wanted:
                self.plot.removeItem(self.curves.pop(key))
        for key in wanted:
            t, y = self.data[key]
            curve = self.curves.get(key)
            if curve is None:
                curve = self.plot.plot(
                    pen=pg.mkPen(color_for(key), width=1),
                    symbol="o", symbolSize=4,
                    symbolBrush=color_for(key), symbolPen=None,
                    name=f"{short_addr(key)}  ({self.counts.get(key, 0)})",
                    connect="finite",
                )
                self.curves[key] = curve
            ts = np.fromiter(t, float)
            ys = np.fromiter(y, float)
            xs, yy = self._break_gaps(ts, ys)
            curve.setData(xs, yy, connect="finite")
        self.status.setText(self.status_text())

    @classmethod
    def _break_gaps(cls, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Insert NaN where samples are far apart so the pen lifts.

        Advertising is bursty and addresses rotate, so a series routinely has
        two points many seconds apart.  Joined, they draw a straight line right
        across the plot -- which is what made these charts unreadable.  With
        `connect='finite'` and a NaN between them, they simply do not connect.
        """
        if t.size < 2:
            return t, y
        gaps = np.flatnonzero(np.diff(t) > cls.MAX_JOIN_S)
        if gaps.size == 0:
            return t, y
        idx = gaps + 1
        return np.insert(t, idx, np.nan), np.insert(y, idx, np.nan)

    def clear(self) -> None:
        for c in self.curves.values():
            self.plot.removeItem(c)
        self.data.clear()
        self.curves.clear()
        self.counts.clear()
        if self.legend is not None:
            self.legend.clear()


# --------------------------------------------------------------------------
# feature scatter
# --------------------------------------------------------------------------

class ScatterTab(QWidget):
    """2D feature scatter with selectable axes and per-device baseline ellipses."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addLayout(InfoBar(self, "scatter", "Feature scatter", self._csv_rows))

        bar = QHBoxLayout()
        bar.addWidget(QLabel("X:"))
        self.x_box = QComboBox()
        self.x_box.addItems(FEATURE_VECTOR_KEYS)
        self.x_box.setCurrentText("cfo_ppm")
        bar.addWidget(self.x_box)
        bar.addWidget(QLabel("Y:"))
        self.y_box = QComboBox()
        self.y_box.addItems(FEATURE_VECTOR_KEYS)
        self.y_box.setCurrentText("modulation_index")
        bar.addWidget(self.y_box)
        self.by_cluster = QCheckBox("colour by cluster")
        self.by_cluster.setChecked(True)
        bar.addWidget(self.by_cluster)
        self.only_selected = QCheckBox("only selected device")
        bar.addWidget(self.only_selected)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        lay.addWidget(self.plot)

        self.scatter = pg.ScatterPlotItem(size=7, pen=None)
        self.plot.addItem(self.scatter)
        self.points: deque = deque(maxlen=4000)
        self.focus: str | None = None
        self._ellipses: list = []
        self.x_box.currentTextChanged.connect(self.refresh)
        self.y_box.currentTextChanged.connect(self.refresh)
        self.by_cluster.stateChanged.connect(self.refresh)
        self.only_selected.stateChanged.connect(self.refresh)

    def _csv_rows(self):
        xk, yk = self.x_box.currentText(), self.y_box.currentText()
        rows = []
        for rec in self.points:
            x, y = rec.feature(xk), rec.feature(yk)
            if math.isfinite(x) and math.isfinite(y):
                rows.append([rec.adva, rec.cluster_id, float(x), float(y)])
        return ["adva", "cluster", xk, yk], rows

    def set_focus(self, address: str | None) -> None:
        if address != self.focus:
            self.focus = address
            if self.only_selected.isChecked():
                self.refresh()

    def add(self, rec) -> None:
        if rec.is_event or rec.features is None or not rec.crc_ok:
            return
        self.points.append(rec)

    def refresh(self) -> None:
        xk = self.x_box.currentText()
        yk = self.y_box.currentText()
        self.plot.setLabel("bottom", xk)
        self.plot.setLabel("left", yk)
        only = self.only_selected.isChecked() and self.focus
        spots = []
        for rec in self.points:
            if only and rec.adva != self.focus:
                continue
            x = rec.feature(xk)
            y = rec.feature(yk)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            key = str(rec.cluster_id) if self.by_cluster.isChecked() else rec.adva
            c = color_for(key)
            spots.append({"pos": (x, y), "brush": pg.mkBrush(*c, 190), "data": rec})
        self.scatter.setData(spots)

    def draw_baselines(self, baselines: dict) -> None:
        """One-sigma and two-sigma ellipses per enrolled device."""
        for e in self._ellipses:
            self.plot.removeItem(e)
        self._ellipses = []
        xk, yk = self.x_box.currentText(), self.y_box.currentText()
        keys = list(FEATURE_VECTOR_KEYS)
        if xk not in keys or yk not in keys:
            return
        ix, iy = keys.index(xk), keys.index(yk)
        for addr, bl in baselines.items():
            if bl.stats.n < 10:
                continue
            mx, my = bl.stats.mean[ix], bl.stats.mean[iy]
            sx, sy = bl.stats.std[ix], bl.stats.std[iy]
            if not (sx > 0 and sy > 0):
                continue
            for k in (1.0, 2.0):
                el = pg.QtWidgets.QGraphicsEllipseItem(
                    mx - k * sx, my - k * sy, 2 * k * sx, 2 * k * sy
                )
                el.setPen(pg.mkPen(color_for(addr), width=1, style=Qt.PenStyle.DashLine))
                el.setBrush(pg.mkBrush(*color_for(addr), 18))
                el.setZValue(-5)
                self.plot.addItem(el)
                self._ellipses.append(el)

    def clear(self) -> None:
        self.points.clear()
        self.scatter.setData([])


# --------------------------------------------------------------------------
# selected packet
# --------------------------------------------------------------------------

class PacketDetailTab(QWidget):
    """Instantaneous-frequency trace and eye diagram for the selected packet."""

    MAX_EYE_TRACES = 120

    def __init__(self, sample_rate: float, parent=None) -> None:
        super().__init__(parent)
        self.sample_rate = sample_rate
        lay = QVBoxLayout(self)
        lay.addLayout(InfoBar(self, "packet", "Selected packet", self._csv_rows))
        self.caption = QLabel("select a packet in the list")
        lay.addWidget(self.caption)

        self.freq_plot = pg.PlotWidget()
        self.freq_plot.setLabel("bottom", "Time", units="s")
        self.freq_plot.setLabel("left", "Instantaneous frequency", units="Hz")
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.freq_curve = self.freq_plot.plot(
            pen=pg.mkPen((80, 170, 255), width=1), name="frequency"
        )

        # The envelope belongs on its own axis.  Rescaling it onto the frequency
        # axis by an arbitrary factor -- the previous approach -- made a clean
        # packet look like two noise traces stacked on each other.
        self.env_view = pg.ViewBox()
        self.freq_plot.scene().addItem(self.env_view)
        self.env_axis = pg.AxisItem("right")
        self.freq_plot.plotItem.layout.addItem(self.env_axis, 2, 3)
        self.env_axis.linkToView(self.env_view)
        self.env_view.setXLink(self.freq_plot.plotItem)
        self.env_axis.setLabel("Envelope |x|", color="#ffa040")
        self.env_curve = pg.PlotCurveItem(pen=pg.mkPen((255, 160, 60), width=1))
        self.env_view.addItem(self.env_curve)
        self.freq_plot.plotItem.vb.sigResized.connect(self._sync_env_view)

        self.ramp_region = pg.LinearRegionItem(
            brush=pg.mkBrush(255, 200, 80, 45), pen=pg.mkPen(None), movable=False
        )
        self.ramp_region.setZValue(-10)
        self.freq_plot.addItem(self.ramp_region)
        # Nominal decision rails at +/-250 kHz (h = 0.5 at 1 Msym/s) and zero.
        # Without them a noisy trace has no scale and every packet looks equally
        # bad; with them it is obvious at a glance whether the rails are being
        # reached.
        self.rails = []
        for y, style in ((0.0, Qt.PenStyle.SolidLine),
                         (250e3, Qt.PenStyle.DashLine),
                         (-250e3, Qt.PenStyle.DashLine)):
            ln = pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen((150, 150, 150, 110), width=1, style=style),
            )
            ln.setZValue(-8)
            self.freq_plot.addItem(ln)
            self.rails.append(ln)
        lay.addWidget(self.freq_plot, 2)

        self.eye_plot = pg.PlotWidget()
        self.eye_plot.setLabel("bottom", "Symbol periods")
        self.eye_plot.setLabel("left", "Frequency", units="Hz")
        self.eye_plot.showGrid(x=True, y=True, alpha=0.3)
        for y in (0.0, 250e3, -250e3):
            ln = pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen((150, 150, 150, 110), width=1,
                             style=Qt.PenStyle.SolidLine if y == 0 else Qt.PenStyle.DashLine),
            )
            ln.setZValue(-8)
            self.eye_plot.addItem(ln)
        lay.addWidget(self.eye_plot, 2)
        self._eye_curves: list = []

    def _csv_rows(self):
        t, f = self.freq_curve.getData()
        if t is None or not len(t):
            return ["time_s", "frequency_hz", "envelope"], []
        _, env = self.env_curve.getData()
        env = env if env is not None and len(env) == len(t) else [float("nan")] * len(t)
        return (
            ["time_s", "frequency_hz", "envelope"],
            [[float(a), float(b), float(c)] for a, b, c in zip(t, f, env)],
        )

    def _sync_env_view(self) -> None:
        self.env_view.setGeometry(self.freq_plot.plotItem.vb.sceneBoundingRect())
        self.env_view.linkedViewChanged(self.freq_plot.plotItem.vb, self.env_view.XAxis)

    def burst_window(self, rec, n_samples: int) -> tuple[int, int]:
        """Sample range of the burst itself within the retained slice."""
        sps = self.sample_rate / 1e6
        start = int(rec.sync_offset_in_slice)
        n_sym = getattr(rec, "n_symbols", 0) or 0
        if n_sym <= 0:
            # Fall back to the decoded length when the symbol count is absent.
            n_sym = 40 + (2 + max(rec.length, 0) + 3) * 8
        phase = getattr(rec, "sym_offset", None)
        if phase is None or not math.isfinite(phase):
            phase = sps / 2.0
        end = int(start + phase + n_sym * sps)
        return max(start, 0), min(end, n_samples)

    def show_packet(self, rec) -> None:
        self.freq_curve.setData([], [])
        self.env_curve.setData([], [])
        for c in self._eye_curves:
            self.eye_plot.removeItem(c)
        self._eye_curves = []

        if rec is None or getattr(rec, "iq", None) is None or len(rec.iq) < 32:
            self.caption.setText(
                "no retained samples for this packet"
                if rec is not None and not rec.is_event
                else "select a packet in the list"
            )
            return

        from ..dsp import (
            apply_channel_filter,
            design_channel_filter,
            instantaneous_frequency,
        )
        from ..features import MEASUREMENT_FILTER_HZ

        iq = rec.iq
        taps = design_channel_filter(self.sample_rate, MEASUREMENT_FILTER_HZ)
        f = instantaneous_frequency(apply_channel_filter(iq, taps), self.sample_rate)
        t = np.arange(f.size) / self.sample_rate

        b0, b1 = self.burst_window(rec, f.size)

        # Blank the frequency trace outside the burst.  The instantaneous
        # frequency of receiver noise is uniform over the whole +/-fs/2 range --
        # measured here at +/-3.95 MHz against the signal's +/-285 kHz -- so
        # drawing it auto-ranges the plot to fourteen times the signal's span
        # and squashes the actual GFSK into a thin band. It is also meaningless:
        # there is no carrier there whose frequency could be estimated.
        shown = f.astype(np.float64).copy()
        if b1 > b0:
            shown[:b0] = np.nan
            shown[b1:] = np.nan
        self.freq_curve.setData(t, shown, connect="finite")

        inside = f[b0:b1]
        if inside.size:
            lim = float(np.percentile(np.abs(inside), 99.5)) * 1.6
            lim = max(lim, 1e5)
            self.freq_plot.setYRange(-lim, lim, padding=0)

        env = np.abs(iq)
        self.env_curve.setData(np.arange(env.size) / self.sample_rate, env)
        self.env_view.setYRange(0.0, float(max(env.max(), 1e-6)) * 1.15, padding=0)
        self._sync_env_view()

        sync = rec.sync_offset_in_slice / self.sample_rate
        self.ramp_region.setRegion((max(sync - 50e-6, 0.0), sync + 5e-6))

        snr = ""
        if math.isfinite(rec.snr_db):
            snr = f"  SNR {rec.snr_db:.1f} dB"
        self.caption.setText(
            f"#{rec.number}  {rec.pdu_name}  {rec.adva or '(no address)'}  "
            f"CRC {'OK' if rec.crc_ok else 'FAIL'}  "
            f"RSSI {rec.rssi_dbfs:.1f} dBFS{snr}  h={rec.modulation_index:.3f}  "
            f"(frequency shown for the burst only)"
        )
        self._draw_eye(f, rec, b0, b1)

    def _draw_eye(self, f: np.ndarray, rec, b0: int, b1: int) -> None:
        """Fold the frequency trace at the recovered symbol phase.

        Two things matter here.  The fold has to start at the sub-sample phase
        timing recovery found -- folding on the integer sync index alone starts
        every segment at a different point in the symbol and the overlay smears
        into noise.  And it has to stay inside the burst: segments taken from
        the noise either side contribute full-scale garbage.
        """
        sps = self.sample_rate / 1e6
        phase = getattr(rec, "sym_offset", None)
        if phase is None or not math.isfinite(phase):
            phase = sps / 2.0
        start = rec.sync_offset_in_slice + phase - sps / 2.0
        span = int(round(2 * sps))
        if span < 4 or b1 - b0 < 4 * span:
            return
        n_sym = int((min(b1, f.size) - start - span) / sps)
        n_sym = max(min(n_sym, self.MAX_EYE_TRACES), 0)
        xs = np.arange(span) / sps
        pen = pg.mkPen((80, 170, 255, 70), width=1)
        lim = 0.0
        for k in range(n_sym):
            i0 = int(round(start + k * sps))
            if i0 < b0 or i0 + span > b1:
                continue
            seg = f[i0 : i0 + span]
            lim = max(lim, float(np.max(np.abs(seg))))
            self._eye_curves.append(self.eye_plot.plot(xs, seg, pen=pen))
        if lim > 0:
            self.eye_plot.setYRange(-lim * 1.15, lim * 1.15, padding=0)


# --------------------------------------------------------------------------
# angle of arrival
# --------------------------------------------------------------------------

class DirectionTab(QWidget):
    """Bearing per advertising address, from the two-antenna phase difference.

    Drawn as a half-plane fan because that is what a two-element array can
    actually resolve: the phase difference is symmetric about the array axis,
    so a source at +30 and one at -30 degrees are indistinguishable, and the
    broadside ambiguity is a property of the geometry rather than of this
    estimator.  Showing a full 360-degree compass would imply a resolution the
    hardware does not have.

    This is the plot where an impersonator and the device it copies separate
    spatially: they share an address, so they land on one row of the table, but
    they are rarely in the same direction.
    """

    MAXLEN = 400

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addLayout(InfoBar(self, "direction", "Direction (AoA)", self._csv_rows))

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#9aa2ad;")
        lay.addWidget(self.note)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=False, y=False)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.setXRange(-1.15, 1.15)
        self.plot.setYRange(-0.15, 1.15)
        self.plot.setMouseEnabled(x=False, y=False)
        lay.addWidget(self.plot, 1)

        self._draw_compass()
        self.scatter = pg.ScatterPlotItem(size=6, pen=None)
        self.plot.addItem(self.scatter)
        self._rays: list = []
        self._density: list = []
        self.data: dict[str, deque] = {}
        self.focus: str | None = None

    # ------------------------------------------------------------------
    def _draw_compass(self) -> None:
        """Arc, bearing spokes and the two antenna positions."""
        th = np.linspace(-np.pi / 2, np.pi / 2, 181)
        for r in (0.25, 0.5, 0.75, 1.0):
            c = self.plot.plot(
                r * np.sin(th), r * np.cos(th),
                pen=pg.mkPen((110, 118, 128, 130), width=1),
            )
            c.setZValue(-20)
        for deg in (-90, -60, -30, 0, 30, 60, 90):
            a = np.radians(deg)
            spoke = self.plot.plot(
                [0, np.sin(a)], [0, np.cos(a)],
                pen=pg.mkPen((90, 96, 105, 110), width=1, style=Qt.PenStyle.DashLine),
            )
            spoke.setZValue(-20)
            lbl = pg.TextItem(f"{deg}", color=(150, 158, 168), anchor=(0.5, 0.5))
            lbl.setPos(1.08 * np.sin(a), 1.08 * np.cos(a))
            self.plot.addItem(lbl)

        # the array itself: RX0 and RX1, half a wavelength apart
        ant = pg.ScatterPlotItem(
            [-0.06, 0.06], [0.0, 0.0], size=11, symbol="s",
            brush=pg.mkBrush(220, 180, 90, 230), pen=None,
        )
        ant.setZValue(-10)
        self.plot.addItem(ant)
        for x, name in ((-0.06, "RX0"), (0.06, "RX1")):
            t = pg.TextItem(name, color=(220, 180, 90), anchor=(0.5, 0.0))
            t.setPos(x, -0.03)
            self.plot.addItem(t)

    # ------------------------------------------------------------------
    def set_focus(self, address: str | None) -> None:
        if address != self.focus:
            self.focus = address
            self.refresh()

    def add(self, rec) -> None:
        if rec.is_event or not rec.crc_ok or not rec.adva:
            return
        aoa = rec.feature("aoa_deg")
        if aoa is None or not math.isfinite(aoa):
            return
        hist = self.data.setdefault(rec.adva, deque(maxlen=self.MAXLEN))
        hist.append((float(aoa), float(rec.rssi_dbfs)))

    #: Bearing histogram resolution, in degrees.
    BIN_DEG = 6

    def _density_curve(self, angles: np.ndarray) -> tuple:
        """Bearing histogram as a closed polar outline.

        The scatter alone cannot show bimodality: a strong transmitter puts
        every point at the same radius, because radius carries signal strength,
        so 400 packets draw one thin arc whether they came from one direction
        or two.  The density is what makes two lobes visible, and two lobes
        under one address is the thing worth seeing.
        """
        edges = np.arange(-90, 90 + self.BIN_DEG, self.BIN_DEG)
        counts, _ = np.histogram(np.clip(angles, -90, 90), bins=edges)
        if counts.max() <= 0:
            return np.zeros(0), np.zeros(0)
        centres = np.radians(edges[:-1] + self.BIN_DEG / 2.0)
        # 0.15 keeps an empty bin visible as a floor rather than collapsing to
        # the origin, which would make the outline meaningless.
        r = 0.15 + 0.80 * (counts / counts.max())
        xs = np.concatenate([[0.0], r * np.sin(centres), [0.0]])
        ys = np.concatenate([[0.0], r * np.cos(centres), [0.0]])
        return xs, ys

    def refresh(self) -> None:
        for r in self._rays + self._density:
            self.plot.removeItem(r)
        self._rays = []
        self._density = []

        if not self.data:
            self.note.setText(
                "No angle-of-arrival data. Enable <b>Dual antenna (AoA)</b> and "
                "restart the capture; both RX channels share one LO, so RX1 is a "
                "second coherent antenna at the same frequency."
            )
            self.scatter.setData([])
            return

        spots = []
        lines = []
        for addr, hist in self.data.items():
            if not hist:
                continue
            angles = np.array([a for a, _ in hist])
            rssi = np.array([r for _, r in hist])
            colour = color_for(addr)
            dim = self.focus is not None and addr != self.focus
            alpha = 60 if dim else 200
            # radius carries signal strength: strong sources sit further out
            lo, hi = -80.0, 0.0
            rad = np.clip((rssi - lo) / (hi - lo), 0.12, 1.0)
            th = np.radians(np.clip(angles, -90, 90))
            spots.extend(
                {"pos": (r * np.sin(t), r * np.cos(t)),
                 "brush": pg.mkBrush(*colour, alpha)}
                for t, r in zip(th, rad)
            )
            # the mean bearing, drawn as a ray so a device reads as a direction
            mean = float(np.degrees(np.arctan2(
                np.mean(np.sin(th)), np.mean(np.cos(th))
            )))
            a = np.radians(mean)
            ray = self.plot.plot(
                [0, np.sin(a)], [0, np.cos(a)],
                pen=pg.mkPen(*colour, width=3 if not dim else 1),
            )
            self._rays.append(ray)

            dx, dy = self._density_curve(angles)
            if dx.size:
                curve = self.plot.plot(
                    dx, dy,
                    pen=pg.mkPen(*colour, width=2 if not dim else 1),
                    fillLevel=None,
                    brush=pg.mkBrush(*colour, 40 if not dim else 12),
                )
                curve.setFillLevel(0)
                curve.setBrush(pg.mkBrush(*colour, 40 if not dim else 12))
                curve.setZValue(-15)
                self._density.append(curve)
            split = self._bearing_split(angles)
            warn = ""
            if split[0] >= 2.5:
                warn = (
                    f" &mdash; <b>two lobes, {split[0]:.1f} sigma apart "
                    f"({split[1]:+.0f} and {split[2]:+.0f} deg)</b>"
                )
            lines.append(
                f"<span style='color:rgb{colour}'>&#9632;</span> "
                f"{short_addr(addr)}: {mean:+.0f} deg "
                f"(spread {np.degrees(np.std(th)):.0f} deg, "
                f"{len(hist)} packets){warn}"
            )
        self.scatter.setData(spots)
        self.note.setText(
            "Bearing relative to the array axis; +/-90 is endfire, 0 is "
            "broadside. A two-element array cannot tell +theta from -theta.<br>"
            + "<br>".join(lines)
        )

    @staticmethod
    def _bearing_split(angles: np.ndarray) -> tuple:
        """Best two-lobe split of a bearing distribution, in pooled sigma.

        Two transmitters sharing an address are rarely in the same direction,
        so this is an independent check on the RF fingerprint -- it holds even
        if their oscillators happen to match.
        """
        v = np.sort(np.asarray(angles, dtype=float))
        n = v.size
        k = max(int(n * 0.15), 3)
        if n < 40 or n - k <= k:
            return 0.0, 0.0, 0.0
        best = (0.0, 0.0, 0.0)
        c1 = np.cumsum(v)
        c2 = np.cumsum(v * v)
        for i in range(k, n - k):
            n_lo, n_hi = float(i), float(n - i)
            m_lo = c1[i - 1] / n_lo
            m_hi = (c1[-1] - c1[i - 1]) / n_hi
            v_lo = max(c2[i - 1] / n_lo - m_lo**2, 0.0)
            v_hi = max((c2[-1] - c2[i - 1]) / n_hi - m_hi**2, 0.0)
            pooled = np.sqrt((v_lo + v_hi) / 2.0)
            if pooled > 1e-9:
                sig = (m_hi - m_lo) / pooled
                if sig > best[0]:
                    best = (float(sig), float(m_lo), float(m_hi))
        return best

    def _csv_rows(self):
        rows = []
        for addr, hist in self.data.items():
            rows.extend([addr, float(a), float(r)] for a, r in hist)
        return ["adva", "aoa_deg", "rssi_dbfs"], rows

    def clear(self) -> None:
        for r in self._rays + self._density:
            self.plot.removeItem(r)
        self._rays = []
        self._density = []
        self.data.clear()
        self.scatter.setData([])


# --------------------------------------------------------------------------
# interference
# --------------------------------------------------------------------------

class InterferenceTab(QWidget):
    """Noise floor, PDR versus RSSI and the interference classification."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addLayout(InfoBar(self, "interference", "Interference monitor",
                              self._csv_rows))
        self.label = QLabel("interference: -")
        self.label.setWordWrap(True)
        lay.addWidget(self.label)

        self.noise_plot = pg.PlotWidget()
        self.noise_plot.setLabel("left", "Noise floor (dBFS)")
        self.noise_plot.setLabel("bottom", "Time", units="s")
        self.noise_plot.showGrid(x=True, y=True, alpha=0.3)
        self.noise_curve = self.noise_plot.plot(
            pen=pg.mkPen((200, 120, 255), width=1), connect="finite"
        )
        lay.addWidget(self.noise_plot)

        self.pdr_plot = pg.PlotWidget()
        self.pdr_plot.setLabel("left", "Delivery ratio")
        self.pdr_plot.setLabel("bottom", "RSSI (dBFS)")
        self.pdr_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pdr_plot.setYRange(-0.05, 1.05)
        # Sparse RSSI bins must not be joined into one long diagonal.  Marker
        # area carries the sample count, so a bin with three packets is visibly
        # not the same evidence as a bin with three hundred.
        self.pdr_points = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(120, 220, 140, 200))
        self.pdr_plot.addItem(self.pdr_points)
        self.pdr_curve = self.pdr_plot.plot(
            pen=pg.mkPen((120, 220, 140, 140), width=1), connect="finite"
        )
        lay.addWidget(self.pdr_plot)

        self.noise_t: deque = deque(maxlen=1800)
        self.noise_v: deque = deque(maxlen=1800)
        self._pdr: dict = defaultdict(lambda: [0, 0])
        self._last_noise = float("nan")
        self._last_epoch = 0

    def _csv_rows(self):
        rows = [["noise_floor", float(t), float(v)]
                for t, v in zip(self.noise_t, self.noise_v) if math.isfinite(v)]
        for b, (ok, tot) in sorted(self._pdr.items()):
            if tot >= 3:
                rows.append(["pdr", float(b), ok / tot])
        return ["series", "x", "y"], rows

    def update_stats(self, stats, elapsed: float) -> None:
        v = stats.noise_floor_dbfs
        if math.isfinite(v):
            # Break the trace at a retune: the noise floor is a property of the
            # channel and is reset with it, so joining across the change would
            # draw a vertical line between two unrelated levels.
            if stats.epoch != self._last_epoch:
                self.noise_t.append(elapsed)
                self.noise_v.append(float("nan"))
                self._last_epoch = stats.epoch
            # The DSP stage republishes at 4 Hz while the GUI ticks at 25 Hz;
            # appending every tick built a staircase of repeated points.
            if not math.isclose(v, self._last_noise, abs_tol=1e-6):
                self.noise_t.append(elapsed)
                self.noise_v.append(v)
                self._last_noise = v
            self.noise_curve.setData(
                np.fromiter(self.noise_t, float),
                np.fromiter(self.noise_v, float),
                connect="finite",
            )
        i = stats.interference or {}
        if i:
            bw = i.get("occupied_bandwidth_hz", float("nan"))
            self.label.setText(
                f"class: <b>{i.get('classification','-')}</b> "
                f"(confidence {i.get('confidence',0):.0%}) &mdash; {i.get('detail','')}<br>"
                f"occupied BW {bw/1e6:.2f} MHz, "
                f"duty {i.get('duty_cycle',0):.1%}, "
                f"spectral kurtosis {i.get('spectral_kurtosis',float('nan')):.1f}, "
                f"reaction latency {i.get('reaction_latency_us',float('nan')):.1f} us"
            )

    def add(self, rec) -> None:
        if rec.is_event or not math.isfinite(rec.rssi_dbfs):
            return
        b = int(rec.rssi_dbfs // 3) * 3
        self._pdr[b][1] += 1
        if rec.crc_ok:
            self._pdr[b][0] += 1

    def refresh(self) -> None:
        pts = [(b, ok / tot, tot) for b, (ok, tot) in sorted(self._pdr.items()) if tot >= 3]
        if not pts:
            self.pdr_points.setData([])
            self.pdr_curve.setData([], [])
            return
        spots = [
            {"pos": (b, r), "size": float(np.clip(5 + 3 * np.log2(n), 5, 22))}
            for b, r, n in pts
        ]
        self.pdr_points.setData(spots)
        # Only join bins that are actually adjacent, so a gap in coverage stays
        # a gap rather than a line drawn across the whole plot.
        xs, ys = [], []
        prev_b = None
        for b, r, _ in pts:
            if prev_b is not None and b - prev_b > 3:
                xs.append(float("nan"))
                ys.append(float("nan"))
            xs.append(b)
            ys.append(r)
            prev_b = b
        self.pdr_curve.setData(
            np.array(xs, float), np.array(ys, float), connect="finite"
        )

    def clear(self) -> None:
        self.noise_t.clear()
        self.noise_v.clear()
        self._pdr.clear()
        self._last_noise = float("nan")
        self.noise_curve.setData([], [])
        self.pdr_points.setData([])
        self.pdr_curve.setData([], [])


# --------------------------------------------------------------------------
# the dock
# --------------------------------------------------------------------------

class PlotDock(QTabWidget):
    """The tabbed plot dock."""

    def __init__(self, sample_rate: float, parent=None) -> None:
        super().__init__(parent)
        self.spectrum = SpectrumTab(sample_rate)
        self.cfo = SeriesTab(
            "cfo", "Carrier frequency offset vs time", "CFO", lambda r: r.cfo_ppm, "ppm"
        )
        self.rssi = SeriesTab(
            "rssi", "RSSI vs time", "RSSI", lambda r: r.rssi_dbfs, "dBFS"
        )
        self.scatter = ScatterTab()
        self.packet = PacketDetailTab(sample_rate)
        self.direction = DirectionTab()
        self.interference = InterferenceTab()

        # Short tab labels: the full title is on the chart itself, and six long
        # ones squeezed the dock and started eliding.  Tooltips carry the rest.
        for widget, label, tip in (
            (self.spectrum, "Spectrum", "Spectrum and waterfall"),
            (self.cfo, "CFO", "Carrier frequency offset vs time"),
            (self.scatter, "Features", "Feature scatter"),
            (self.rssi, "RSSI", "RSSI vs time"),
            (self.packet, "Packet", "Selected packet: frequency trace and eye"),
            (self.direction, "AoA", "Direction (angle of arrival)"),
            (self.interference, "Interf.", "Interference monitor"),
        ):
            i = self.addTab(widget, label)
            self.setTabToolTip(i, tip)

    def add_records(self, records: list) -> None:
        for r in records:
            self.cfo.add(r)
            self.rssi.add(r)
            self.scatter.add(r)
            self.direction.add(r)
            self.interference.add(r)

    def set_focus(self, address: str | None) -> None:
        """Point every per-device plot at the address selected in the list."""
        self.cfo.set_focus(address)
        self.rssi.set_focus(address)
        self.scatter.set_focus(address)
        self.direction.set_focus(address)

    def refresh(self) -> None:
        self.cfo.refresh()
        self.rssi.refresh()
        self.interference.refresh()
        if self.currentWidget() is self.scatter:
            self.scatter.refresh()
        elif self.currentWidget() is self.direction:
            self.direction.refresh()

    def clear(self) -> None:
        self.cfo.clear()
        self.rssi.clear()
        self.scatter.clear()
        self.direction.clear()
        self.interference.clear()
