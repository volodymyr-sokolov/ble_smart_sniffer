"""Modal dialogs: calibration history, and device-error reporting."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


def _fmt(v, spec="{:.3f}", dash="-"):
    if v is None:
        return dash
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return dash
        return spec.format(v)
    return str(v)


def calibration_html(entry: dict) -> str:
    """One calibration run rendered as a readable report."""
    if not entry:
        return "<p>No calibration has been run yet.</p>"

    rows = [
        ("When", entry.get("when") or entry.get("timestamp")),
        ("Source", entry.get("source")),
        ("Channel", f"{entry.get('channel')}  ({_fmt(entry.get('frequency_hz'), '{:.3f}')} Hz)"),
        ("Sample rate", f"{_fmt(entry.get('sample_rate', 0) / 1e6, '{:.3f}')} MSPS"),
        ("RX gain", f"{entry.get('gain_db')} dB"),
        ("Samples analysed", f"{entry.get('samples', 0):,}"),
    ]
    levels = [
        ("Noise floor", f"{_fmt(entry.get('noise_floor_dbfs'), '{:.2f}')} dBFS"),
        ("RMS level", f"{_fmt(entry.get('rms_dbfs'), '{:.2f}')} dBFS"),
        ("Peak level", f"{_fmt(entry.get('peak_dbfs'), '{:.2f}')} dBFS"),
        ("Clipping", _fmt(entry.get("clipping"))),
    ]
    impair = [
        ("DC offset I", _fmt(entry.get("dc_i"), "{:+.6f}")),
        ("DC offset Q", _fmt(entry.get("dc_q"), "{:+.6f}")),
        ("DC magnitude", f"{_fmt(entry.get('dc_magnitude_dbfs'), '{:.1f}')} dBFS"),
        ("IQ gain imbalance", _fmt(entry.get("gain_imbalance"), "{:.5f}")),
        ("Quadrature skew", f"{_fmt(entry.get('quadrature_skew_deg'), '{:+.3f}')} deg"),
        ("Image rejection", f"{_fmt(entry.get('image_rejection_db'), '{:.1f}')} dB"),
    ]
    array = [
        ("Packets used", entry.get("antenna_packets", 0)),
        ("Phase offset", f"{_fmt(entry.get('antenna_phase_offset_deg'), '{:+.2f}')} deg"),
        ("Phase spread", f"{_fmt(entry.get('antenna_phase_spread_deg'), '{:.1f}')} deg"),
        ("Coherence", _fmt(entry.get("antenna_coherence"), "{:.3f}")),
    ]
    env = [
        ("RFIC temperature", f"{_fmt(entry.get('temperature_c'), '{:.1f}')} C"),
        ("Reference", "locked" if entry.get("calibrated_reference") else "UNCALIBRATED"),
        ("Clock", entry.get("clock_detail") or "-"),
    ]

    def table(title, items):
        body = "".join(
            f"<tr><td style='padding:2px 16px 2px 0;white-space:nowrap;'>{k}</td>"
            f"<td style='padding:2px 0;'><b>{v}</b></td></tr>"
            for k, v in items
        )
        return f"<h3>{title}</h3><table cellspacing='0'>{body}</table>"

    notes = entry.get("notes") or []
    note_html = "".join(
        f"<li>{n}</li>" for n in notes
    ) or "<li>no notes</li>"
    verdict = (
        "<p style='color:#1a7f37;'><b>Result: usable</b></p>"
        if entry.get("ok", True)
        else "<p style='color:#b00000;'><b>Result: action needed</b></p>"
    )

    return (
        f"<h2>Calibration report</h2>{verdict}"
        + table("Configuration", rows)
        + table("Levels", levels)
        + table("Receiver impairments (corrected before feature extraction)", impair)
        + (table("Antenna array", array) if entry.get("antenna_packets") else "")
        + table("Environment", env)
        + f"<h3>Notes</h3><ul>{note_html}</ul>"
    )


class CalibrationDialog(QDialog):
    """History on the left, the selected run's full report on the right."""

    def __init__(self, history, parent=None) -> None:
        super().__init__(parent)
        self.history = history
        self.setWindowTitle("Calibration history")
        self.setModal(True)
        self.resize(1040, 700)

        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(QLabel(f"<b>{len(history.entries)} calibration run(s)</b>"))
        head.addStretch(1)
        drift = history.drift("noise_floor_dbfs")
        if drift:
            head.addWidget(
                QLabel(
                    f"noise floor drift across the log: "
                    f"{drift[0]:.1f} -> {drift[1]:.1f} dBFS"
                )
            )
        lay.addLayout(head)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.list = QTextBrowser()
        self.list.setOpenLinks(False)
        self.list.anchorClicked.connect(self._select)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.list.setFont(mono)
        split.addWidget(self.list)

        self.detail = QTextBrowser()
        split.addWidget(self.detail)
        split.setSizes([380, 660])
        lay.addWidget(split, 1)

        row = QHBoxLayout()
        clear = QPushButton("Clear history")
        clear.clicked.connect(self._clear)
        row.addWidget(clear)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        row.addWidget(buttons)
        lay.addLayout(row)

        self._refresh()
        if history.entries:
            self._show(len(history.entries) - 1)

    def _refresh(self) -> None:
        if not self.history.entries:
            self.list.setHtml("<p>No calibration runs recorded.</p>")
            self.detail.setHtml(calibration_html({}))
            return
        rows = []
        for i, e in enumerate(reversed(self.history.entries)):
            idx = len(self.history.entries) - 1 - i
            flag = "" if e.get("ok", True) else "  [!]"
            ref = "locked" if e.get("calibrated_reference") else "UNCAL"
            rows.append(
                f"<a href='{idx}'>{e.get('when', '')}</a>  "
                f"ch{e.get('channel')}  g{e.get('gain_db')}  "
                f"noise {_fmt(e.get('noise_floor_dbfs'), '{:.1f}')}  "
                f"peak {_fmt(e.get('peak_dbfs'), '{:.1f}')}  {ref}{flag}"
            )
        self.list.setHtml("<br>".join(rows))

    def _select(self, url) -> None:
        try:
            self._show(int(url.toString()))
        except (ValueError, IndexError):
            pass

    def _show(self, index: int) -> None:
        self.detail.setHtml(calibration_html(self.history.entries[index]))

    def _clear(self) -> None:
        if (
            QMessageBox.question(
                self,
                "Clear calibration history",
                "Delete every recorded calibration run?",
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.history.clear()
            self._refresh()
            self.detail.setHtml(calibration_html({}))


def device_error(parent, title: str, message: str, detail: str = "") -> None:
    """A modal for a device problem the operator has to know about."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if detail:
        box.setInformativeText(detail)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()
