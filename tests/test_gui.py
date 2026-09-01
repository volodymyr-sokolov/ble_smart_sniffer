"""GUI model and view tests, run offscreen so they work without a display.

These exercise the parts that carry logic -- the table model's filtering and
capacity handling, the detail tree's construction, the hex view's byte mapping
-- rather than trying to assert on pixels.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtWidgets import QApplication

from sniffer.features import Measurement, PacketFeatures
from sniffer.gui.filters import compile_filter
from sniffer.gui.model import COLUMNS, DetailTreeModel, PacketTableModel
from sniffer.packet import PacketRecord, make_event


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def mkrec(n=1, **kw):
    feats = PacketFeatures(
        measurements={
            "cfo_ppm": Measurement(kw.pop("cfo_ppm", 5.0), 0.4, "ppm",
                                   calibrated=kw.pop("cal", True)),
            "cfo_hz": Measurement(12000.0, 900.0, "Hz", True, -150e3, 150e3),
            "modulation_index": Measurement(0.5, 0.01, "", True, 0.45, 0.55),
            "rise_time_us": Measurement(2.0, 0.1, "us"),
            "rssi_dbfs": Measurement(-50.0, 0.5, "dBFS"),
        }
    )
    base = dict(
        number=n, timestamp_us=n * 1000.0, wall_time=1.7e9,
        adva="AA:BB:CC:DD:EE:FF", adva_kind="random static",
        pdu_name="ADV_IND", pdu_type=0, length=30, crc_ok=True,
        rssi_dbfs=-50.0, channel=37, features=feats,
        raw_bytes=bytes(range(40)), payload=bytes(range(30)),
        ad_structures=[{"type": 0x01, "name": "Flags", "raw": b"\x06",
                        "value": "0x06", "truncated": False}],
        info="Flags",
    )
    base.update(kw)
    return PacketRecord(**base)


# --------------------------------------------------------------------------
# table model
# --------------------------------------------------------------------------

def test_model_columns_match_spec(app):
    m = PacketTableModel()
    names = [c[0] for c in COLUMNS]
    for required in ("#", "Ch", "PDU type", "AdvA", "RSSI (dBm)", "CFO (ppm)",
                     "Mod idx", "CRC", "Cluster", "Anomaly", "Info"):
        assert required in names
    assert m.columnCount() == len(COLUMNS)


def test_headers_are_always_bold(app):
    """Qt un-bolds the current column's header by default, which reads as a bug."""
    m = PacketTableModel()
    for col in range(m.columnCount()):
        f = m.headerData(col, Qt.Orientation.Horizontal, Qt.ItemDataRole.FontRole)
        assert f is not None and f.bold(), f"column {col} header not bold"


def test_numeric_columns_are_right_aligned(app):
    from sniffer.gui.model import RIGHT_ALIGNED

    m = PacketTableModel()
    m.append([mkrec(1)])
    for col in range(m.columnCount()):
        align = m.data(m.index(0, col), Qt.ItemDataRole.TextAlignmentRole)
        right = bool(int(align) & int(Qt.AlignmentFlag.AlignRight))
        assert right == (col in RIGHT_ALIGNED), f"column {col} alignment"
    # the columns the brief named, by title
    titles = {COLUMNS[i][0] for i in RIGHT_ALIGNED}
    for name in ("#", "Time (us)", "Ch", "Len", "RSSI (dBm)", "CFO (ppm)",
                 "Mod idx", "Cluster"):
        assert name in titles


def test_append_and_render(app):
    m = PacketTableModel()
    m.append([mkrec(i) for i in range(1, 6)])
    assert m.rowCount() == 5
    idx = m.index(0, 5)
    assert m.data(idx, Qt.ItemDataRole.DisplayRole) == "AA:BB:CC:DD:EE:FF"
    assert m.data(m.index(0, 11), Qt.ItemDataRole.DisplayRole) == "OK"


def test_capacity_is_bounded_and_view_stays_consistent(app):
    m = PacketTableModel(capacity=50)
    for chunk in range(4):
        m.append([mkrec(chunk * 40 + i) for i in range(40)])
    assert m.total == 50
    assert m.rowCount() == 50
    # the newest record must still be present and renderable
    assert m.record(49) is not None
    assert m.data(m.index(49, 0), Qt.ItemDataRole.DisplayRole) != ""


def test_filter_applies_to_existing_and_new_rows(app):
    m = PacketTableModel()
    m.append([mkrec(1, crc_ok=True), mkrec(2, crc_ok=False)])
    m.set_filter(compile_filter("crc == fail"), "crc == fail")
    assert m.rowCount() == 1
    assert m.record(0).number == 2
    m.append([mkrec(3, crc_ok=False), mkrec(4, crc_ok=True)])
    assert m.rowCount() == 2
    assert m.total == 4


def test_crc_failure_is_greyed(app):
    from sniffer.gui.model import COLOR_CRC_FAIL

    m = PacketTableModel()
    m.append([mkrec(1, crc_ok=False)])
    fg = m.data(m.index(0, 0), Qt.ItemDataRole.ForegroundRole)
    assert fg is not None and fg.color() == COLOR_CRC_FAIL


def test_alert_rows_are_coloured(app):
    m = PacketTableModel()
    m.append([mkrec(1, alerts=["two separated feature clusters under X (4.5 sigma)"])])
    bg = m.data(m.index(0, 0), Qt.ItemDataRole.BackgroundRole)
    assert bg is not None
    c = bg.color()
    assert c.red() > c.green() and c.red() > c.blue()  # red-ish


def test_event_rows_render_inline(app):
    m = PacketTableModel()
    m.append([mkrec(1), make_event(2, 5000.0, 37, "interference", "reactive emitter")])
    assert m.rowCount() == 2
    info_col = [c[0] for c in COLUMNS].index("Info")
    assert m.data(m.index(1, info_col), Qt.ItemDataRole.DisplayRole) == "reactive emitter"
    assert m.data(m.index(1, 4), Qt.ItemDataRole.DisplayRole) == "interference"


def test_uncalibrated_is_marked_in_the_header_not_every_cell(app):
    """The caveat belongs once in the header, not repeated on every row."""
    m = PacketTableModel()
    m.append([mkrec(1, calibrated=False)])
    cell = m.data(m.index(0, 9), Qt.ItemDataRole.DisplayRole)
    assert not cell.endswith("U")
    float(cell)  # a bare number, parseable

    m.set_units(rssi_in_dbm=False, calibrated=False)
    head = m.headerData(9, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
    assert head.startswith("CFO (ppm)") and "*" in head
    tip = m.headerData(9, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole)
    assert "UNCALIBRATED" in tip

    m.set_units(rssi_in_dbm=False, calibrated=True)
    assert m.headerData(
        9, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
    ) == "CFO (ppm)"


def test_rssi_units_are_stated_in_the_header(app):
    m = PacketTableModel()
    m.append([mkrec(1)])  # rssi_dbm is NaN by default
    cell = m.data(m.index(0, 8), Qt.ItemDataRole.DisplayRole)
    assert not cell.endswith(" F")
    float(cell)

    m.set_units(rssi_in_dbm=False, calibrated=False)
    assert m.headerData(
        8, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
    ) == "RSSI (dBFS)"
    m.set_units(rssi_in_dbm=True, calibrated=False)
    assert m.headerData(
        8, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
    ) == "RSSI (dBm)"


# --------------------------------------------------------------------------
# detail tree
# --------------------------------------------------------------------------

def test_aoa_columns_exist_and_are_blank_without_a_second_antenna(app):
    """Only --dual-antenna produces these; they must not invent a value."""
    names = [c[0] for c in COLUMNS]
    assert "AoA (deg)" in names and "d-phi (deg)" in names
    m = PacketTableModel()
    m.append([mkrec(1)])
    for title in ("AoA (deg)", "d-phi (deg)"):
        col = names.index(title)
        assert m.data(m.index(0, col), Qt.ItemDataRole.DisplayRole) == ""
        tip = m.headerData(col, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole)
        assert "dual-antenna" in tip


def test_aoa_columns_render_when_present(app):
    names = [c[0] for c in COLUMNS]
    rec = mkrec(1)
    rec.features.measurements["aoa_deg"] = Measurement(-27.5, 3.0, "deg")
    rec.features.measurements["antenna_phase_deg"] = Measurement(-81.2, 4.0, "deg")
    m = PacketTableModel()
    m.append([rec])
    assert m.data(
        m.index(0, names.index("AoA (deg)")), Qt.ItemDataRole.DisplayRole
    ) == "-27.5"
    assert m.data(
        m.index(0, names.index("d-phi (deg)")), Qt.ItemDataRole.DisplayRole
    ) == "-81.2"


def test_detail_tree_sections(app):
    t = DetailTreeModel()
    t.set_record(mkrec(1))
    names = [
        t.data(t.index(r, 0, QModelIndex()), Qt.ItemDataRole.DisplayRole)
        for r in range(t.rowCount(QModelIndex()))
    ]
    for section in ("Frame", "Radio", "PHY features", "Link layer",
                    "Advertising data", "CRC"):
        assert section in names, names


def test_detail_tree_shows_spec_limits(app):
    t = DetailTreeModel()
    t.set_record(mkrec(1))
    found = []

    def walk(parent, depth=0):
        for r in range(t.rowCount(parent)):
            i0 = t.index(r, 0, parent)
            found.append((
                t.data(i0, Qt.ItemDataRole.DisplayRole),
                t.data(t.index(r, 2, parent), Qt.ItemDataRole.DisplayRole),
            ))
            walk(i0, depth + 1)

    walk(QModelIndex())
    cfo = [v for k, v in found if k == "cfo_hz"]
    assert cfo and "150000" in cfo[0].replace("+", "")


def test_detail_tree_byte_ranges_map_into_the_pdu(app):
    t = DetailTreeModel()
    rec = mkrec(1)
    t.set_record(rec)

    ranges = []

    def walk(parent):
        for r in range(t.rowCount(parent)):
            i0 = t.index(r, 0, parent)
            br = t.data(i0, Qt.ItemDataRole.UserRole)
            if br:
                ranges.append(br)
            walk(i0)

    walk(QModelIndex())
    assert ranges
    for start, length in ranges:
        assert 0 <= start < len(rec.raw_bytes)
        assert length > 0


def test_detail_tree_handles_event_rows(app):
    t = DetailTreeModel()
    t.set_record(make_event(9, 1.0, 38, "channel change", "retuned to ch38"))
    assert t.rowCount(QModelIndex()) == 1


def test_detail_tree_empty_record(app):
    t = DetailTreeModel()
    t.set_record(None)
    assert t.rowCount(QModelIndex()) == 0


# --------------------------------------------------------------------------
# hex view
# --------------------------------------------------------------------------

def test_hex_view_layout_and_highlight(app):
    from sniffer.gui.app import HexView

    h = HexView()
    data = bytes(range(40))
    h.set_data(data)
    text = h.toPlainText()
    assert text.splitlines()[0].startswith("0000  00 01 02")
    assert len(text.splitlines()) == 3
    h.highlight((2, 4))
    assert len(h.extraSelections()) == 4


def test_hex_view_empty(app):
    from sniffer.gui.app import HexView

    h = HexView()
    h.set_data(b"")
    assert h.toPlainText() == ""
    h.highlight((0, 4))  # must not raise


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def test_plot_dock_accepts_batches(app):
    from sniffer.gui.plots import PlotDock

    d = PlotDock(8e6)
    recs = [mkrec(i, adva=f"AA:BB:CC:DD:EE:{i%4:02X}") for i in range(1, 40)]
    d.add_records(recs)
    d.refresh()
    d.scatter.refresh()
    assert len(d.cfo.data) == 4
    d.spectrum.update_spectrum(
        (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64) * 0.05
    )
    d.clear()
    assert not d.cfo.data


def test_selected_packet_plot_handles_missing_iq(app):
    from sniffer.gui.plots import PacketDetailTab

    t = PacketDetailTab(8e6)
    t.show_packet(mkrec(1))  # iq is None
    rec = mkrec(2)
    rec.iq = (np.random.randn(3000) + 1j * np.random.randn(3000)).astype(np.complex64)
    rec.sync_offset_in_slice = 400
    t.show_packet(rec)
