"""Main-window behaviour: controls, layout persistence, hex highlighting.

Built without starting a capture, so these run anywhere.  The pipeline is left
unstarted and the window is driven directly.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication, QPushButton

from sniffer.channels import ChannelPlan, channel_to_freq
from sniffer.gui.app import MainWindow
from sniffer.radio import RadioConfig


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    # Keep every test out of the real user settings.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    monkeypatch.chdir(tmp_path)
    cfg = RadioConfig(plan=ChannelPlan.from_args(channel=37), gain_db=45)
    w = MainWindow(cfg, autostart=False)
    yield w
    w.timer.stop()
    w.close()


# --------------------------------------------------------------------------
# channel / frequency / gain
# --------------------------------------------------------------------------

def test_settings_are_file_backed_so_they_can_be_isolated(win):
    """The default on Windows is the registry, which no test can sandbox."""
    assert win.settings.format() == QSettings.Format.IniFormat
    assert win.settings.fileName().endswith(".ini")


def test_frequency_field_shows_the_selected_channel(win):
    for ch in (38, 39, 0, 36):
        win.chan_box.setCurrentIndex(win.chan_box.findData(ch))
        win._on_channel_change(0)
        shown = float(win.freq_edit.text())
        assert shown == pytest.approx(channel_to_freq(ch) / 1e6, abs=1e-3)


def test_fixed_channel_owns_the_frequency_field(win):
    """A fixed channel displays its frequency read-only; only custom is typed.

    The channel selector and the frequency are the same setting; letting both
    be edited independently lets them disagree.
    """
    for ch in (37, 12, 39):
        win.chan_box.setCurrentIndex(win.chan_box.findData(ch))
        win._on_channel_change(0)
        assert win.freq_edit.isReadOnly(), f"channel {ch} should be read-only"
        assert float(win.freq_edit.text()) == pytest.approx(
            channel_to_freq(ch) / 1e6, abs=1e-3
        )

    win.chan_box.setCurrentIndex(win.chan_box.findData(-1))
    win._on_channel_change(0)
    assert not win.freq_edit.isReadOnly(), "custom frequency must be editable"


def test_a_read_only_field_cannot_retune(win):
    win.chan_box.setCurrentIndex(win.chan_box.findData(37))
    win._on_channel_change(0)
    before = win.cfg.plan.frequency_hz
    win.freq_edit.setText("2480.000")
    win._on_custom_freq()
    assert win.cfg.plan.frequency_hz == before


def test_frequency_field_is_right_aligned(win):
    assert win.freq_edit.alignment() & Qt.AlignmentFlag.AlignRight


def test_custom_frequency_tunes(win):
    win.chan_box.setCurrentIndex(win.chan_box.findData(-1))
    win._on_channel_change(0)
    win.freq_edit.setText("2480.000")
    win._on_custom_freq()
    assert win.cfg.plan.frequency_hz == pytest.approx(2.48e9)
    assert win.cfg.plan.channel == 39


def test_a_nonsense_frequency_is_rejected_without_retuning(win, monkeypatch):
    import sniffer.gui.app as A

    monkeypatch.setattr(A.QMessageBox, "warning", lambda *a, **k: None)
    win.chan_box.setCurrentIndex(win.chan_box.findData(-1))
    win._on_channel_change(0)
    before = win.cfg.plan.frequency_hz
    win.freq_edit.setText("not a number")
    win._on_custom_freq()
    assert win.cfg.plan.frequency_hz == before


def test_entry_fields_are_compact(win):
    """Both held far more space than their content needs."""
    assert win.freq_edit.width() <= 90
    assert win.gain_spin.width() <= 90


def test_spin_arrows_are_drawn_from_real_images(win):
    """The border-triangle trick renders as plain rectangles in a spin box.

    Qt reserves the ::up-arrow sub-control box and paints nothing into it, so
    the arrows have to be actual images.
    """
    import os

    from sniffer.gui.app import _arrow_icons

    icons = _arrow_icons()
    assert set(icons) == {"up", "down", "up_off", "down_off"}
    for path in icons.values():
        assert os.path.exists(path) and os.path.getsize(path) > 0

    css = win.styleSheet()
    assert "QSpinBox::up-arrow" in css and "QSpinBox::down-arrow" in css
    assert 'image:url("' in css
    assert "border-bottom:5px solid" not in css, "the rectangle version is back"


def test_menu_button_arrows_sit_beside_the_text(win):
    """Save inherited Export's bug: the indicator fell to the bottom corner."""
    css = win.styleSheet()
    assert "QToolButton::menu-indicator" in css
    assert "subcontrol-position:center right" in css
    assert "QToolButton#export::menu-indicator" not in css, "must apply to all"


def test_dropdown_menus_are_styled_for_the_dark_theme(win):
    """Popup menus take the desktop palette, not the window's."""
    css = win.styleSheet()
    assert "QMenu {" in css or "QMenu {{" in css
    assert "QMenu::item:selected" in css


def test_right_click_offers_and_performs_clear_selection(win):
    from sniffer.features import Measurement, PacketFeatures
    from sniffer.packet import PacketRecord

    feats = PacketFeatures(measurements={"cfo_ppm": Measurement(4.0, 0.3, "ppm")})
    win.model.append([
        PacketRecord(number=1, adva="AA:BB:CC:DD:EE:01", pdu_name="ADV_IND",
                     crc_ok=True, features=feats, raw_bytes=bytes(range(20)))
    ])
    win.table.setCurrentIndex(win.model.index(0, 0))
    assert win.table.currentIndex().isValid()
    assert win.plots.cfo.focus == "AA:BB:CC:DD:EE:01"

    win.clear_selection()
    assert not win.table.currentIndex().isValid()
    assert not win.table.selectionModel().hasSelection()
    assert win.plots.cfo.focus is None
    assert win.detail.rowCount() == 0
    assert win.hex.toPlainText() == ""

    # the context menu offers the same action, and is disabled with no row
    menu = win.selection_menu()
    labels = [a.text() for a in menu.actions()]
    assert labels == ["Clear selection"]
    assert not menu.actions()[0].isEnabled(), "nothing selected -> nothing to clear"
    win.table.setCurrentIndex(win.model.index(0, 0))
    assert win.selection_menu().actions()[0].isEnabled()


class _FakePipeline:
    """Enough of a pipeline to drive one _on_tick without a radio."""

    def __init__(self, records):
        self._records = list(records)
        self.alive = True
        self.ready = True
        self.rssi_cal_db = None
        self.log_lines = []
        self.stats = _TickStats()
        self.ring = _FakeRing()

    def poll_stats(self):
        return self.stats

    def drain(self, *a, **k):
        out, self._records = self._records, []
        return out

    def stop(self, *a, **k):
        pass

    def clear(self):
        pass


class _FakeRing:
    n_channels = 1

    def written(self):
        return 0

    def read(self, _i):
        return None


class _TickStats:
    calibrated = False
    clock_detail = "onboard VCTCXO"
    packets_per_s = 3.0
    crc_rate = 0.9
    noise_floor_dbfs = -40.0
    last_rssi_dbfs = -50.0
    clipping = False
    samples = 1000
    lost_samples = 0
    skipped_samples = 0
    usb_overruns = 0
    temperature_c = 30.0
    running = True
    epoch = 0
    interference = {}


def _dual_record(number, dphi):
    from sniffer.features import Measurement, PacketFeatures
    from sniffer.packet import PacketRecord

    feats = PacketFeatures(
        measurements={
            "cfo_ppm": Measurement(4.0, 0.3, "ppm"),
            "antenna_phase_deg": Measurement(dphi, 4.0, "deg"),
            "aoa_deg": Measurement(dphi / 3.0, 4.0, "deg"),
        }
    )
    return PacketRecord(
        number=number, adva="AA:BB:CC:DD:EE:01", pdu_name="ADV_IND",
        crc_ok=True, rssi_dbfs=-45.0, features=feats, n_antennas=2,
        raw_bytes=bytes(range(20)),
    )


def test_tick_collects_antenna_phase_without_crashing(win):
    """Regression: `_antenna_pairs` was used on every tick but never created.

    A patch anchored on the wrong line left the attribute uninitialised, and
    nothing exercised `_on_tick` with real records, so the first dual-antenna
    capture raised AttributeError on the GUI timer.
    """
    assert hasattr(win, "_antenna_pairs"), "attribute missing before any tick"
    win.pipeline = _FakePipeline([_dual_record(i, 30.0 + i) for i in range(5)])
    win._on_tick()
    assert len(win._antenna_pairs) == 5
    assert win._antenna_pairs[0][0] == pytest.approx(30.0)
    assert win.model.total == 5


def test_tick_ignores_single_antenna_and_failed_packets(win):
    from sniffer.features import Measurement, PacketFeatures
    from sniffer.packet import PacketRecord

    single = _dual_record(1, 20.0)
    single.n_antennas = 1
    failed = _dual_record(2, 20.0)
    failed.crc_ok = False
    win.pipeline = _FakePipeline([single, failed])
    win._on_tick()
    assert len(win._antenna_pairs) == 0


def test_every_attribute_touched_on_tick_exists_before_the_timer_starts(win):
    """The timer fires before any capture; nothing it reads may be missing."""
    for name in ("_antenna_pairs", "_stopping", "_stop_thread", "_stop_done",
                 "_starting", "_device_warned", "pipeline", "model", "plots"):
        assert hasattr(win, name), name
    win.pipeline = None
    win._on_tick()   # must be a no-op, not an exception


def test_stored_antenna_offset_is_applied(win):
    win.settings.setValue("antenna/phase_offset_deg", 42.0)
    assert win.antenna_offset_rad() == pytest.approx(np.radians(42.0))
    win.settings.setValue("antenna/phase_offset_deg", "nonsense")
    assert win.antenna_offset_rad() == 0.0


def test_stop_does_not_block_the_gui(win):
    """Teardown waits on three processes; it must not do that on this thread."""
    import time as _t

    class _Slow:
        alive = True

        def stop(self, *a, **k):
            _t.sleep(1.5)

        def clear(self):
            pass

    win.pipeline = _Slow()
    t0 = _t.time()
    win.stop_capture()
    assert _t.time() - t0 < 0.3, "stop_capture blocked the GUI thread"
    assert win.pipeline is None
    assert win._stopping is not None
    assert not win.btn_start.isEnabled(), "Start must wait for the teardown"

    for _ in range(60):
        win._poll_teardown()
        if win._stopping is None:
            break
        _t.sleep(0.1)
    assert win._stopping is None
    assert win.btn_start.isEnabled()


def test_start_refuses_while_a_teardown_is_in_flight(win, monkeypatch):
    import sniffer.gui.app as A

    made = []
    monkeypatch.setattr(A, "list_devices", lambda: [{"serial": "x"}])
    monkeypatch.setattr(A, "SnifferPipeline", lambda *a, **k: made.append(1))

    class _Slow:
        alive = True

        def stop(self, *a, **k):
            import time as _t

            _t.sleep(0.8)

        def clear(self):
            pass

    win.pipeline = _Slow()
    win.stop_capture()
    win.start_capture()
    assert not made, "a second pipeline was built while the first was closing"


def test_gain_range_is_stated(win):
    assert "-15" in win.gain_hint.text() and "60" in win.gain_hint.text()
    assert "10-15 dB below full scale" in win.gain_spin.toolTip()
    assert win.gain_spin.minimum() == -15 and win.gain_spin.maximum() == 60


def test_reference_and_receiver_calibration_are_reported_separately(win):
    """They are different things and were previously conflated.

    Pressing Calibrate measures receiver impairments; it cannot make ppm-scale
    features absolute, which needs a disciplined clock on the U.FL input.
    Reporting both under one "UNCALIBRATED" made calibration look broken.
    """
    class _Stats:
        calibrated = False
        clock_detail = "onboard VCTCXO"
        packets_per_s = 0.0
        crc_rate = 0.0
        noise_floor_dbfs = -40.0
        last_rssi_dbfs = float("nan")
        clipping = False
        samples = lost_samples = skipped_samples = 0
        usb_overruns = 0
        temperature_c = 30.0
        running = True

    win._update_status(_Stats())
    assert "Ref" in win.lbl_ref.text()
    assert "UNCALIBRATED" not in win.lbl_ref.text()
    assert "different thing" in win.lbl_ref.toolTip()

    # and the receiver-calibration field is its own indicator
    assert "never" in win.lbl_cal.text()
    from sniffer.calibration import calibrate_from_samples

    rng = np.random.default_rng(0)
    iq = (rng.normal(0, 0.01, 60_000) + 1j * rng.normal(0, 0.01, 60_000)).astype(
        "complex64"
    )
    win.calibration.add(calibrate_from_samples(
        iq, 8e6, source="test", channel=37, frequency_hz=2.402e9, gain_db=45,
    ))
    win._update_cal_label()
    assert "never" not in win.lbl_cal.text()
    assert "Rx cal" in win.lbl_cal.text()
    # the reference indicator must be untouched by that
    win._update_status(_Stats())
    assert win.lbl_ref.text() == "Ref: internal"


def test_sample_rate_is_in_the_status_bar_not_the_toolbar(win):
    assert "MSPS" in win.lbl_fs.text()
    assert win.statusBar().children()  # the label is parented to the status bar


# --------------------------------------------------------------------------
# buttons
# --------------------------------------------------------------------------

def test_start_and_stop_toggle(win):
    parent = win.btn_start.parentWidget()
    win._update_buttons()
    assert win.btn_start.isVisibleTo(parent), "idle should offer Start"
    assert not win.btn_stop.isVisibleTo(parent)

    class _Live:
        alive = True

        def stop(self, *a, **k):
            pass

        def clear(self):
            pass

    win.pipeline = _Live()
    win._update_buttons()
    assert win.btn_stop.isVisibleTo(parent), "running should offer Stop"
    assert not win.btn_start.isVisibleTo(parent)
    win.pipeline = None
    win._update_buttons()
    assert win.btn_start.isVisibleTo(parent)


def test_calibrate_is_always_offered(win):
    parent = win.btn_calib.parentWidget()
    win._update_buttons()
    assert win.btn_calib.isVisibleTo(parent) and win.btn_calib.isEnabled()
    win.clear_all()
    assert win.btn_calib.isVisibleTo(parent) and win.btn_calib.isEnabled()


def test_buttons_are_styled_as_buttons(win):
    for btn, name in (
        (win.btn_start, "start"), (win.btn_stop, "stop"),
        (win.btn_clear, "clear"), (win.btn_calib, "calib"),
    ):
        assert isinstance(btn, QPushButton)
        assert btn.objectName() == name
        assert f"QPushButton#{name}" in win.styleSheet()


def test_exports_are_one_menu(win):
    menu = win.export_btn.menu()
    assert menu is not None
    labels = [a.text() for a in menu.actions()]
    assert len(labels) == 4
    assert any("CSV" in x for x in labels)
    assert any("Parquet" in x for x in labels)
    assert any("PCAP" in x for x in labels)
    assert any("SigMF" in x for x in labels)


# --------------------------------------------------------------------------
# device presence
# --------------------------------------------------------------------------

def test_status_bar_reports_a_missing_device(win, monkeypatch):
    import sniffer.gui.app as A

    monkeypatch.setattr(A, "list_devices", lambda: [])
    assert win._refresh_device_state() is False
    assert "NOT CONNECTED" in win.lbl_device.text()

    monkeypatch.setattr(A, "list_devices", lambda: [{"serial": "abc"}])
    assert win._refresh_device_state() is True
    assert "connected" in win.lbl_device.text()


def test_start_without_a_device_shows_a_modal(win, monkeypatch):
    import sniffer.gui.app as A

    seen = {}
    monkeypatch.setattr(A, "list_devices", lambda: [])
    monkeypatch.setattr(
        A, "device_error",
        lambda parent, title, msg, detail="": seen.update(title=title, msg=msg),
    )
    win.start_capture()
    assert win.pipeline is None
    assert "No bladeRF" in seen.get("title", "")


# --------------------------------------------------------------------------
# plots dock
# --------------------------------------------------------------------------

def test_plots_can_be_hidden_and_brought_back(win):
    win.show_plots.setChecked(False)
    assert win.dock.isHidden()
    win.show_plots.setChecked(True)
    assert not win.dock.isHidden()


def test_closing_the_dock_unticks_the_checkbox(win):
    """Closing with the dock's own X used to be irreversible."""
    win.show_plots.setChecked(True)
    win.dock.setVisible(False)
    assert not win.show_plots.isChecked()
    win.show_plots.setChecked(True)
    assert not win.dock.isHidden()


# --------------------------------------------------------------------------
# dual antenna
# --------------------------------------------------------------------------

def test_dual_antenna_switches_the_rx_channels(win):
    assert tuple(win.cfg.rx_channels) == (0,)
    win.dual_antenna.setChecked(True)
    assert tuple(win.cfg.rx_channels) == (0, 1)
    assert "2 RX" in win.lbl_fs.text() or "x2" in win.lbl_fs.text()
    win.dual_antenna.setChecked(False)
    assert tuple(win.cfg.rx_channels) == (0,)


# --------------------------------------------------------------------------
# hex view
# --------------------------------------------------------------------------

def test_every_highlighted_octet_is_yellow(win):
    """The first octet used to come out blue: it was left as a text selection."""
    win.hex.set_data(bytes(range(40)))
    win.hex.highlight((2, 5))
    sels = win.hex.extraSelections()
    assert len(sels) == 5
    colours = {s.format.background().color().name() for s in sels}
    assert colours == {"#ffdc78"}, colours
    # and the caret must not add a sixth, differently coloured, selection
    assert not win.hex.textCursor().hasSelection()


def test_hex_highlight_clears_between_packets(win):
    win.hex.set_data(bytes(range(40)))
    win.hex.highlight((0, 4))
    assert win.hex.extraSelections()
    win.hex.set_data(bytes(range(20)))
    assert not win.hex.extraSelections()


# --------------------------------------------------------------------------
# layout persistence
# --------------------------------------------------------------------------

def test_column_widths_and_panes_persist(app, tmp_path, monkeypatch):
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    # setPath only governs IniFormat, which is why the window asks for it
    # explicitly; on Windows the default would be the real user's registry.
    monkeypatch.chdir(tmp_path)
    cfg = RadioConfig(plan=ChannelPlan.from_args(channel=37), gain_db=45)

    first = MainWindow(cfg, autostart=False)
    first.table.setColumnWidth(0, 137)
    first.table.setColumnWidth(5, 211)
    first.tree.setColumnWidth(0, 321)
    first.split.setSizes([400, 250, 150])
    saved = first.split.sizes()
    first.timer.stop()
    first._save_layout()
    first.close()

    second = MainWindow(cfg, autostart=False)
    second.timer.stop()
    try:
        assert second.table.columnWidth(0) == 137
        assert second.table.columnWidth(5) == 211
        assert second.tree.columnWidth(0) == 321
        assert second.split.sizes() == saved
    finally:
        second.close()


def test_fixed_columns_are_sized_to_their_widest_value(win):
    from sniffer.gui.model import COLUMN_SAMPLES, SIZE_TO_CONTENTS

    fm = win.table.fontMetrics()
    for col in SIZE_TO_CONTENTS:
        need = fm.horizontalAdvance(COLUMN_SAMPLES[col])
        got = win.table.columnWidth(col)
        assert got >= need, f"column {col} too narrow for its widest value"
        assert got <= need + 60, f"column {col} much wider than it needs"


def test_detail_field_column_fits_its_longest_label(win):
    from sniffer.features import Measurement, PacketFeatures
    from sniffer.packet import PacketRecord

    feats = PacketFeatures(
        measurements={
            "a_very_long_feature_name_indeed": Measurement(1.0, 0.1, "Hz"),
            "cfo_ppm": Measurement(5.0, 0.5, "ppm"),
        }
    )
    rec = PacketRecord(number=1, adva="AA:BB:CC:DD:EE:FF", pdu_name="ADV_IND",
                       crc_ok=True, features=feats, raw_bytes=bytes(range(20)))
    win.detail.set_record(rec)
    widest = win.detail.widest_field()
    assert widest
    win._fit_detail_column()
    need = win.tree.fontMetrics().horizontalAdvance(widest)
    assert win.tree.columnWidth(0) >= min(need, 620)
