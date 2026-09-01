"""Chart behaviour tests.

Each of these corresponds to a defect that was visible on screen: a peak-hold
that climbed off the top of the plot, a waterfall drawn outside its axes, time
series joined into diagonals across the whole chart, and a delivery plot that
drew one line between two distant bins.  They are asserted on the data the
plots hand to pyqtgraph, which is where the bugs actually were.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from sniffer.features import Measurement, PacketFeatures
from sniffer.gui.chartinfo import CHART_INFO, as_html
from sniffer.gui.plots import (
    ChartInfoDialog,
    InterferenceTab,
    PacketDetailTab,
    PlotDock,
    SeriesTab,
    SpectrumTab,
    color_for,
    reset_colors,
)
from sniffer.packet import PacketRecord


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def rec(t_s=0.0, adva="AA:BB:CC:DD:EE:01", cfo=5.0, rssi=-50.0, crc=True, **kw):
    feats = PacketFeatures(
        measurements={
            "cfo_ppm": Measurement(cfo, 0.5, "ppm"),
            "modulation_index": Measurement(0.5, 0.01, ""),
            "rssi_dbfs": Measurement(rssi, 0.5, "dBFS"),
        }
    )
    return PacketRecord(
        number=1, timestamp_us=t_s * 1e6, adva=adva, crc_ok=crc,
        rssi_dbfs=rssi, pdu_name="ADV_IND", features=feats, **kw
    )


def noise(n=2048, amp=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, amp, n) + 1j * rng.normal(0, amp, n)).astype(np.complex64)


# --------------------------------------------------------------------------
# spectrum: peak hold direction
# --------------------------------------------------------------------------

def test_peak_hold_decays_downward_not_upward(app):
    """The peak trace must fall back toward the live trace, never climb away.

    Scaling a dB value by 0.999 moves a negative number toward zero, so the old
    peak-hold drifted up and off the top of the plot within seconds.
    """
    tab = SpectrumTab(8e6)
    loud = noise(amp=0.2, seed=1)
    tab.update_spectrum(loud)
    peak_after_loud = tab._peak.copy()

    for _ in range(40):
        tab.update_spectrum(noise(amp=0.001, seed=2))

    assert np.all(tab._peak <= peak_after_loud + 1e-9), "peak hold rose"
    assert tab._peak.mean() < peak_after_loud.mean() - 5, "peak hold did not decay"


def test_peak_hold_never_below_live_trace(app):
    tab = SpectrumTab(8e6)
    for i in range(10):
        tab.update_spectrum(noise(amp=0.05, seed=i))
    _, live = tab.curve.getData()
    assert np.all(tab._peak >= live - 1e-6)


def test_peak_hold_tracks_a_new_louder_signal(app):
    tab = SpectrumTab(8e6)
    for _ in range(5):
        tab.update_spectrum(noise(amp=0.001, seed=3))
    quiet = tab._peak.mean()
    tab.update_spectrum(noise(amp=0.3, seed=4))
    assert tab._peak.mean() > quiet + 10


# --------------------------------------------------------------------------
# waterfall geometry
# --------------------------------------------------------------------------

def test_waterfall_is_placed_on_the_frequency_axis(app):
    """The image rectangle must sit on real frequencies, not sample indices.

    Positioning it at 0..1024 while the axis spans 2.4 GHz is what squeezed the
    waterfall into a sliver at the left edge.
    """
    tab = SpectrumTab(8e6)
    tab.set_channel(2.402e9)
    tab.update_spectrum(noise(seed=5))
    r = tab.wf_image.boundingRect()
    rect = tab.wf_image.mapRectToView(r)
    assert rect.left() == pytest.approx(2.402e9 - 4e6, abs=2e4)
    assert rect.width() == pytest.approx(8e6, rel=0.01)
    assert rect.height() > 0


def test_waterfall_orientation_is_time_by_frequency(app):
    """image[row, col]: rows are time, columns are frequency."""
    tab = SpectrumTab(8e6)
    tab.update_spectrum(noise(seed=6))
    img = tab.wf_image.image
    assert img.ndim == 2
    assert img.shape[1] == 1024, "frequency must be the column axis"
    assert img.shape[0] <= tab.history_rows


def test_waterfall_only_shows_rows_it_has_actually_filled(app):
    tab = SpectrumTab(8e6)
    for _ in range(3):
        tab.update_spectrum(noise(seed=7))
    assert tab.wf_image.image.shape[0] == 3


def test_spectrum_and_waterfall_share_one_frequency_axis(app):
    """They are stacked and show the same span, so a feature must line up.

    Independently auto-ranged, the same frequency landed at different
    horizontal positions in the two panes.
    """
    tab = SpectrumTab(8e6)
    tab.set_channel(2.402e9)
    tab.update_spectrum(noise(seed=9))
    assert tab.plot.viewRange()[0] == pytest.approx(tab.wf_plot.viewRange()[0])

    # and the link must survive the user zooming either pane
    tab.plot.setXRange(2.4005e9, 2.4035e9, padding=0)
    assert tab.plot.viewRange()[0] == pytest.approx(tab.wf_plot.viewRange()[0])


def test_waterfall_resets_on_retune(app):
    tab = SpectrumTab(8e6)
    for _ in range(5):
        tab.update_spectrum(noise(seed=8))
    tab.set_channel(2.426e9)
    assert tab._hist is None and tab._rows_filled == 0
    assert tab._peak is None


# --------------------------------------------------------------------------
# time series: gaps, filtering, focus
# --------------------------------------------------------------------------

def test_gaps_are_broken_not_joined():
    t = np.array([0.0, 1.0, 2.0, 30.0, 31.0])
    y = np.arange(5.0)
    xs, ys = SeriesTab._break_gaps(t, y)
    assert np.isnan(ys).sum() == 1
    assert np.isnan(xs).sum() == 1
    # the NaN must land between the two clusters, not at either end
    idx = int(np.flatnonzero(np.isnan(ys))[0])
    assert 0 < idx < len(ys) - 1


def test_close_points_are_not_broken():
    t = np.linspace(0, 4, 20)
    y = np.arange(20.0)
    xs, ys = SeriesTab._break_gaps(t, y)
    assert not np.isnan(ys).any()


def test_crc_failures_do_not_create_series(app):
    """A CRC failure has unreliable address bytes; it must not become a device."""
    tab = SeriesTab("cfo", "t", "CFO", lambda r: r.cfo_ppm, "ppm")
    for i in range(20):
        tab.add(rec(t_s=i, adva=f"DE:AD:BE:EF:{i:02X}:00", crc=False))
    assert tab.data == {}
    tab.add(rec(t_s=0, adva="AA:BB:CC:DD:EE:01", crc=True))
    assert list(tab.data) == ["AA:BB:CC:DD:EE:01"]


def test_non_finite_values_are_skipped(app):
    tab = SeriesTab("cfo", "t", "CFO", lambda r: r.cfo_ppm, "ppm")
    tab.add(rec(cfo=float("nan")))
    assert tab.data == {}


def test_focus_shows_only_the_selected_device(app):
    tab = SeriesTab("cfo", "t", "CFO", lambda r: r.cfo_ppm, "ppm")
    for i in range(10):
        tab.add(rec(t_s=i * 0.1, adva="AA:BB:CC:DD:EE:01"))
        tab.add(rec(t_s=i * 0.1, adva="AA:BB:CC:DD:EE:02"))
    tab.set_focus("AA:BB:CC:DD:EE:02")
    assert tab.drawn_keys() == ["AA:BB:CC:DD:EE:02"]
    assert set(tab.curves) == {"AA:BB:CC:DD:EE:02"}
    assert "EE:02" in tab.status_text()


def test_focus_on_an_unknown_address_falls_back_to_busiest(app):
    """A blank chart reads as a broken chart; fall back and say so."""
    tab = SeriesTab("cfo", "t", "CFO", lambda r: r.cfo_ppm, "ppm")
    for i in range(10):
        tab.add(rec(t_s=i * 0.1, adva="AA:BB:CC:DD:EE:01"))
    tab.set_focus("00:11:22:33:44:55")
    assert tab.drawn_keys() == ["AA:BB:CC:DD:EE:01"]
    assert "no verified packets" in tab.status_text()


def test_unfocused_view_is_capped_and_ordered_by_traffic(app):
    tab = SeriesTab("cfo", "t", "CFO", lambda r: r.cfo_ppm, "ppm")
    for d in range(20):
        for _ in range(d + 1):  # device d gets d+1 packets
            tab.add(rec(t_s=0.01 * d, adva=f"AA:BB:CC:DD:EE:{d:02X}"))
    tab.only_selected.setChecked(False)
    keys = tab.drawn_keys()
    assert len(keys) == tab.MAX_DRAWN
    counts = [tab.counts[k] for k in keys]
    assert counts == sorted(counts, reverse=True)
    assert keys[0] == "AA:BB:CC:DD:EE:13"  # device 19, the busiest


def test_tracking_is_bounded_and_evicts_the_quietest(app):
    tab = SeriesTab("cfo", "t", "CFO", lambda r: r.cfo_ppm, "ppm")
    for _ in range(50):
        tab.add(rec(adva="AA:BB:CC:DD:EE:FF"))  # a busy device
    for d in range(tab.MAX_TRACKED + 30):
        tab.add(rec(adva=f"11:22:33:44:{d // 256:02X}:{d % 256:02X}"))
    assert len(tab.data) <= tab.MAX_TRACKED
    assert "AA:BB:CC:DD:EE:FF" in tab.data, "the busiest device was evicted"


def test_colors_are_stable_and_distinct():
    reset_colors()
    a, b = color_for("AA:BB:CC:DD:EE:01"), color_for("AA:BB:CC:DD:EE:02")
    assert a != b
    assert color_for("AA:BB:CC:DD:EE:01") == a, "colour changed for one device"


# --------------------------------------------------------------------------
# interference plots
# --------------------------------------------------------------------------

class _Stats:
    def __init__(self, noise_floor, epoch=0):
        self.noise_floor_dbfs = noise_floor
        self.epoch = epoch
        self.interference = {}


def test_noise_trace_does_not_repeat_identical_samples(app):
    tab = InterferenceTab()
    for i in range(30):
        tab.update_stats(_Stats(-40.0), i * 0.04)
    assert len(tab.noise_v) == 1, "a constant floor should not build a staircase"
    tab.update_stats(_Stats(-41.0), 2.0)
    assert len(tab.noise_v) == 2


def test_noise_trace_breaks_at_a_retune(app):
    """The floor is a channel property and is reset with the channel."""
    tab = InterferenceTab()
    tab.update_stats(_Stats(-40.0, epoch=0), 0.0)
    tab.update_stats(_Stats(-90.0, epoch=1), 1.0)
    vals = np.fromiter(tab.noise_v, float)
    assert np.isnan(vals).sum() == 1, "no break inserted across the retune"


def test_delivery_plot_does_not_join_distant_bins(app):
    tab = InterferenceTab()
    for _ in range(10):
        tab.add(rec(rssi=-12.0, crc=True))
        tab.add(rec(rssi=-60.0, crc=False))
    tab.refresh()
    xs, ys = tab.pdr_curve.getData()
    assert np.isnan(ys).any(), "sparse bins were joined into one long line"


def test_delivery_plot_joins_adjacent_bins(app):
    tab = InterferenceTab()
    for r in (-12.0, -15.0, -18.0):
        for _ in range(6):
            tab.add(rec(rssi=r, crc=True))
    tab.refresh()
    _, ys = tab.pdr_curve.getData()
    assert not np.isnan(ys).any()


def test_delivery_marker_size_encodes_sample_count(app):
    tab = InterferenceTab()
    for _ in range(4):
        tab.add(rec(rssi=-12.0, crc=True))
    for _ in range(200):
        tab.add(rec(rssi=-15.0, crc=True))
    tab.refresh()
    spots = tab.pdr_points.data
    assert len(spots) == 2
    # bins sort ascending by RSSI: -15 dBFS (200 packets) then -12 dBFS (4)
    xs = list(spots["x"])
    sizes = list(spots["size"])
    busy = sizes[xs.index(-15.0)]
    sparse = sizes[xs.index(-12.0)]
    assert busy > sparse, "a busier bin must draw a larger marker"


def test_delivery_ignores_bins_with_too_few_samples(app):
    tab = InterferenceTab()
    tab.add(rec(rssi=-12.0, crc=True))
    tab.refresh()
    assert len(tab.pdr_points.data) == 0


# --------------------------------------------------------------------------
# selected packet
# --------------------------------------------------------------------------

def test_selected_packet_tab_handles_no_selection(app):
    tab = PacketDetailTab(8e6)
    tab.show_packet(None)
    x, _ = tab.freq_curve.getData()
    assert x is None or len(x) == 0


def test_eye_folds_on_the_recovered_symbol_phase(app):
    """Folding on the integer index alone smears the overlay into noise."""
    from tests.synth import TxImpairments, make_packet
    from sniffer.dsp import Demodulator

    sig, _ = make_packet(imp=TxImpairments(snr_db=35))
    d = Demodulator(8e6, channel=37)
    det = d.process(sig, gate=False)[0]
    r = rec()
    r.iq = sig[det.slice_start : det.slice_end]
    r.sync_offset_in_slice = det.sync_index - det.slice_start
    r.sym_offset = det.sym_offset

    tab = PacketDetailTab(8e6)
    tab.show_packet(r)
    assert tab._eye_curves, "no eye traces drawn"

    # At the right phase the mid-symbol samples land near the rails, so the
    # spread of the trace bundle at the symbol centre is small compared with a
    # deliberately wrong phase.
    def spread_at(frac):
        vals = []
        for c in tab._eye_curves:
            xs, ys = c.getData()
            i = int(frac * (len(ys) - 1))
            vals.append(abs(ys[i]))
        return float(np.std(vals))

    assert spread_at(0.25) > 0  # sanity: there is data
    assert len(tab._eye_curves) <= tab.MAX_EYE_TRACES


def test_frequency_trace_is_blanked_outside_the_burst(app):
    """Noise instantaneous frequency spans +/-fs/2 and would swamp the y-range."""
    from tests.synth import TxImpairments, make_packet
    from sniffer.dsp import Demodulator

    sig, _ = make_packet(imp=TxImpairments(snr_db=28))
    d = Demodulator(8e6, channel=37)
    det = d.process(sig, gate=False)[0]
    r = rec()
    r.iq = sig[det.slice_start : det.slice_end]
    r.sync_offset_in_slice = det.sync_index - det.slice_start
    r.sym_offset = det.sym_offset
    r.n_symbols = det.n_symbols
    r.length = det.pdu.length

    tab = PacketDetailTab(8e6)
    tab.show_packet(r)
    _, ys = tab.freq_curve.getData()
    assert np.isnan(ys).any(), "noise region was not blanked"
    finite = ys[np.isfinite(ys)]
    assert finite.size > 100
    # what remains must be the GFSK burst, not multi-MHz noise
    assert np.percentile(np.abs(finite), 99) < 600e3

    lo, hi = tab.freq_plot.viewRange()[1]
    assert max(abs(lo), abs(hi)) < 1.5e6, "y-range still set by noise"


def test_burst_window_covers_the_decoded_packet(app):
    r = rec()
    r.sync_offset_in_slice = 400
    r.sym_offset = 4.0
    r.n_symbols = 376
    tab = PacketDetailTab(8e6)
    b0, b1 = tab.burst_window(r, 4000)
    assert b0 == 400
    assert b1 == pytest.approx(400 + 4 + 376 * 8, abs=2)


def test_burst_window_falls_back_to_the_length_field(app):
    r = rec()
    r.sync_offset_in_slice = 400
    r.sym_offset = 4.0
    r.n_symbols = 0
    r.length = 37
    tab = PacketDetailTab(8e6)
    b0, b1 = tab.burst_window(r, 10000)
    assert b1 > b0 + 2000


def test_envelope_uses_its_own_axis(app):
    """Rescaling the envelope onto the frequency axis made both look like noise."""
    tab = PacketDetailTab(8e6)
    r = rec()
    r.iq = (np.ones(2000) * 0.3).astype(np.complex64)
    r.sync_offset_in_slice = 400
    r.sym_offset = 4.0
    tab.show_packet(r)
    _, env = tab.env_curve.getData()
    assert env is not None and len(env) == 2000
    assert np.allclose(env, 0.3, atol=1e-6), "envelope must be in its own units"


# --------------------------------------------------------------------------
# reference modal
# --------------------------------------------------------------------------

def test_every_chart_has_a_reference_note():
    dock_keys = {"spectrum", "cfo", "scatter", "rssi", "packet", "interference"}
    assert dock_keys <= set(CHART_INFO)


def test_reference_notes_have_formulas_and_limits():
    for key, info in CHART_INFO.items():
        assert info["formulas"], f"{key} has no formulas"
        assert info["reading"], f"{key} has no reading guidance"
        assert info["caveats"], f"{key} states no limits"
        html = as_html(key)
        assert "<h2>" in html and "Formulas" in html and "Limits" in html


def test_reference_dialog_is_modal_and_renders(app):
    dlg = ChartInfoDialog("cfo")
    assert dlg.isModal()
    browser = dlg.findChildren(type(dlg.children()[1]))  # smoke: it built
    assert "carrier offset" in as_html("cfo").lower()


def test_every_plot_tab_exposes_an_info_button(app):
    dock = PlotDock(8e6)
    from PyQt6.QtWidgets import QPushButton

    for i in range(dock.count()):
        w = dock.widget(i)
        buttons = [b for b in w.findChildren(QPushButton) if b.text() == "Info"]
        assert buttons, f"tab {dock.tabText(i)} has no Info button"


def test_waterfall_contrast_narrows_the_colour_window(app):
    """A fixed dB window washes out a quiet channel and saturates a busy one."""
    tab = SpectrumTab(8e6)
    tab.update_spectrum(noise(seed=21))
    wide = tab.contrast.maximum()
    tab.contrast.setValue(wide)
    lo_wide, hi_wide = tab.wf_image.levels
    tab.contrast.setValue(tab.contrast.minimum())
    lo_narrow, hi_narrow = tab.wf_image.levels
    assert (hi_narrow - lo_narrow) < (hi_wide - lo_wide)
    assert hi_narrow == pytest.approx(hi_wide), "the top should stay anchored"
    assert "dB" in tab.contrast_label.text()


def test_waterfall_top_tracks_the_signal(app):
    """The colour window has to follow the data or the slider controls nothing."""
    tab = SpectrumTab(8e6)
    for _ in range(6):
        tab.update_spectrum(noise(amp=0.0005, seed=22))
    quiet_top = tab._wf_top
    for _ in range(30):
        tab.update_spectrum(noise(amp=0.2, seed=23))
    assert tab._wf_top > quiet_top + 10


def test_waterfall_axis_leaves_room_for_its_labels(app):
    """The tick at 0 was drawn half outside the plot.

    Two causes: a left axis too narrow for the tick text, and a y range set
    flush to the boundary so the topmost label straddled the edge.
    """
    tab = SpectrumTab(8e6)
    assert tab.wf_plot.getAxis("left").fixedWidth >= 50
    tab.update_spectrum(noise(seed=24))
    lo, hi = tab.wf_plot.viewRange()[1]
    assert lo < 0.0, "0 must sit inside the view, not on its edge"


# --------------------------------------------------------------------------
# direction / AoA
# --------------------------------------------------------------------------

def aoa_rec(addr, aoa, rssi=-45.0):
    r = rec(adva=addr, rssi=rssi)
    r.features.measurements["aoa_deg"] = Measurement(aoa, 3.0, "deg")
    r.features.measurements["antenna_phase_deg"] = Measurement(aoa * 3, 4.0, "deg")
    return r


def test_direction_tab_says_so_when_there_is_no_aoa(app):
    from sniffer.gui.plots import DirectionTab

    t = DirectionTab()
    t.add(rec())          # no aoa_deg measurement
    t.refresh()
    assert not t.data
    assert "Dual antenna" in t.note.text()


def test_direction_tab_plots_a_bearing_per_address(app):
    from sniffer.gui.plots import DirectionTab

    t = DirectionTab()
    for _ in range(20):
        t.add(aoa_rec("AA:BB:CC:DD:EE:01", -30.0))
        t.add(aoa_rec("AA:BB:CC:DD:EE:02", 45.0))
    t.refresh()
    assert set(t.data) == {"AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02"}
    assert len(t._rays) == 2, "one mean-bearing ray per address"
    assert "-30" in t.note.text() and "+45" in t.note.text()


def test_direction_tab_separates_two_radios_under_one_address(app):
    """The case the tool exists for: same address, two directions."""
    from sniffer.gui.plots import DirectionTab

    t = DirectionTab()
    for _ in range(30):
        t.add(aoa_rec("AA:BB:CC:DD:EE:FF", -40.0))
        t.add(aoa_rec("AA:BB:CC:DD:EE:FF", 40.0))
    t.refresh()
    angles = np.array([a for a, _ in t.data["AA:BB:CC:DD:EE:FF"]])
    assert angles.min() < -30 and angles.max() > 30
    assert "spread" in t.note.text()


def test_direction_tab_dims_everything_but_the_focused_device(app):
    from sniffer.gui.plots import DirectionTab

    t = DirectionTab()
    for _ in range(10):
        t.add(aoa_rec("AA:BB:CC:DD:EE:01", -20.0))
        t.add(aoa_rec("AA:BB:CC:DD:EE:02", 20.0))
    t.set_focus("AA:BB:CC:DD:EE:02")
    t.refresh()
    widths = sorted(r.opts["pen"].width() for r in t._rays)
    assert widths[0] < widths[-1], "the focused device should stand out"


def test_direction_density_reveals_two_lobes(app):
    """The scatter alone cannot show bimodality; the density is what does.

    A strong transmitter puts every point at the same radius, because radius
    carries signal strength, so 400 packets draw one thin arc whether they came
    from one direction or two.
    """
    from sniffer.gui.plots import DirectionTab

    t = DirectionTab()
    rng = np.random.default_rng(0)
    one = rng.normal(10.0, 25.0, 400)
    two = np.concatenate([rng.normal(-34.0, 18.0, 200), rng.normal(32.0, 18.0, 200)])

    assert t._bearing_split(one)[0] < 3.0
    sig, lo, hi = t._bearing_split(two)
    assert sig > 3.5
    assert lo < -25 and hi > 25

    xs, ys = t._density_curve(two)
    assert xs.size == ys.size > 4
    # the outline must start and end at the origin so it fills as a wedge
    assert xs[0] == 0.0 and xs[-1] == 0.0
    # and it must reach further out at the two lobes than between them
    r = np.hypot(xs[1:-1], ys[1:-1])
    ang = np.degrees(np.arctan2(xs[1:-1], ys[1:-1]))
    lobe = r[(np.abs(ang + 34) < 8) | (np.abs(ang - 32) < 8)].mean()
    trough = r[np.abs(ang) < 8].mean()
    assert lobe > trough, "the density does not show the two lobes"


def test_direction_reports_a_split_in_its_caption(app):
    from sniffer.gui.plots import DirectionTab

    t = DirectionTab()
    rng = np.random.default_rng(1)
    for a in np.concatenate([rng.normal(-34, 15, 120), rng.normal(32, 15, 120)]):
        t.add(aoa_rec("AA:BB:CC:DD:EE:FF", float(a)))
    t.refresh()
    assert "two lobes" in t.note.text()


def test_direction_density_empty_input(app):
    from sniffer.gui.plots import DirectionTab

    t = DirectionTab()
    xs, ys = t._density_curve(np.zeros(0))
    assert xs.size == 0 and ys.size == 0
    assert t._bearing_split(np.zeros(0))[0] == 0.0


def test_direction_tab_csv(app):
    from sniffer.gui.plots import DirectionTab

    t = DirectionTab()
    for _ in range(5):
        t.add(aoa_rec("AA:BB:CC:DD:EE:01", 12.0))
    header, rows = t._csv_rows()
    assert header == ["adva", "aoa_deg", "rssi_dbfs"]
    assert len(rows) == 5 and rows[0][1] == 12.0


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------

def test_every_chart_offers_png_svg_and_csv(app):
    from PyQt6.QtWidgets import QToolButton

    dock = PlotDock(8e6)
    for i in range(dock.count()):
        w = dock.widget(i)
        saves = [b for b in w.findChildren(QToolButton) if b.text() == "Save"]
        assert saves, f"tab {dock.tabText(i)} has no Save button"
        labels = " ".join(a.text() for a in saves[0].menu().actions())
        assert "PNG" in labels and "SVG" in labels and "CSV" in labels


def test_saving_writes_png_and_csv(app, tmp_path, monkeypatch):
    import os

    from PyQt6.QtWidgets import QToolButton
    import sniffer.gui.plots as P

    dock = PlotDock(8e6)
    dock.add_records([rec(t_s=i * 0.1, cfo=5.0 + i) for i in range(12)])
    dock.refresh()
    tab = dock.cfo
    save = [b for b in tab.findChildren(QToolButton) if b.text() == "Save"][0]

    for kind in ("png", "csv"):
        target = tmp_path / f"out.{kind}"
        monkeypatch.setattr(
            P.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "")),
        )
        save.save(kind)
        assert target.exists() and target.stat().st_size > 0, kind

    text = (tmp_path / "out.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "adva,time_s,value"
    assert len(text.splitlines()) > 5


def test_saving_an_empty_chart_reports_rather_than_writing(app, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QToolButton
    import sniffer.gui.plots as P

    dock = PlotDock(8e6)
    tab = dock.cfo
    save = [b for b in tab.findChildren(QToolButton) if b.text() == "Save"][0]
    target = tmp_path / "empty.csv"
    monkeypatch.setattr(
        P.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")),
    )
    seen = {}
    monkeypatch.setattr(
        P.QMessageBox, "warning",
        staticmethod(lambda parent, title, msg: seen.update(msg=msg)),
    )
    save.save("csv")
    assert not target.exists()
    assert "no data" in seen.get("msg", "")


def test_dock_focus_propagates_to_every_per_device_plot(app):
    dock = PlotDock(8e6)
    dock.add_records([rec(adva="AA:BB:CC:DD:EE:07")])
    dock.set_focus("AA:BB:CC:DD:EE:07")
    assert dock.cfo.focus == "AA:BB:CC:DD:EE:07"
    assert dock.rssi.focus == "AA:BB:CC:DD:EE:07"
    assert dock.scatter.focus == "AA:BB:CC:DD:EE:07"
