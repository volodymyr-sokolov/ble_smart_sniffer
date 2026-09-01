"""Reference notes for each chart, shown in a modal dialog.

Kept as data rather than as docstrings so the same text can be shown in the GUI,
checked by a test, and edited without touching plotting code.  Each entry says
what is plotted, the formula behind it, how to read it, and -- importantly --
what it cannot tell you.  A fingerprinting tool that only documents what a plot
shows invites the operator to over-read it.
"""

from __future__ import annotations

CHART_INFO: dict[str, dict] = {
    "spectrum": {
        "title": "Spectrum and waterfall",
        "what": (
            "Power spectral density of the current sample block, and its history "
            "stacked in time. The green bands are the BLE channel plan at the "
            "tuned frequency, so energy that falls outside them is not BLE."
        ),
        "formulas": [
            ("Spectrum", "P(f) = 10·log₁₀( |FFT(w·x)|² / N² )  [dBFS]"),
            ("Window", "w = Hann, N = 1024 samples"),
            ("Full scale", "|s| = 1 corresponds to 2048 in the 12-bit SC16_Q11 ADC word"),
            ("Peak hold", "P̂(f) ← max( P̂(f) − δ , P(f) ),  δ = 0.35 dB per frame"),
            ("Channel centre", "f = 2402 + 2k MHz (k = 0…39 in channel-index order)"),
        ],
        "reading": [
            "The blue trace is instantaneous; the orange trace is a decaying peak "
            "hold, so brief bursts stay visible for a couple of seconds.",
            "A BLE packet is about 1 MHz wide and sits inside one green band. "
            "Energy spanning several bands is Wi-Fi, a microwave oven, or a jammer.",
            "The waterfall's brightness is the same dB scale; a vertical stripe is "
            "a constant carrier, a horizontal streak is a burst.",
        ],
        "caveats": [
            "dBFS is receiver-referred. It depends on the RX gain setting and is "
            "not a transmitter power.",
            "Only ±(sample rate)/2 around the tuned LO is visible. Interference "
            "outside that window is invisible, not absent.",
        ],
    },
    "cfo": {
        "title": "Carrier frequency offset vs time",
        "what": (
            "Per-packet carrier offset, one series per advertising address. This "
            "is the plot where a spoofer usually becomes visible: two radios "
            "under one address sit at two different offsets."
        ),
        "formulas": [
            ("Instantaneous frequency", "f[n] = (f_s / 2π) · arg( x[n+1] · x*[n] )"),
            ("Residual", "r[k] = f_sym[k] − d·(2·b[k] − 1)   (d = measured deviation)"),
            ("Carrier offset", "Δf = mean(r)   [Hz]"),
            ("In ppm", "Δf_ppm = Δf / f_carrier × 10⁶"),
            ("Uncertainty", "σ(Δf) = std(r) / √K over K symbols"),
        ],
        "reading": [
            "A healthy transmitter holds a nearly constant offset; the spread "
            "within one device is the measurement noise plus its own drift.",
            "Two clearly separated horizontal bands under one address mean two "
            "transmitters. That is what the red row colouring flags.",
            "Select a packet in the list to show only that device.",
        ],
        "caveats": [
            "Without a locked 10 MHz reference this includes the receiver's own "
            "VCTCXO drift — about ±40 ppm over room temperature, which is larger "
            "than the between-device spread. The status bar shows 'Ref: "
            "internal' and the column header carries an asterisk.",
            "The spec limit is ±150 kHz (±62.4 ppm at 2.402 GHz).",
        ],
    },
    "scatter": {
        "title": "Feature scatter",
        "what": (
            "Any two physical-layer features plotted against each other, coloured "
            "by cluster or by address, with one- and two-sigma ellipses for each "
            "enrolled baseline."
        ),
        "formulas": [
            ("Whitening", "z_i = (v_i − μ_i) / σ_i  over the session population"),
            ("Cluster distance", "d = ‖z − c‖ / √D   (D = feature dimensions)"),
            ("Separation", "S = ‖c₁ − c₂‖ / √( r₁² + r₂² )   (r = RMS cluster radius)"),
            ("Anomaly score", "A = √( (1/M) · Σ_i ((v_i − μ_i)/σ_i)² )  over M valid features"),
            ("Baseline ellipse", "(x−μ_x)²/σ_x² + (y−μ_y)²/σ_y² = k²,  k = 1, 2"),
        ],
        "reading": [
            "One tight blob per physical radio is the expected picture.",
            "Two blobs under one address is the signature this tool exists to find.",
            "Points far outside an enrolled ellipse drive the anomaly score; the "
            "detail tree lists which feature contributed most.",
        ],
        "caveats": [
            "Distances are relative to the session's own spread, so the axes are "
            "only comparable within one capture.",
            "Deviation asymmetry is deliberately absent from the feature vector: "
            "it is exactly degenerate with carrier offset.",
        ],
    },
    "rssi": {
        "title": "RSSI vs time",
        "what": "Received burst power per packet, one series per address.",
        "formulas": [
            ("Burst power", "P = mean( |x[n]|² ) over the packet"),
            ("In dBFS", "RSSI = 10·log₁₀(P)"),
            ("In dBm", "RSSI_dBm = RSSI_dBFS − G_rx + C   (C from a calibration table)"),
            ("SNR", "SNR = 10·log₁₀( (P_burst − P_noise) / P_noise )"),
        ],
        "reading": [
            "A stationary device gives a nearly flat line; movement gives slow "
            "drift; a step change with no other feature change is usually the "
            "antenna orientation.",
            "A step in RSSI together with a step in the multipath profile, for a "
            "device that should be stationary, is worth investigating.",
        ],
        "caveats": [
            "Shown in dBFS unless an RSSI calibration offset was supplied "
            "(--rssi-cal); the packet list's column header states which.",
            "dBFS depends on the RX gain. Changing gain mid-capture shifts every "
            "series at once.",
        ],
    },
    "packet": {
        "title": "Selected packet: frequency trace and eye",
        "what": (
            "The instantaneous frequency of the selected packet with its envelope, "
            "and its eye diagram folded at the recovered symbol phase. The shaded "
            "band is the PA turn-on window."
        ),
        "formulas": [
            ("Instantaneous frequency", "f[n] = (f_s / 2π) · arg( x[n+1] · x*[n] )"),
            ("Envelope", "a[n] = |x[n]|  (right axis, normalised to full scale)"),
            ("Modulation index", "h = 2·d / R_s,  d = ½(d_one + d_zero), R_s = 1 Msym/s"),
            ("Eye fold", "segments of 2 symbols starting at t₀ + k·T_s + φ"),
            ("PA rise time", "t₁₀₋₉₀ of a[n] across the turn-on edge"),
        ],
        "reading": [
            "Clean GFSK sits at ±250 kHz for a modulation index of 0.5; the eye "
            "should show two well-separated rails with a visible opening.",
            "The shaded window is where the PA ramp lives — its shape is one of "
            "the more device-specific features.",
            "A closed or smeared eye means low SNR or a collision; features from "
            "such a packet carry correspondingly large uncertainties.",
        ],
        "caveats": [
            "Drawn from the retained slice through the 2.5 MHz measurement filter, "
            "not the narrower filter used for bit decisions.",
            "A CRC-failed packet is still drawn, because its trace is often the "
            "most informative thing about why it failed.",
        ],
    },
    "direction": {
        "title": "Direction (angle of arrival)",
        "what": (
            "Bearing of each advertising address, from the phase difference "
            "between the two RX antennas. Requires --dual-antenna. This is "
            "where an impersonator and the device it copies separate: they "
            "share an address, so they are one row in the table, but they are "
            "rarely in the same direction."
        ),
        "formulas": [
            ("Phase difference", "dphi = arg( mean( x0 . conj(x1) ) ) - dphi_cal"),
            ("Bearing", "theta = asin( dphi / (2.pi.d/lambda) )"),
            ("Element spacing", "d = lambda/2 assumed, so theta = asin(dphi/pi)"),
            ("Radius on the plot", "signal strength, clipped to -80..0 dBFS"),
            ("Mean bearing", "circular mean: atan2( mean sin, mean cos )"),
        ],
        "reading": [
            "0 degrees is broadside to the array, +/-90 is along its axis.",
            "A stationary transmitter holds a bearing; the scatter around the "
            "ray is the measurement noise plus multipath.",
            "Two rays under one address means two transmitters in two places.",
        ],
        "caveats": [
            "A two-element array cannot distinguish +theta from -theta: the "
            "phase difference is symmetric about the array axis. The plot is a "
            "half plane for that reason.",
            "Meaningless until the fixed per-channel phase offset between the "
            "two RX chains has been calibrated out; they share an LO but not a "
            "signal path, and the offset is tens of degrees.",
            "Multipath biases the bearing indoors, sometimes badly.",
        ],
    },
    "interference": {
        "title": "Interference monitor",
        "what": (
            "Channel noise floor over time, and packet delivery ratio against "
            "received power. Runs independently of decoding — the point is to "
            "describe a channel on which decoding is failing."
        ),
        "formulas": [
            ("Noise floor", "N = 10·log₁₀( Q₀.₁₅( boxcar(|x|², 4 µs) ) )  [dBFS]"),
            ("Duty cycle", "fraction of samples with |x|² > 6·N_linear"),
            ("Occupied bandwidth", "99% energy width of the block spectrum"),
            ("Spectral kurtosis", "K = E[(p − μ)⁴] / σ⁴ over the normalised spectrum"),
            ("Delivery ratio", "PDR(r) = N_crc_ok(r) / N_total(r) per RSSI bin r"),
            ("Reaction latency", "median( t_energy_onset − t_preamble_start )  [µs]"),
        ],
        "reading": [
            "Delivery collapsing at HIGH received power indicates interference, "
            "not distance — that is the diagnostic shape to look for here.",
            "Reaction latency of a few microseconds means a reactive emitter: "
            "something is listening for the preamble and then keying.",
            "Low spectral kurtosis with wide occupied bandwidth is noise-like; "
            "high kurtosis in a narrow band is a carrier.",
        ],
        "caveats": [
            "Marker size on the delivery plot is proportional to how many packets "
            "fell in that bin; a bin with three packets means very little.",
            "The noise-floor trace breaks at a retune, because the floor is a "
            "property of the channel and is reset with it.",
        ],
    },
}


def as_html(key: str) -> str:
    """Render one chart's reference note as rich text for the dialog."""
    info = CHART_INFO.get(key)
    if info is None:
        return "<p>No reference available for this chart.</p>"

    def li(items):
        return "".join(f"<li>{x}</li>" for x in items)

    rows = "".join(
        f"<tr><td style='padding:2px 12px 2px 0; white-space:nowrap;'>"
        f"<b>{name}</b></td>"
        f"<td style='padding:2px 0;'><code>{expr}</code></td></tr>"
        for name, expr in info["formulas"]
    )
    return f"""
<h2>{info['title']}</h2>
<p>{info['what']}</p>
<h3>Formulas</h3>
<table cellspacing='0'>{rows}</table>
<h3>How to read it</h3>
<ul>{li(info['reading'])}</ul>
<h3>Limits</h3>
<ul>{li(info['caveats'])}</ul>
"""
