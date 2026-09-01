# BLE Single-Channel Sniffer with Live RF-Fingerprint GUI

A single-channel Bluetooth Low Energy advertising sniffer for the **Nuand
bladeRF 2.0 micro**. It tunes to one BLE channel, decodes advertising PDUs in
real time, extracts physical-layer features from every packet, and shows them in
a Wireshark-style GUI: scrolling packet list, per-packet detail tree, hex dump,
plus live spectrum, feature, and angle-of-arrival plots.

The purpose is **defensive**: identify transmitters that share an advertised
address but not a radio (address-spoofing / beacon cloning), and characterise
interference on the selected channel. **Receive only** — no transmit path exists
in the software.

> The carrier-frequency-offset fingerprint used here separates a replayed
> (spoofed) advertising packet from the genuine device whose address it wears by
> tens of ppm, with an empty gap between the two crystal modes. See the
> accompanying paper (`paper/`) for a measured evaluation against an nRF52840.

---

## 1. Hardware requirements

| Item | Requirement | Link |
|---|---|---|
| **SDR** | Nuand bladeRF 2.0 micro (xA4 or xA9). Verified on **xA4**, FPGA 0.14.0, firmware 2.4.0, libbladeRF 2.4.1 | <https://www.nuand.com/bladerf-2-0-micro/> |
| **USB** | USB 3.0 SuperSpeed port (required to sustain 8 MSPS; USB 2.0 will drop samples) | — |
| **Antenna** | One 2.4 GHz antenna on **RX1** (SMA). A second on **RX2** enables angle-of-arrival | e.g. <https://www.nuand.com/product/2-4ghz-antenna/> |
| **Host** | A mid-range laptop/desktop. A hybrid Intel CPU (P+E cores) is handled automatically by pinning the capture and DSP stages to performance cores. Verified on an i7-1255U | — |
| **Optional: 10 MHz reference** | A disciplined GPSDO into the **U.FL clock input** makes carrier-offset features absolute rather than session-relative | e.g. <https://www.leobodnar.com/shop/index.php?main_page=product_info&products_id=234> |
| **Optional: reference beacon** | A Nordic **nRF52840** (e.g. Adafruit Feather) to generate known test traffic for validation | <https://www.nordicsemi.com/Products/nRF52840> |

The bladeRF's own USB drivers and the `bladeRF.dll` / `libbladeRF.so` shared
library must be installed (the standard [bladeRF
installer](https://github.com/Nuand/bladeRF/wiki/Getting-Started%3A-Windows) on
Windows, or your distribution's `libbladerf` package on Linux). No Python
bladeRF binding is needed — the app binds the shared library directly with
`ctypes` (`sniffer/libbladerf.py`), so it works on current CPython without a
build step. Set `BLADERF_LIBRARY` if the library is not on the default path.

---

## 2. Python environment and running

Python **3.11+** (developed and tested on **3.14**). Install the dependencies:

```bash
pip install -r requirements.txt
#   numpy scipy PyQt6 pyqtgraph pyarrow numba pytest
```

### Run the GUI

```bash
python main.py                       # channel 37 (2402 MHz), default
python main.py --channel 38          # 2426 MHz
python main.py --freq 2426e6         # direct LO override
python main.py --gain 45             # manual RX gain, dB
python main.py --enroll AA:BB:CC:DD:EE:FF   # collect a baseline for one device
python main.py --external-clock      # discipline to a 10 MHz U.FL reference
```

Data channels require the access address (and, for a live connection, the CRC
init) explicitly, because the advertising defaults will not decode them:

```bash
python main.py --channel 5 --access-address 0x9E8B4B27 --crc-init 0x555555
```

### Headless capture (no GUI) — writes CSV / Parquet / PCAP / JSON

```bash
python main.py --headless --seconds 60 --channel 37 --out capture
```

The PCAP uses `LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR`, so it opens in Wireshark
next to this tool.

### Angle of arrival (two antennas)

```bash
python main.py --dual-antenna        # RX2 as a second coherent antenna
```

Both RX channels share one LO, so this is a **spatial** measurement at one
frequency — it does not watch two BLE channels at once. Calibrate the array's
fixed phase offset from a known source (the **Calibrate** button) before reading
bearings.

### Self-test (no radio needed)

```bash
python main.py --self-test           # or: python -m pytest -q
```

The suite (200+ tests) validates every estimator against a synthetic GFSK
generator with known injected impairments, plus the DSP, analysis, export, and
GUI logic — none of it needs the hardware.

### Offline three-channel join

```bash
python main.py --join s37.parquet s38.parquet s39.parquet
```

Reconstructs cross-channel RSSI ratios from three sequential single-channel
sessions (with an explicit non-simultaneity warning).

---

## 3. Screenshots

See [`screenshots/`](screenshots/) (`spectrum.png` and `interference.png` are
included; `window.png` and `aoa.png` are produced by
`python experiment/make_screenshots.py --dual` with the radio connected). The
GUI is a dark, Wireshark-style layout:
a live packet table (colour-coded by CRC and by spoofing alerts), an expandable
per-packet detail tree showing every feature with its spec limit and baseline
sigma, a hex dump, and a tabbed plot dock (spectrum/waterfall, CFO-vs-time,
feature scatter, RSSI, per-packet eye diagram, angle-of-arrival, interference).

| | |
|---|---|
| `window.png` | full application during a live capture |
| `spectrum.png` | spectrum + waterfall of channel 37 with the BLE channel mask |
| `aoa.png` | angle-of-arrival polar plot (dual-antenna) |
| `interference.png` | interference monitor (noise floor, PDR vs RSSI, classifier) |

---

## 4. How it works (calculation)

Everything is vectorised: Python only ever touches whole arrays or short
per-packet byte slices, never per-sample loops at 8 MSPS. The stages run in
three processes (capture / DSP / feature+analysis) decoupled by bounded queues
and a lock-free shared-memory IQ ring.

**Instantaneous frequency** of the GFSK signal, from the argument of the
conjugate product:

```
f[n] = (fs / 2π) · arg( x[n+1] · conj(x[n]) )
```

**Packet detection**: correlate the instantaneous frequency against the known
preamble + access-address template (FFT overlap-save), recover symbol timing,
de-whiten with the channel-seeded 7-bit LFSR (`x⁷+x⁴+1`), and check the BLE
**CRC-24** (`x²⁴+x¹⁰+x⁹+x⁶+x⁴+x³+x+1`, init `0x555555`). Bluetooth Core
Specification, Vol. 6, Part B.

**Receiver calibration** (removed before any transmitter feature, so receiver
impairments are never attributed to the device) — DC offset plus Gram–Schmidt
quadrature correction:

```
θ = E[iq] / E[i²]          (quadrature skew)
q' = (q − θ·i) · sqrt( E[i²] / E[(q − θ·i)²] )   (gain-imbalance corrected)
```

**Carrier frequency offset** — the key fingerprint — as the mean residual after
subtracting the ideal per-symbol deviation implied by the decoded bits:

```
Δf   = (1/K) Σ ( f_sym[k] − d·(2·b[k] − 1) )        [Hz]
σ_Δf = std(residual) / sqrt(K)                       (< 2 ppm typical)
Δf_ppm = 10⁶ · Δf / f_carrier
```

**Spoofer detection** — two transmitters under one address form two CFO modes;
the separation in pooled standard deviations is the alarm score:

```
S = |μ_spoof − μ_genuine| / sqrt( (σ_spoof² + σ_genuine²) / 2 )
```

A per-feature bimodality test (`analysis.one_dimensional_split`) scans every
feature for two populations, which is what catches a second radio that a
full-vector distance would dilute.

**Angle of arrival** (dual-antenna), from the inter-antenna phase difference
after removing the array's calibrated offset:

```
Δφ = arg( Σ x0[n]·conj(x1[n]) ) − φ_cal
θ  = arcsin( Δφ / (2π · d/λ) )          (d = λ/2 assumed; half-plane only)
```

### Background reading

- V. Brik et al., "Wireless device identification with radiometric signatures,"
  *ACM MobiCom*, 2008 — RF fingerprinting foundations.
- H. Givehchian et al., "Evaluating Physical-Layer BLE Location Tracking Attacks
  on Mobile Devices," *IEEE S&P*, 2022 — BLE CFO fingerprint stability.
- J.-L. Wu et al., "BlueShield: Detecting Spoofing Attacks in BLE Networks,"
  *RAID*, 2020 — cyber-physical anomaly detection for BLE.
- Bluetooth SIG, *Bluetooth Core Specification* v5.4, Vol. 6 Part B (Link Layer)
  — whitening, CRC-24, advertising PDUs.

A fuller feature list and the calibration procedure are in
[`docs/ALGORITHMS.md`](docs/ALGORITHMS.md); a table of which features remain
valid without a GPSDO is in the code (`sniffer/features.py`) and the paper.

### Reproducing the paper

```bash
python experiment/run_experiment.py all     # calibrate, scan, replay (needs the nRF52840)
python experiment/make_figures.py           # white-background result figures
python experiment/make_scan_figures.py      # spectrum + interference figures
cd experiment/paper && pdflatex spoofing.tex
```

---

## Layout

```
sniffer/          the library
  channels.py     channel map, whitening LFSR, CRC-24, ChannelPlan consistency
  libbladerf.py   ctypes binding to libbladeRF (receive-side entry points only)
  radio.py        device setup, calibration, clocking, capture
  shmring.py      lock-free shared-memory IQ ring
  dsp.py          gate, correlation, demod, de-whitening, CRC, PDU parsing
  features.py     per-packet feature extraction, one function per feature
  analysis.py     clustering, baselines, anomaly scoring, interference monitor
  calibration.py  receiver + antenna-array calibration and history
  packet.py       the packet record
  pipeline.py     the three processes, backpressure and stats
  export.py       SigMF, PCAP, CSV/Parquet, offline multi-channel join
  gui/            PyQt6 app: window, table model, plots, filter parser, dialogs
tests/            synthetic-signal test suite (200+ tests, no hardware needed)
main.py           CLI entry point
docs/             algorithm notes
screenshots/      GUI screenshots used by this README
experiment/       reproducibility scripts + the IEEE paper
  run_experiment.py, make_figures.py, make_scan_figures.py, make_screenshots.py
  paper/          IEEE paper source (spoofing.tex) and built PDF
  figures/        the figures the paper embeds
```

## What to upload to GitHub

This folder **is** the upload set — every file in it is source, documentation, or
a paper figure, and all of it is meant to be committed. Concretely, upload:

- `sniffer/`, `tests/`, `main.py` — the tool and its test suite (source);
- `README.md`, `LICENSE`, `requirements.txt`, `pytest.ini`, `.gitignore`, `docs/`;
- `screenshots/` — the four PNGs this README links;
- `experiment/` — the reproducibility scripts, `paper/spoofing.tex`,
  `paper/spoofing.pdf`, and `figures/` (the plots the paper embeds).

**Do not upload experimental result data — it is the tool's *output*, not
source, and is regenerated by `experiment/run_experiment.py`.** The `.gitignore`
already blocks all of it, so a normal `git add .` is safe; it excludes:

- `experiment_results.json`, `experiment_results_summary.json` — measured run data;
- `captures/`, `*.pcap`, `*.sigmf-*`, `*.parquet`, `*.csv`, `capture.json` — capture exports;
- `calibration_history.json` — per-device calibration logs;
- `__pycache__/`, `.pytest_cache/`, and LaTeX build products (`*.aux`, `*.log`, …).

The `figures/` PNG/PDFs *are* kept: they are the published paper's figures, not
raw data, and are needed to rebuild `spoofing.pdf`. If you would rather ship the
paper as the PDF only, delete `experiment/figures/` and `experiment/paper/*.tex`
and keep `spoofing.pdf`.

## Non-goals and ethics

No transmission. No interference generation over the air. No decryption of
connection traffic. No defeat of address randomisation beyond observing that
physical-layer features persist across address rotation. Test only against
devices you own or are authorised to test.

## License

MIT — see [`LICENSE`](LICENSE).
