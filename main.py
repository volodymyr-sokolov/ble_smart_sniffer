#!/usr/bin/env python3
"""Single-channel BLE advertising sniffer with live RF-fingerprint GUI.

Receive only.  No transmit path exists anywhere in this application.

    python main.py                         # channel 37, GUI
    python main.py --channel 38
    python main.py --freq 2426e6
    python main.py --channel 5 --access-address 0x9E8B4B27 --crc-init 0x123456
    python main.py --external-clock        # discipline to a 10 MHz reference
    python main.py --enroll AA:BB:CC:DD:EE:FF
    python main.py --headless --seconds 60 --out capture
    python main.py --join s37.parquet s38.parquet s39.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _int_auto(v: str) -> int:
    return int(v, 0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ble-sniffer",
        description="Single-channel BLE advertising sniffer for the bladeRF 2.0 micro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ch = p.add_argument_group("channel")
    ch.add_argument("--channel", type=int, default=None, metavar="N",
                    help="BLE channel index 0-39 (37/38/39 are the advertising "
                         "channels; default 37)")
    ch.add_argument("--freq", type=float, default=None, metavar="HZ",
                    help="direct LO override in Hz, bypassing the channel map")
    ch.add_argument("--access-address", type=_int_auto, default=None, metavar="AA",
                    help="access address; required for data channels")
    ch.add_argument("--crc-init", type=_int_auto, default=None, metavar="INIT",
                    help="CRC init value as the spec states it (advertising: 0x555555)")

    rf = p.add_argument_group("radio")
    rf.add_argument("--sample-rate", type=float, default=8e6, metavar="SPS",
                    help="8e6 default (8 samples/symbol); 16e6 for high-resolution "
                         "fingerprinting; 4e6 works but costs transient accuracy")
    rf.add_argument("--bandwidth", type=float, default=3e6, metavar="HZ")
    rf.add_argument("--gain", type=int, default=45, metavar="DB",
                    help="manual RX gain; aim for a peak 10-15 dB below full scale")
    rf.add_argument("--agc", action="store_true",
                    help="enable AGC (off by default: per-packet gain must be known)")
    rf.add_argument("--dual-antenna", action="store_true",
                    help="enable RX1 as a second coherent antenna for AoA and "
                         "diversity. Both RX channels share one LO, so this does "
                         "NOT watch two BLE channels at once")
    rf.add_argument("--block-size", type=int, default=16384, metavar="N")
    rf.add_argument("--device", default=None, metavar="ID",
                    help="libbladeRF device identifier")
    rf.add_argument("--bias-tee", action="store_true")

    cal = p.add_argument_group("calibration")
    cal.add_argument("--external-clock", action="store_true",
                     help="use the U.FL 10 MHz reference input. Without it every "
                          "ppm-scale feature is marked UNCALIBRATED")
    cal.add_argument("--refclk", type=float, default=10e6, metavar="HZ")
    cal.add_argument("--rssi-cal", type=float, default=None, metavar="DB",
                     help="offset mapping dBFS to dBm; without it RSSI is dBFS only")

    an = p.add_argument_group("analysis")
    an.add_argument("--enroll", action="append", default=[], metavar="ADDR",
                    help="collect a baseline from this address (repeatable)")
    an.add_argument("--ring-seconds", type=float, default=2.0, metavar="S",
                    help="raw IQ history kept for SigMF dumps")

    out = p.add_argument_group("mode")
    out.add_argument("--headless", action="store_true",
                     help="capture without a GUI and write files")
    out.add_argument("--seconds", type=float, default=30.0, metavar="S")
    out.add_argument("--out", default="capture", metavar="PREFIX")
    out.add_argument("--no-autostart", action="store_true")
    out.add_argument("--join", nargs="+", metavar="FILE",
                     help="offline: join sequential single-channel sessions by AdvA "
                          "and reconstruct cross-channel RSSI ratios")
    out.add_argument("--self-test", action="store_true",
                     help="run the estimator suite against synthetic signals and exit")
    return p


def make_config(args):
    from sniffer.channels import ChannelPlan
    from sniffer.radio import RadioConfig

    plan = ChannelPlan.from_args(
        channel=args.channel,
        freq_hz=args.freq,
        access_address=args.access_address,
        crc_init=args.crc_init,
    )
    return RadioConfig(
        plan=plan,
        sample_rate=args.sample_rate,
        bandwidth=args.bandwidth,
        gain_db=args.gain,
        agc=args.agc,
        rx_channels=(0, 1) if args.dual_antenna else (0,),
        block_size=args.block_size,
        external_clock=args.external_clock,
        refclk_hz=args.refclk,
        bias_tee=args.bias_tee,
        device_id=args.device,
    )


def run_join(paths: list[str]) -> int:
    """Offline reconstruction of cross-channel RSSI ratios."""
    import csv

    from sniffer.export import ChannelSession, join_sessions_by_adva

    sessions = []
    for path in paths:
        rows = []
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            rows = pq.read_table(path).to_pylist()
        else:
            with open(path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    r["crc_ok"] = str(r.get("crc_ok", "")).lower() in ("true", "1")
                    try:
                        r["rssi_dbfs"] = float(r.get("rssi_dbfs", "nan"))
                    except ValueError:
                        r["rssi_dbfs"] = float("nan")
                    rows.append(r)
        ch = rows[0].get("channel") if rows else None
        sessions.append(ChannelSession(int(ch) if ch is not None else -1, path, rows))

    result = join_sessions_by_adva(sessions)
    print(json.dumps(result, indent=2, default=str))
    print("\n" + "!" * 72, file=sys.stderr)
    print(result["warning"], file=sys.stderr)
    print("!" * 72, file=sys.stderr)
    return 0


def run_headless(cfg, args) -> int:
    from sniffer import export as X
    from sniffer.pipeline import SnifferPipeline

    pipe = SnifferPipeline(
        cfg, enroll=tuple(args.enroll), ring_seconds=args.ring_seconds,
        keep_iq=True, rssi_cal_db=args.rssi_cal,
    )
    pipe.start()
    print(f"capturing {cfg.plan.label} for {args.seconds:.0f} s ...", flush=True)

    records: list = []
    deadline = time.time() + args.seconds + 6.0
    started = False
    t0 = None
    while time.time() < deadline:
        for line in pipe.log_lines[len(getattr(run_headless, "_seen", [])):]:
            print("  " + line, flush=True)
        run_headless._seen = list(pipe.log_lines)
        records.extend(pipe.drain())
        st = pipe.poll_stats()
        if not started and pipe.ring.written() > 4:
            started = True
            t0 = time.time()
            deadline = t0 + args.seconds
        time.sleep(0.05)

    st = pipe.poll_stats()
    records.extend(pipe.drain())
    elapsed = (time.time() - t0) if t0 else args.seconds
    pipe.stop()

    packets = [r for r in records if not r.is_event]
    good = [r for r in packets if r.crc_ok]
    lost_pct = 100 * st.lost_samples / max(st.samples + st.lost_samples, 1)
    print(
        f"\n{len(packets)} packets, {len(good)} CRC-OK "
        f"({100*len(good)/max(len(packets),1):.1f}%), "
        f"{len(good)/max(elapsed,1e-9):.1f} good packets/s"
    )
    print(
        f"blocks {st.blocks}, queue drops {st.capture_drops}, USB overruns "
        f"{st.usb_overruns}, proven lost samples {st.lost_samples} ({lost_pct:.4f}%), "
        f"DSP load {100*st.dsp_load:.0f}%"
    )
    print(f"reference: {'GPSDO locked' if st.calibrated else 'UNCALIBRATED'} -- {st.clock_detail}")

    if records:
        X.write_csv(args.out + ".csv", records)
        try:
            X.write_parquet(args.out + ".parquet", records)
        except RuntimeError as exc:
            print(f"  {exc}")
        X.write_pcap(args.out + ".pcap", records)
        X.write_session_manifest(args.out + ".json", {
            "channel": cfg.plan.channel,
            "frequency_hz": cfg.plan.frequency_hz,
            "access_address": f"0x{cfg.plan.access_address:08X}",
            "crc_init": f"0x{cfg.plan.crc_init:06X}",
            "sample_rate": cfg.sample_rate,
            "bandwidth": cfg.bandwidth,
            "gain_db": cfg.gain_db,
            "agc": cfg.agc,
            "external_clock": cfg.external_clock,
            "calibrated": st.calibrated,
            "clock_detail": st.clock_detail,
            "packets": len(packets),
            "crc_ok": len(good),
            "lost_samples": st.lost_samples,
            "lost_percent": lost_pct,
            "elapsed_s": elapsed,
        })
        print(f"wrote {args.out}.csv/.parquet/.pcap/.json")

    addrs: dict = {}
    for r in good:
        if r.adva:
            addrs.setdefault(r.adva, [0, r.pdu_name, r.adva_kind])[0] += 1
    if addrs:
        print("\ndevices seen:")
        for a, (n, ty, kind) in sorted(addrs.items(), key=lambda kv: -kv[1][0]):
            print(f"  {a}  {n:5d}  {ty:<16s} {kind}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.join:
        return run_join(args.join)

    if args.self_test:
        import pytest

        return pytest.main(["-q", "tests/"])

    try:
        cfg = make_config(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dual_antenna:
        print(
            "note: --dual-antenna adds a second coherent antenna at the SAME "
            "frequency (both RX channels share one LO). It does not capture two "
            "BLE channels at once.",
            file=sys.stderr,
        )

    if args.headless:
        return run_headless(cfg, args)

    from sniffer.gui.app import run_gui

    return run_gui(cfg, enroll=tuple(args.enroll), autostart=not args.no_autostart)


if __name__ == "__main__":
    sys.exit(main())
