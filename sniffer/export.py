"""Exports: SigMF for IQ, PCAP for Wireshark, CSV/Parquet for the feature store.

The PCAP path uses LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR (256) so a capture can be
opened in real Wireshark next to this tool and the two dissections compared --
which is the only practical way to be confident the link-layer parsing here is
right rather than merely self-consistent.
"""

from __future__ import annotations

import csv
import json
import os
import struct
import time
from dataclasses import dataclass

import numpy as np

# ---- PCAP ----------------------------------------------------------------

LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR = 256

PCAP_MAGIC_US = 0xA1B2C3D4

# bits in the BLE pseudo-header flags field (per the Wireshark dissector)
PHDR_FLAG_DEWHITENED = 0x0001
PHDR_FLAG_SIGPOWER_VALID = 0x0002
PHDR_FLAG_NOISEPOWER_VALID = 0x0004
PHDR_FLAG_DECRYPTED = 0x0008
PHDR_FLAG_REFAA_VALID = 0x0010
PHDR_FLAG_AA_CRC_CHECKED = 0x0020
PHDR_FLAG_AA_CRC_VALID = 0x0040
PHDR_FLAG_CRC_CHECKED = 0x0400
PHDR_FLAG_CRC_VALID = 0x0800


class PcapWriter:
    """Streaming PCAP writer for BLE link-layer packets with the phdr."""

    def __init__(self, path: str, snaplen: int = 512) -> None:
        self.path = path
        self._fh = open(path, "wb")
        self._fh.write(
            struct.pack(
                "<IHHiIII",
                PCAP_MAGIC_US,
                2,
                4,
                0,  # thiszone
                0,  # sigfigs
                snaplen,
                LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR,
            )
        )
        self.count = 0

    def write(self, rec) -> None:
        """Append one PacketRecord.  Events and undecoded rows are skipped."""
        if rec.is_event or not rec.raw_bytes:
            return

        body = bytes(rec.raw_bytes)
        # The phdr wants access address + PDU + CRC; raw_bytes holds PDU+CRC.
        aa = struct.pack("<I", rec.access_address & 0xFFFFFFFF)
        payload = aa + body

        flags = PHDR_FLAG_DEWHITENED | PHDR_FLAG_REFAA_VALID | PHDR_FLAG_CRC_CHECKED
        if rec.crc_ok:
            flags |= PHDR_FLAG_CRC_VALID
        sig = int(round(rec.rssi_dbm)) if np.isfinite(rec.rssi_dbm) else 0
        if np.isfinite(rec.rssi_dbm):
            flags |= PHDR_FLAG_SIGPOWER_VALID
        noise = int(round(rec.noise_floor_dbfs)) if np.isfinite(rec.noise_floor_dbfs) else 0
        if np.isfinite(rec.noise_floor_dbfs):
            flags |= PHDR_FLAG_NOISEPOWER_VALID

        # struct: uint8 channel, int8 signal, int8 noise, uint8 aa_offenses,
        #         uint32 ref_aa, uint16 flags
        phdr = struct.pack(
            "<BbbBIH",
            rec.channel & 0xFF,
            max(min(sig, 127), -128),
            max(min(noise, 127), -128),
            0,
            rec.access_address & 0xFFFFFFFF,
            flags,
        )
        blob = phdr + payload

        ts = rec.wall_time or time.time()
        sec = int(ts)
        usec = int((ts - sec) * 1e6)
        self._fh.write(struct.pack("<IIII", sec, usec, len(blob), len(blob)))
        self._fh.write(blob)
        self.count += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_pcap(path: str, records) -> int:
    with PcapWriter(path) as w:
        for r in records:
            w.write(r)
        return w.count


# ---- SigMF ---------------------------------------------------------------

SIGMF_VERSION = "1.0.0"


def write_sigmf(
    base_path: str,
    iq: np.ndarray,
    sample_rate: float,
    frequency_hz: float,
    *,
    description: str = "",
    channel: int | None = None,
    gain_db: float | None = None,
    calibrated: bool = False,
    hardware: str = "Nuand bladeRF 2.0 micro",
    extra: dict | None = None,
    annotations: list | None = None,
) -> tuple[str, str]:
    """Write `base_path`.sigmf-data and `.sigmf-meta`.

    Samples are stored as ci16_le -- the format they arrived in -- rather than
    converted to float32.  That halves the file and keeps the recording bit
    exact with respect to the ADC, which matters if the capture is going to be
    re-analysed by a different tool later.
    """
    if base_path.endswith((".sigmf-data", ".sigmf-meta")):
        base_path = base_path.rsplit(".", 1)[0]
    data_path = base_path + ".sigmf-data"
    meta_path = base_path + ".sigmf-meta"

    x = np.asarray(iq)
    if np.iscomplexobj(x):
        scaled = np.clip(np.stack([x.real, x.imag], axis=-1) * 2048.0, -2048, 2047)
        raw = scaled.astype("<i2").reshape(-1)
    else:
        raw = x.astype("<i2").reshape(-1)
    raw.tofile(data_path)

    meta = {
        "global": {
            "core:datatype": "ci16_le",
            "core:sample_rate": float(sample_rate),
            "core:version": SIGMF_VERSION,
            "core:hw": hardware,
            "core:description": description,
            "core:recorder": "ble-bladerf-sniffer",
            "core:num_channels": 1,
            # The scale factor back to full scale, so a later tool does not have
            # to guess what 2048 meant.
            "ble:full_scale": 2048,
            "ble:calibrated_reference": bool(calibrated),
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(frequency_hz),
                "core:datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ],
        "annotations": list(annotations or []),
    }
    if channel is not None:
        meta["global"]["ble:channel"] = int(channel)
    if gain_db is not None:
        meta["captures"][0]["core:global_index"] = 0
        meta["global"]["ble:rx_gain_db"] = float(gain_db)
    if not calibrated:
        meta["global"]["core:description"] = (
            (description + " " if description else "")
            + "[UNCALIBRATED: no disciplined reference locked; ppm-scale values "
            "in any derived analysis include receiver drift]"
        ).strip()
    if extra:
        meta["global"].update(extra)

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return data_path, meta_path


def export_packet_iq(path: str, rec, sample_rate: float) -> tuple[str, str] | None:
    """Dump the retained IQ behind one packet, annotated with where it starts."""
    if rec.iq is None or not len(rec.iq):
        return None
    ann = [
        {
            "core:sample_start": int(rec.sync_offset_in_slice),
            "core:sample_count": int(len(rec.iq) - rec.sync_offset_in_slice),
            "core:label": f"{rec.pdu_name} {rec.adva}",
            "core:description": (
                f"CRC {'OK' if rec.crc_ok else 'FAIL'}; "
                f"RSSI {rec.rssi_dbfs:.1f} dBFS; packet #{rec.number}"
            ),
        }
    ]
    return write_sigmf(
        path,
        rec.iq,
        sample_rate,
        rec.frequency_hz,
        description=f"BLE packet #{rec.number} on channel {rec.channel}",
        channel=rec.channel,
        gain_db=rec.gain_db,
        calibrated=rec.calibrated,
        annotations=ann,
    )


# ---- CSV / Parquet -------------------------------------------------------

def _rows(records) -> list[dict]:
    return [r.as_row() for r in records if not r.is_event]


def write_csv(path: str, records) -> int:
    rows = _rows(records)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return 0
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def write_parquet(path: str, records) -> int:
    """Parquet via pyarrow; falls back to CSV beside it if pyarrow is absent."""
    rows = _rows(records)
    if not rows:
        return 0
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        alt = os.path.splitext(path)[0] + ".csv"
        write_csv(alt, records)
        raise RuntimeError(
            f"pyarrow is not installed; wrote {alt} instead of {path}"
        )

    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    cols = {k: [r.get(k) for r in rows] for k in keys}
    table = pa.table(cols)
    pq.write_table(table, path)
    return len(rows)


# ---- session manifest ----------------------------------------------------

def write_session_manifest(path: str, info: dict) -> str:
    """Record how the session was configured, including calibration state."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)
    return path


# ---- offline multi-channel join -----------------------------------------

@dataclass
class ChannelSession:
    """One single-channel session, for the offline three-channel join."""

    channel: int
    path: str
    rows: list


def join_sessions_by_adva(sessions: list[ChannelSession]) -> dict:
    """Reconstruct cross-channel RSSI ratios from sequential single-channel runs.

    THIS IS NOT A SIMULTANEOUS MEASUREMENT.  The sessions were recorded one
    after another, so the ratio is only meaningful for a device that did not
    move, rotate, or change transmit power between them, and for an environment
    whose multipath did not change either.  A person walking through the room
    between session one and session three will produce a ratio that looks
    exactly like a different transmitter.

    The warning is returned in the result rather than logged, so it travels with
    the data into whatever report is built from it.
    """
    per_addr: dict[str, dict[int, list[float]]] = {}
    for s in sessions:
        for row in s.rows:
            a = row.get("adva")
            if not a or not row.get("crc_ok"):
                continue
            v = row.get("rssi_dbfs")
            if v is None or not np.isfinite(v):
                continue
            per_addr.setdefault(a, {}).setdefault(s.channel, []).append(float(v))

    out = {
        "warning": (
            "Sessions are sequential, not simultaneous. Cross-channel RSSI "
            "ratios are valid only for a device that did not move and an "
            "environment that did not change between sessions."
        ),
        "sessions": [{"channel": s.channel, "path": s.path, "packets": len(s.rows)} for s in sessions],
        "devices": [],
    }
    for a, by_ch in sorted(per_addr.items()):
        if len(by_ch) < 2:
            continue
        means = {ch: float(np.mean(v)) for ch, v in by_ch.items()}
        counts = {ch: len(v) for ch, v in by_ch.items()}
        stds = {ch: float(np.std(v)) for ch, v in by_ch.items()}
        ratios = {}
        chans = sorted(means)
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                a_, b_ = chans[i], chans[j]
                ratios[f"{a_}/{b_}"] = means[a_] - means[b_]
        out["devices"].append(
            {
                "adva": a,
                "channels": chans,
                "rssi_mean_dbfs": means,
                "rssi_std_db": stds,
                "packets": counts,
                "rssi_ratio_db": ratios,
                "valid_only_if_stationary": True,
            }
        )
    return out
