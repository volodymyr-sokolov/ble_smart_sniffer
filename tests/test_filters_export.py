"""Filter parser and export format tests."""

from __future__ import annotations

import os
import struct
import tempfile

import numpy as np
import pytest

from sniffer import export as X
from sniffer.features import Measurement, PacketFeatures
from sniffer.gui.filters import FilterError, compile_filter
from sniffer.packet import PacketRecord, make_event


def mkrec(**kw) -> PacketRecord:
    feats = PacketFeatures(
        measurements={
            "cfo_ppm": Measurement(kw.pop("cfo_ppm", 5.0), 0.5, "ppm"),
            "modulation_index": Measurement(kw.pop("modidx", 0.50), 0.01, ""),
            "rise_time_us": Measurement(kw.pop("rise", 2.0), 0.1, "us"),
        }
    )
    base = dict(
        number=1,
        adva="AA:BB:CC:DD:EE:FF",
        pdu_name="ADV_IND",
        length=30,
        crc_ok=True,
        rssi_dbfs=-50.0,
        channel=37,
        anomaly_score=0.2,
        features=feats,
        raw_bytes=bytes(range(20)),
        payload=bytes(range(12)),
    )
    base.update(kw)
    return PacketRecord(**base)


# --------------------------------------------------------------------------
# filter parser
# --------------------------------------------------------------------------

def test_equality_on_address():
    f = compile_filter("adva == AA:BB:CC:DD:EE:FF")
    assert f(mkrec())
    assert not f(mkrec(adva="11:22:33:44:55:66"))


def test_address_match_is_case_insensitive():
    f = compile_filter("adva == aa:bb:cc:dd:ee:ff")
    assert f(mkrec())


def test_crc_fail():
    f = compile_filter("crc == fail")
    assert f(mkrec(crc_ok=False))
    assert not f(mkrec(crc_ok=True))


def test_numeric_comparison():
    f = compile_filter("cfo_ppm > 15")
    assert f(mkrec(cfo_ppm=20.0))
    assert not f(mkrec(cfo_ppm=5.0))


def test_conjunction_from_the_spec():
    f = compile_filter("pdu_type == ADV_IND && rssi > -60")
    assert f(mkrec(rssi_dbfs=-50.0))
    assert not f(mkrec(rssi_dbfs=-70.0))
    assert not f(mkrec(pdu_name="SCAN_REQ", rssi_dbfs=-50.0))


def test_anomaly_threshold():
    f = compile_filter("anomaly > 0.8")
    assert f(mkrec(anomaly_score=0.9))
    assert not f(mkrec(anomaly_score=0.1))


def test_disjunction_and_negation_and_parens():
    f = compile_filter("(crc == fail || rssi < -80) && !(pdu_type == SCAN_REQ)")
    assert f(mkrec(crc_ok=False))
    assert not f(mkrec(crc_ok=False, pdu_name="SCAN_REQ"))
    assert not f(mkrec(crc_ok=True, rssi_dbfs=-40.0))


def test_contains_operator():
    f = compile_filter('info contains "Flags"')
    assert f(mkrec(info="Flags | LE General"))
    assert not f(mkrec(info="Complete Local Name"))


def test_bare_field_is_truthiness():
    f = compile_filter("crc_ok")
    assert f(mkrec(crc_ok=True))
    assert not f(mkrec(crc_ok=False))


def test_nan_feature_matches_nothing_except_not_equal():
    rec = mkrec()
    rec.features.measurements["cfo_ppm"] = Measurement(float("nan"))
    assert not compile_filter("cfo_ppm > 1")(rec)
    assert not compile_filter("cfo_ppm < 1")(rec)
    assert not compile_filter("cfo_ppm == 1")(rec)
    assert compile_filter("cfo_ppm != 1")(rec)


def test_empty_filter_is_none():
    assert compile_filter("") is None
    assert compile_filter("   ") is None


@pytest.mark.parametrize(
    "expr",
    [
        "adva ==",
        "== AA",
        "nosuchfield > 3",
        "adva == AA:BB:CC:DD:EE:FF &&",
        "(crc == fail",
        "crc == fail)",
        "rssi >> 3",
        "@@@",
    ],
)
def test_invalid_expressions_raise(expr):
    with pytest.raises(FilterError):
        compile_filter(expr)


def test_filter_never_executes_arbitrary_code():
    """The bar must not be an eval() in disguise."""
    with pytest.raises(FilterError):
        compile_filter("__import__('os').system('echo pwned')")


def test_predicate_survives_a_broken_record():
    f = compile_filter("cfo_ppm > 1")

    class Broken:
        is_event = False

        def feature(self, k):
            raise RuntimeError("boom")

    assert f(Broken()) is False


# --------------------------------------------------------------------------
# exports
# --------------------------------------------------------------------------

def test_pcap_header_and_linktype():
    recs = [mkrec(number=i) for i in range(5)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pcap")
        n = X.write_pcap(p, recs)
        assert n == 5
        with open(p, "rb") as fh:
            hdr = fh.read(24)
        magic, vmaj, vmin, tz, sig, snap, link = struct.unpack("<IHHiIII", hdr)
        assert magic == X.PCAP_MAGIC_US
        assert (vmaj, vmin) == (2, 4)
        assert link == X.LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR == 256


def test_pcap_marks_crc_validity():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pcap")
        X.write_pcap(p, [mkrec(crc_ok=True), mkrec(crc_ok=False)])
        with open(p, "rb") as fh:
            fh.read(24)
            flags = []
            for _ in range(2):
                _, _, cap, _ = struct.unpack("<IIII", fh.read(16))
                blob = fh.read(cap)
                flags.append(struct.unpack("<BbbBIH", blob[:10])[5])
    assert flags[0] & X.PHDR_FLAG_CRC_VALID
    assert not flags[1] & X.PHDR_FLAG_CRC_VALID


def test_pcap_skips_event_rows():
    recs = [mkrec(), make_event(2, 0.0, 37, "interference", "CW detected")]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pcap")
        assert X.write_pcap(p, recs) == 1


def test_sigmf_roundtrip_and_metadata():
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64) * 0.1
    with tempfile.TemporaryDirectory() as d:
        data, meta = X.write_sigmf(
            os.path.join(d, "cap"), iq, 8e6, 2.402e9, channel=37, calibrated=False
        )
        assert os.path.exists(data) and os.path.exists(meta)
        import json

        m = json.load(open(meta))
        assert m["global"]["core:datatype"] == "ci16_le"
        assert m["global"]["core:sample_rate"] == 8e6
        assert m["global"]["ble:channel"] == 37
        assert "UNCALIBRATED" in m["global"]["core:description"]
        raw = np.fromfile(data, dtype="<i2")
        assert raw.size == iq.size * 2
        back = (raw[0::2] + 1j * raw[1::2]) / 2048.0
        assert np.max(np.abs(back - iq)) < 2e-3


def test_sigmf_calibrated_has_no_warning():
    iq = np.zeros(1024, dtype=np.complex64)
    with tempfile.TemporaryDirectory() as d:
        _, meta = X.write_sigmf(
            os.path.join(d, "c"), iq, 8e6, 2.402e9, calibrated=True, description="x"
        )
        import json

        assert "UNCALIBRATED" not in json.load(open(meta))["global"]["core:description"]


def test_csv_and_parquet():
    recs = [mkrec(number=i, adva=f"AA:BB:CC:DD:EE:{i:02X}") for i in range(7)]
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "x.csv")
        assert X.write_csv(c, recs) == 7
        text = open(c, encoding="utf-8").read()
        assert "adva" in text and "cfo_ppm" in text
        pq_path = os.path.join(d, "x.parquet")
        n = X.write_parquet(pq_path, recs)
        assert n == 7
        import pyarrow.parquet as pq

        t = pq.read_table(pq_path)
        assert t.num_rows == 7
        assert "modulation_index" in t.column_names


def test_offline_join_carries_its_warning():
    sessions = [
        X.ChannelSession(
            ch,
            f"s{ch}.parquet",
            [
                {"adva": "AA:BB:CC:DD:EE:FF", "crc_ok": True, "rssi_dbfs": -50.0 - ch * 0.1}
                for _ in range(10)
            ],
        )
        for ch in (37, 38, 39)
    ]
    out = X.join_sessions_by_adva(sessions)
    assert "not simultaneous" in out["warning"].lower()
    assert len(out["devices"]) == 1
    dev = out["devices"][0]
    assert dev["valid_only_if_stationary"] is True
    assert set(dev["channels"]) == {37, 38, 39}
    assert "37/38" in dev["rssi_ratio_db"]
