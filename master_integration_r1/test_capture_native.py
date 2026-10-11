import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import capture_native as c


class FakeResponse:
    status = 200
    def __init__(self, data):
        self.body = io.BytesIO(json.dumps(data).encode())
        self.headers = {"Date": "Sun, 11 Oct 2026 11:20:00 GMT", "Content-Type": "application/json"}
    def __enter__(self):
        return self
    def __exit__(self, a, b, d):
        return False
    def read(self, n):
        return self.body.read(n)


def row(start, opening="100", closing="101", quote="1000", buyer="540"):
    return [start, opening, opening, opening, closing, "12", start + 3599999, quote, 1, "3", buyer, "0"]


def pair15(p0="100", p1="101", q0="400", q1="600", t0="160", t1="380"):
    a = row(0, opening=p0, closing="100", quote=q0, buyer=t0)
    b = row(900000, opening="100", closing=p1, quote=q1, buyer=t1)
    return a, b


class CaptureTests(unittest.TestCase):
    def test_universe_canonical(self):
        u = ["BTC", "ETH"]
        self.assertEqual(c.ordered_universe_hash(u), c.digest(b'["BTC","ETH"]'))

    def test_r0_exact_price_gate(self):
        a, b = pair15(p1="101", t1="0")
        self.assertTrue(c.r0_replay_only(a, b))
        a, b = pair15(p1="100.800", t1="380")
        self.assertTrue(c.r0_replay_only(a, b))
        a, b = pair15(p1="100.799999", t1="600")
        self.assertFalse(c.r0_replay_only(a, b))

    def test_r0_weighted_taker_only(self):
        # 20% on one bar and ~77% on the other -> weighted 54%, not mean.
        a, b = pair15(p1="100.9", q0="100", q1="900", t0="20", t1="520")
        self.assertTrue(c.r0_replay_only(a, b))
        b[10] = "519.999"
        self.assertFalse(c.r0_replay_only(a, b))

    def test_r0_unknown_is_not_false(self):
        a, b = pair15(q0="0", q1="0", t0="0", t1="0")
        self.assertIsNone(c.r0_replay_only(a, b))

    def test_v82_strict_thresholds(self):
        prev, curr = row(0, quote="500"), row(3600000, opening="100", closing="101", quote="1000", buyer="550")
        self.assertTrue(c.v82_replay_only(prev, curr))
        curr[10] = "549.999999999"
        self.assertFalse(c.v82_replay_only(prev, curr))
        curr[10] = "550"
        curr[7] = "999.999999"
        self.assertFalse(c.v82_replay_only(prev, curr))

    def test_open_and_gap_fail_closed(self):
        start = 1791709200000
        pair = [row(start), row(start+3600000)]
        now = "2026-10-11T20:20:00+09:00"
        self.assertTrue(c.validate_1h_klines(pair, start, now))
        pair[1][6] = start + 7200000
        self.assertFalse(c.validate_1h_klines(pair, start, now))
        pair[1][6] = start + 7200000-1
        self.assertFalse(c.validate_1h_klines(pair, start, "2026-10-11T19:30:00+09:00"))

    def test_http_only_allowlisted_get(self):
        for u in ["http://api.binance.com/api/v3/time", "https://evil.example/a", "https://api.binance.com.evil.example/x"]:
            with self.assertRaises(ValueError):
                c.http_get_public(u, opener=lambda *_a, **_k: None)

    def test_single_raw_capture_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            src = {"test": ["100.10", 1], "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}
            data = c.capture_request(c.BITHUMB, "/v1/ticker", {"markets": "KRW-BTC"}, "ticker", Path(td), opener=lambda *_a, **_k: FakeResponse(src))
            self.assertEqual(data["capture_status"], "EXECUTED_STRUCTURED_RAW_FILE_VERIFIED")
            self.assertTrue(data["readback_verified"])
            self.assertEqual(c.digest((Path(td) / data["file"]).read_bytes()), data["sha256"])
            reused = c.capture_request(c.BITHUMB, "/v1/ticker", {"markets": "KRW-BTC"}, "ticker", Path(td), opener=lambda *_a, **_k: FakeResponse(src))
            self.assertEqual(reused["capture_status"], "BLOCKED")
            self.assertIn("FileExistsError", reused["error"])

    def test_bithumb_trio_missing_is_unknown_not_zero(self):
        out = c.check_trio([], "KRW-BTC")
        self.assertEqual(out["status"], "BLOCKED")
        self.assertEqual(out["after_cost_EV"], "NOT_COMPUTABLE")
        self.assertIsNone(out["actual_fills"])

    def test_malformed_orderbook_blocks(self):
        now = c.utc_now()
        p = []
        for name, d in [("ticker", [{"market": "KRW-BTC", "timestamp": c.iso_ms(now)}]), ("trades", [{"trade_timestamp": c.iso_ms(now)}]), ("orderbook", [{"market": "KRW-BTC", "timestamp": c.iso_ms(now), "orderbook_units": [{"ask_price": "99", "bid_price": "100"}]}])]:
            p.append({"source_id": name, "capture_status": "EXECUTED_STRUCTURED_RAW_FILE_VERIFIED", "received_at_utc": now, "parsed": d})
        self.assertEqual(c.check_trio(p, "KRW-BTC")["status"], "BLOCKED")

    def test_raw_missing_not_normalized_to_no_signal(self):
        with tempfile.TemporaryDirectory() as td:
            def missing(*_a, **_k):
                raise OSError("DNS intentionally disabled")
            result = c.main(["--mode", "bithumb-trio", "--run-id", "test1011", "--out", td], opener=missing)
            self.assertEqual(result, 2)
            manifest = json.loads((Path(td)/"test1011"/"manifest.json").read_text())
            self.assertEqual(len(manifest["sources"]), 4)
            self.assertEqual(manifest["assessment"]["bithumb_trio"]["status"], "BLOCKED")
            self.assertEqual(manifest["strict_prospective_credit_increment"], 0)
            self.assertEqual(manifest["after_cost_ev"], "NOT_COMPUTABLE")
            self.assertEqual(c.main(["--mode", "verify-manifest", "--manifest", str(Path(td)/"test1011"/"manifest.json")]), 2)

    def test_bithumb_trio_realistic_mock_is_only_data_gate(self):
        stamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        sample = {
            "/v1/market/all": [{"market": "KRW-BTC"}, {"market": "KRW-ETH"}],
            "/v1/ticker": [{"market": "KRW-BTC", "timestamp": stamp, "trade_price": 100000000}],
            "/v1/trades/ticks": [{"trade_timestamp": stamp, "trade_price": 100000000}],
            "/v1/orderbook": [{"market": "KRW-BTC", "timestamp": stamp, "orderbook_units": [{"ask_price": 100000100, "bid_price": 99999900, "ask_size": 1, "bid_size": 1}]}],
        }
        with tempfile.TemporaryDirectory() as td:
            def fake(req, timeout):
                return FakeResponse(sample[urlparse(req.full_url).path])
            code = c.main(["--mode", "bithumb-trio", "--market", "KRW-BTC", "--run-id", "trio123", "--out", td], opener=fake)
            self.assertEqual(code, 0)
            manifest = Path(td)/"trio123"/"manifest.json"
            result = json.loads(manifest.read_text())
            self.assertEqual(result["assessment"]["bithumb_trio"]["status"], "STRUCTURE_AND_FRESHNESS_VERIFIED__COST_INPUTS_STILL_REQUIRED")
            self.assertEqual(result["after_cost_ev"], "NOT_COMPUTABLE")
            self.assertIsNone(result["realized_pnl"])
            self.assertEqual(c.main(["--mode", "verify-manifest", "--manifest", str(manifest)]), 0)

    def test_binance_78_one_network_failure_blocks_universe(self):
        with tempfile.TemporaryDirectory() as td:
            names = ["S" + str(i) for i in range(78)]
            pre = Path(td)/"prefreeze.json"
            pre.write_text(json.dumps({"universe": names, "universe_sha256": c.ordered_universe_hash(names)}))
            start = 1791709200000
            def fake(req, timeout):
                if parse_qs(urlparse(req.full_url).query)["symbol"][0] == "S23USDT":
                    raise OSError("rate limit outage synthetic")
                return FakeResponse([row(start), row(start+3600000)])
            code = c.main(["--mode", "binance-mature", "--prefreeze", str(pre), "--start-ms", str(start), "--run-id", "gap77", "--out", td], opener=fake)
            self.assertEqual(code, 2)
            manifest = Path(td)/"gap77"/"manifest.json"
            result = json.loads(manifest.read_text())
            self.assertEqual(result["assessment"]["captured_and_valid"], 77)
            self.assertEqual(result["assessment"]["status"], "BLOCKED_PARTIAL_NOT_ZERO")
            self.assertEqual(result["strict_prospective_credit_increment"], 0)
            self.assertEqual(c.main(["--mode", "verify-manifest", "--manifest", str(manifest)]), 2)

    def test_78_full_native_test_with_mock(self):
        # The mock tests implementation/archival only, not real exchange access.
        names = ["S" + str(i) for i in range(78)]
        with tempfile.TemporaryDirectory() as td:
            pre = Path(td)/"prefreeze.json"
            pre.write_text(json.dumps({"universe": names, "universe_sha256": c.ordered_universe_hash(names)}))
            start = 1791709200000
            def fake(req, timeout):
                q = parse_qs(urlparse(req.full_url).query)
                self.assertTrue(q["symbol"][0].endswith("USDT"))
                return FakeResponse([row(start), row(start+3600000)])
            code = c.main(["--mode", "binance-mature", "--prefreeze", str(pre), "--start-ms", str(start), "--run-id", "test78", "--out", td], opener=fake)
            self.assertEqual(code, 0)
            manifest = Path(td)/"test78"/"manifest.json"
            d = json.loads(manifest.read_text())
            self.assertEqual(d["assessment"]["captured_and_valid"], 78)
            self.assertEqual(d["strict_prospective_credit_increment"], 0)
            self.assertEqual(c.main(["--mode", "verify-manifest", "--manifest", str(manifest)]), 0)
            raw = Path(td)/"test78"/d["sources"][0]["file"]
            raw.write_bytes(b'TAMPERED')
            self.assertEqual(c.main(["--mode", "verify-manifest", "--manifest", str(manifest)]), 2)

    def test_invalid_identifier_no_path_escape(self):
        for name in ["../x", "", "/absolute", "AA..XX"]:
            with self.assertRaises(ValueError):
                c.valid_run_id(name)


    def test_binance_official_public_data_only_hostname(self):
        from urllib.parse import urlparse
        self.assertEqual(urlparse(c.BINANCE_MARKET_DATA).hostname, "data-api.binance.vision")
        self.assertIn("data-api.binance.vision", c.ALLOWED_HOSTS)
        self.assertNotIn("evil.example", c.ALLOWED_HOSTS)

    def test_binance_rejects_unapproved_data_base(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                c.capture_binance_78(Path(td), "unused", 0, None, base="https://example.org")


if __name__ == "__main__":
    unittest.main(verbosity=2)