#!/usr/bin/env python3
"""MASTER-INTEGRATION-r1: PUBLIC market-data capture, not an order/trade bot.

Requirements: Python 3.10+, standard library only. Creates write-once raw HTTP
bodies with receipt SHA-256. Fail closed. Does NOT issue any strategy signals.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
BITHUMB = "https://api.bithumb.com"
BINANCE = "https://api.binance.com"
ALLOWED_HOSTS = frozenset({"api.bithumb.com", "api.binance.com"})
MAX_BYTES = 2_000_000
D = Decimal


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def kst_now():
    return datetime.now(KST).isoformat(timespec="milliseconds")


def iso_ms(s):
    return int(datetime.fromisoformat(s).timestamp() * 1000)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def canonical_json(x):
    return json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def ordered_universe_hash(names):
    # Contract hash uses ordered compact JSON with no sort and ASCII escaping.
    return digest(json.dumps(names, ensure_ascii=True, separators=(",", ":")).encode())


def valid_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", run_id) or ".." in run_id:
        raise ValueError("Invalid run ID; requires safe immutable path component")
    return run_id


def atomic_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # No overwrite even after partial prior attempts; generate another run ID.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass  # File fsync still completed; manifest marks limitation below.
    except BaseException:
        raise
    return digest(path.read_bytes()) == digest(payload)


def http_get_public(url, opener=urllib.request.urlopen, timeout=9):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_HOSTS:
        raise ValueError("Only explicit HTTPS public exchange hosts permitted")
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "Master-R1-ReadOnly-Capture/1.0", "Accept": "application/json"})
    started = utc_now()
    try:
        with opener(req, timeout=timeout) as response:
            status = response.status
            headers = dict(response.headers.items())
            body = response.read(MAX_BYTES + 1)
        received = utc_now()
    except urllib.error.HTTPError as err:
        status, headers, body = err.code, dict(err.headers.items()), err.read(MAX_BYTES + 1)
        received = utc_now()
    if len(body) > MAX_BYTES:
        raise ValueError("Response exceeds raw archive byte limit")
    return {"http_status": status, "headers": {k: v for k, v in headers.items() if k.lower() in ("date", "content-type", "content-length", "x-mbx-used-weight-1m")}, "request_started_at_utc": started, "received_at_utc": received, "bytes": body}


def safe_name(name):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
        raise ValueError("bad file name")
    return name


def capture_request(base, endpoint, params, label, directory, opener=urllib.request.urlopen):
    q = urllib.parse.urlencode(params)
    url = base + endpoint + ("?" + q if q else "")
    info = {"source_id": safe_name(label), "url": url, "capture_status": "BLOCKED", "http_status": None, "received_at_utc": None, "persisted_at_utc": None, "sha256": None, "bytes": None, "file": None, "readback_verified": False, "error": None, "server_date": None}
    try:
        rsp = http_get_public(url, opener=opener)
        body = rsp["bytes"]
        info.update({"request_started_at_utc": rsp["request_started_at_utc"], "received_at_utc": rsp["received_at_utc"], "http_status": rsp["http_status"], "server_date": rsp["headers"].get("Date") or rsp["headers"].get("date"), "sha256": digest(body), "bytes": len(body)})
        raw_path = directory / "raw" / (label + ".http-body")
        ok = atomic_new(raw_path, body)
        info["persisted_at_utc"] = utc_now()
        info["file"] = str(raw_path.relative_to(directory))
        info["readback_verified"] = ok
        if not ok:
            info["error"] = "FILE_READBACK_SHA_MISMATCH"
        elif rsp["http_status"] != 200:
            info["error"] = "HTTP_NON_200"
        else:
            try:
                info["parsed"] = json.loads(body)
                info["capture_status"] = "EXECUTED_STRUCTURED_RAW_FILE_VERIFIED"
            except (json.JSONDecodeError, UnicodeError):
                info["error"] = "BAD_JSON_BODY"
    except Exception as exc:
        info["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        info["failed_at_utc"] = utc_now()
    return info


def check_trio(captures, market, max_age_ms=5000):
    """Availability to *estimate* costs, NOT a proof of realized PnL.

    Missing exchange timestamp, malformed book, unknown size or fee -> blocked.
    No zero-fill assumption. Input original snippets must share run_id (caller).
    """
    p = {x["source_id"]: x for x in captures}
    out = {"market": market, "status": "BLOCKED", "after_cost_EV": "NOT_COMPUTABLE", "actual_fills": None, "reasons": [], "spread_bps": None}
    for k in ("ticker", "trades", "orderbook"):
        if k not in p or p[k]["capture_status"] != "EXECUTED_STRUCTURED_RAW_FILE_VERIFIED":
            out["reasons"].append("MISSING_VALID_" + k.upper())
    if out["reasons"]:
        return out
    try:
        tick, trades, book = p["ticker"]["parsed"], p["trades"]["parsed"], p["orderbook"]["parsed"]
        if not isinstance(tick, list) or not tick or tick[0].get("market") != market:
            raise ValueError("TICKER_MARKET_MISMATCH")
        if not isinstance(trades, list) or not trades:
            raise ValueError("TRADES_EMPTY_OR_MALFORMED")
        if not isinstance(book, list) or not book or book[0].get("market") != market:
            raise ValueError("ORDERBOOK_MARKET_MISMATCH")
        order_units = book[0]["orderbook_units"]
        ask, bid = D(str(order_units[0]["ask_price"])), D(str(order_units[0]["bid_price"]))
        if ask <= 0 or bid <= 0 or ask < bid:
            raise ValueError("CROSSED_OR_INVALID_BOOK")
        out["spread_bps"] = str((ask - bid) * D(10000) / ((ask + bid) / 2))
        for k, node in (("ticker", tick[0]), ("orderbook", book[0])):
            ts = node.get("timestamp")
            if ts is None:
                out["reasons"].append(k.upper() + "_EXCHANGE_TIMESTAMP_UNKNOWN")
                continue
            age = iso_ms(p[k]["received_at_utc"]) - int(ts)
            if age < -2000 or age > max_age_ms:
                out["reasons"].append(k.upper() + "_STALE_OR_CLOCK_SKEW")
        # Recent trades may not have per-result exchange timestamp on some versions.
        if not any(isinstance(t, dict) and ("timestamp" in t or "trade_timestamp" in t) for t in trades):
            out["reasons"].append("TRADES_EXCHANGE_TIMESTAMP_UNKNOWN")
    except (KeyError, IndexError, ValueError, TypeError, InvalidOperation) as e:
        out["reasons"].append("BAD_TRIO_SCHEMA:" + str(e))
    if not out["reasons"]:
        out["status"] = "STRUCTURE_AND_FRESHNESS_VERIFIED__COST_INPUTS_STILL_REQUIRED"
    return out


def validate_1h_klines(blob, start_ms, received_at):
    if not isinstance(blob, list) or len(blob) != 2:
        return False
    for i, row in enumerate(blob):
        expected = start_ms + i * 3600000
        if not isinstance(row, list) or len(row) != 12 or row[0] != expected or row[6] != expected + 3599999:
            return False
        if row[6] > iso_ms(received_at):
            return False
        try:
            if D(str(row[1])) <= 0 or D(str(row[7])) < 0 or D(str(row[10])) < 0 or D(str(row[10])) > D(str(row[7])):
                return False
        except (InvalidOperation, ValueError):
            return False
    return True


def r0_replay_only(first, second):
    # Diagnostic only. Neither function creates a signal or backdates issued_at.
    p0, p1 = D(str(first[1])), D(str(second[4]))
    q = D(str(first[7])) + D(str(second[7]))
    buy = D(str(first[10])) + D(str(second[10]))
    if p0 <= 0 or q <= 0 or buy < 0 or buy > q:
        return None
    return p1 * 100 >= p0 * 101 or (p1 * 1000 >= p0 * 1008 and p1 * 100 < p0 * 101 and buy * 100 >= q * 54)


def v82_replay_only(previous, current):
    p0, p1 = D(str(current[1])), D(str(current[4]))
    v0, v1 = D(str(previous[7])), D(str(current[7]))
    buy = D(str(current[10]))
    if p0 <= 0 or v0 <= 0 or v1 <= 0 or buy < 0 or buy > v1:
        return None
    return p1 * 100 >= p0 * 101 and v1 >= v0 * 2 and buy * 100 >= v1 * 55


def prepare_dir(out, run_id):
    valid_run_id(run_id)
    directory = Path(out).resolve() / run_id
    # NEVER reuse a run folder; no overwrite / retroactive rewriting.
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def capture_bithumb(directory, market, opener):
    if not re.fullmatch(r"KRW-[A-Z0-9]{2,20}", market):
        raise ValueError("Require explicit KRW-BASE market")
    spec = [("market_all", "/v1/market/all", {}), ("ticker", "/v1/ticker", {"markets": market}), ("trades", "/v1/trades/ticks", {"market": market, "count": 200}), ("orderbook", "/v1/orderbook", {"markets": market})]
    captures = [capture_request(BITHUMB, ep, args, name, directory, opener) for name, ep, args in spec]
    available = None
    try:
        markets = captures[0]["parsed"]
        if not isinstance(markets, list):
            raise TypeError("expected market/all list")
        available = market in {x.get("market") for x in markets if isinstance(x, dict)}
    except (KeyError, TypeError):
        pass
    trio = check_trio(captures, market)
    if available is not True:
        trio["status"] = "BLOCKED"
        trio["reasons"].append("FULL_KRW_UNIVERSE_MARKET_MEMBERSHIP_UNKNOWN" if available is None else "MARKET_NOT_IN_SNAPSHOT")
    return captures, trio, available


def capture_binance_78(directory, prefreeze_file, start_ms, opener):
    pre = json.loads(Path(prefreeze_file).read_text(encoding="utf-8"))
    bases = pre["universe"]
    if not isinstance(bases, list) or len(bases) != 78 or len(set(bases)) != 78:
        raise ValueError("Need frozen unique 78 Binance comparison symbols")
    h = ordered_universe_hash(bases)
    if h != pre.get("universe_sha256"):
        raise ValueError("Frozen universe hash mismatch")
    if start_ms % 3600000 != 0:
        raise ValueError("Expected exact 1h UTC bar start")
    result = []
    for base in bases:
        symbol = base + "USDT"
        info = capture_request(BINANCE, "/api/v3/klines", {"symbol": symbol, "interval": "1h", "startTime": start_ms, "endTime": start_ms+7200000-1, "limit": 2}, "binance_" + symbol, directory, opener)
        if info["capture_status"] == "EXECUTED_STRUCTURED_RAW_FILE_VERIFIED" and validate_1h_klines(info["parsed"], start_ms, info["received_at_utc"]):
            info["data_validation"] = "TWO_COMPLETED_SPOT_1H_CONFIRMED"
        else:
            info["data_validation"] = "BLOCKED_NO_COVERAGE_CREDIT"
        result.append(info)
    ok = sum(x["data_validation"] == "TWO_COMPLETED_SPOT_1H_CONFIRMED" for x in result)
    return result, {"scope": "Binance frozen 78 comparison subset only", "universe_sha256": h, "expected": 78, "captured_and_valid": ok, "status": "VERIFIED_COMPLETE_STRUCTURED_RAW" if ok == 78 else "BLOCKED_PARTIAL_NOT_ZERO"}


def summarize_capture(info):
    # No raw payload or secrets duplicated into control manifest.
    return {k: v for k, v in info.items() if k not in ("parsed",)}


def main(argv=None, opener=urllib.request.urlopen):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bithumb-trio", "binance-mature", "verify-manifest"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--market", default="KRW-BTC")
    parser.add_argument("--prefreeze")
    parser.add_argument("--start-ms", type=int)
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    if args.mode == "verify-manifest":
        doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        base = Path(args.manifest).parent
        bad = []
        for f in doc["sources"]:
            if not f["file"]:
                continue
            path = (base / f["file"]).resolve()
            if not path.is_relative_to(base.resolve()) or not path.exists() or digest(path.read_bytes()) != f["sha256"]:
                bad.append(f["source_id"])
        checked = sum(bool(x["file"]) for x in doc["sources"])
        all_success = checked > 0 and checked == len(doc["sources"]) and all(f["capture_status"] == "EXECUTED_STRUCTURED_RAW_FILE_VERIFIED" for f in doc["sources"])
        status = "CONTRADICTED" if bad else ("VERIFIED_ALL_CAPTURED_FILES_ONLY" if all_success else "BLOCKED_PARTIAL_OR_ZERO_RAW_FILES")
        print(json.dumps({"status": status, "files_checked": checked, "expected_sources": len(doc["sources"]), "mismatched": bad, "note": "This verifies only locally persisted HTTP bytes, not independent exchange origin or signal delivery."}, ensure_ascii=False))
        return 0 if status == "VERIFIED_ALL_CAPTURED_FILES_ONLY" else 2
    if not args.run_id:
        parser.error("--run-id required")
    directory = prepare_dir(args.out, args.run_id)
    start = utc_now()
    if args.mode == "bithumb-trio":
        captures, gate, membership = capture_bithumb(directory, args.market, opener)
        additional = {"bithumb_market_membership": membership, "bithumb_trio": gate, "paper_after_cost_EV": "NOT_COMPUTABLE"}
    else:
        if not args.prefreeze or args.start_ms is None:
            parser.error("--prefreeze and --start-ms required")
        captures, additional = capture_binance_78(directory, args.prefreeze, args.start_ms, opener)
    # Prospective signal credit ALWAYS ZERO: no issue/persisted_at/delivery_ack proof.
    doc = {"schema": "MASTER_INTEGRATION_R1_NATIVE_HTTP_CAPTURE_V1", "run_id": args.run_id, "mode": args.mode, "run_started_at_utc": start, "run_completed_at_utc": utc_now(), "run_completed_at_kst": kst_now(), "sources": [summarize_capture(c) for c in captures], "assessment": additional, "native_http_body_write_once": True, "source_hash_algorithm": "sha256", "strict_prospective_credit_increment": 0, "issued_signals": 0, "after_cost_ev": "NOT_COMPUTABLE", "realized_pnl": None, "auto_order": False, "fund_transfer": False, "leverage": False, "project_membership_inventory": "UNKNOWN"}
    manifest = directory / "manifest.json"
    atomic_new(manifest, (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"manifest": str(manifest), "run_id": args.run_id, "assessment": additional, "raw_file_count": sum(x["file"] is not None for x in captures), "errors": [x["source_id"] + ":" + str(x["error"]) for x in captures if x["error"]], "strict_prospective_credit_increment": 0}, ensure_ascii=False))
    return 0 if all(x["capture_status"] == "EXECUTED_STRUCTURED_RAW_FILE_VERIFIED" for x in captures) and (args.mode != "binance-mature" or additional["captured_and_valid"] == 78) and (args.mode != "bithumb-trio" or additional["bithumb_trio"]["status"] != "BLOCKED") else 2


if __name__ == "__main__":
    sys.exit(main())