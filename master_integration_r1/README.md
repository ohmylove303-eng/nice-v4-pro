# MASTER-INTEGRATION-r1 — Native source persistence and failure gates

**Status: PATCH TESTED OFFLINE; external native HTTP runner outcome must be verified separately.**

This is an evidence-only collector and **matured-outcome evaluator input layer**, not a trading bot. It does not retrospectively issue r0/v8.2 signals or generate actual fill statements. Frozen r0/v8.2 thresholds remain unchanged. No orders, leverage, asset transfer, credential, or authenticated API endpoint.

## Concrete defect fixed

Historical analyses had connector-structured data transiently observed but not durably saved. `capture_native.py` captures **actual read-only HTTPS response body bytes** from official public endpoints into new-only files (`os.O_EXCL`, `fsync`), records receive/persist clocks, SHA-256 and HTTP status, then independently rereads each file. An error is `BLOCKED`, not `0 signals`. Binance 78-subset coverage is `BLOCKED_PARTIAL_NOT_ZERO` unless every expected frozen symbol has an exact completed pair. It explicitly refuses current unfinished 1h bars.

The connected-GitHub connector bridge had already been tested separately on 2026-10-11 20:19 KST: exact connector `structuredContent.result` readback for 2 completed 15m BTC bars succeeded. **That earlier bridge is not native HTTP-byte verification.** On 2026-10-11 20:2x KST, 78/78 mature 19:00-20:00 Spot1h connector arrays were archived and deterministic research replay found SENT +3.081877% and CHIP +4.289711% in the fixed Binance subset; reported r0 candidate intersection 1/2. Neither is full project recall or after-cost actual PnL.

## Run (network-enabled runner; Python 3.10+)

```bash
python -m unittest -v test_capture_native
python capture_native.py --mode bithumb-trio --market KRW-BTC --run-id BITHUMB_smoke_001 --out ./evidence
python capture_native.py --mode verify-manifest --manifest ./evidence/BITHUMB_smoke_001/manifest.json
```

For research-only retrieval of fully **matured** 19:00–20:00 KST cohort 2026-10-11 (two completed Spot1h bars 18:00 and 19:00 KST, not new issuance):

```bash
python capture_native.py --mode binance-mature \
  --prefreeze ./r0_prefreeze_20261011_1930KST.json \
  --start-ms 1791709200000 \
  --run-id BINANCE_1930_mature_001 --out ./evidence
python capture_native.py --mode verify-manifest --manifest ./evidence/BINANCE_1930_mature_001/manifest.json
```

Run IDs are intentionally unique, directories cannot be reused, existing raw files cannot be overwritten, and errors cannot silently become no-signal. Native response bytes are stored in `evidence/<run_id>/raw/*.http-body` with `manifest.json`. **These files are not proof of actual signal delivery or native order fills**; external runner logs and later independent readback must be verified for stronger lineage.

The `Bithumb` smoke requires `market/all`, `ticker`, `trades/ticks` and `orderbook` responses under **one run ID**, with source timestamps and plausible fresh spread; otherwise `BLOCKED`. Even when freshness passes, unknown fee, order amount, exit orderbook and future path keep `after_cost_EV=NOT_COMPUTABLE`, `realized_pnl=null`. Static same-run orderbook snapshots must not be mistaken for an executable future exit.

## No success inflation

- **Synthetic tests**: run and pass offline; they are not proof of exchange connectivity.
- **Connector-structured historic replay**: separately stored GitHub evidence, 78/78 in the Binance comparative subset; not native HTTP bytes or full Bithumb KRW market.
- **Native HTTP capture on GitHub runner**: depends on successful CI run and downloadable artifact + receipt. No claim until observed.
- **r0 strict forward cohort credit**: remains `0` unless prefreeze, complete source lineage, durable issued_at/persisted_at and delivery_ack are supported by independent contemporaneous evidence. Never award historical retro-credit.
- **20 prospective cohorts**: minimum, not auto-promotion; potential better recall is irrelevant when false positives/after-cost EV/drawdown deteriorate or remain unknown.
- **Bithumb full-KRW recall and matched after-cost EV**: `NOT_COMPUTABLE` until full point-in-time market universe, native trio, size/fee, entry/exit/fill constraints and auditable forward pairing are available.
- **Complete project conversations**: `UNKNOWN` until independent membership inventory and per-conversation raw turns/attachments/last cursor exist.

GitHub test workflow is **push/PR smoke only**, not a scheduled strategy or deployment. No background collector is asserted running without a successful workflow/runner response.