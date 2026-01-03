#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCP Scanner - Run Scan Module
Main entry point for running VCP scans
"""
import asyncio
from typing import List, Tuple, Dict
from datetime import datetime

from .config import ScannerCfg
from .models import Candle, SignalEvent
from .universe import build_universe_binance_usdt
from .fetch_async import fetch_ohlcv_batch
from .signals import detect_setups, detect_breakouts, detect_retests
from .scoring import score_batch
from .storage import make_engine, insert_signal, get_recent_signals


def run_vcp_scan(
    cfg: ScannerCfg = None,
    timeframe: str = "4h",
    save_to_db: bool = True,
    db_path: str = "crypto_market/signals.sqlite3"
) -> List[SignalEvent]:
    """
    Run a complete VCP scan cycle.
    
    Args:
        cfg: Scanner configuration
        timeframe: "4h" or "1d"
        save_to_db: Whether to save signals to SQLite
        db_path: Path to SQLite database
        
    Returns:
        List of detected signals
    """
    if cfg is None:
        cfg = ScannerCfg()
    
    tf_cfg = cfg.tf_4h if timeframe == "4h" else cfg.tf_1d
    
    print(f"[{datetime.now().isoformat()}] Starting VCP scan for {timeframe}...")
    
    # 1. Build universe
    import ccxt
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    
    print(f"  → Fetching universe (top {cfg.universe_top_n} by volume)...")
    symbols_with_qv = build_universe_binance_usdt(
        exchange=exchange,
        top_n=cfg.universe_top_n,
        min_quote_vol_usdt=cfg.min_quote_volume_usdt
    )
    print(f"  → Found {len(symbols_with_qv)} symbols")
    
    # 2. Fetch OHLCV data
    print(f"  → Fetching OHLCV data...")
    symbols = [s[0] for s in symbols_with_qv]
    symbols.append("BTC/USDT")  # Always include BTC for regime detection
    
    candles_map = asyncio.run(fetch_ohlcv_batch(
        exchange_name="binance",
        symbols=symbols,
        timeframe=timeframe,
        limit=tf_cfg.limit
    ))
    print(f"  → Fetched data for {len(candles_map)} symbol/timeframe pairs")
    
    # 3. Get BTC candles for regime detection
    btc_candles = candles_map.get(("BTC/USDT", timeframe), [])
    
    # 4. Detect setups
    print(f"  → Detecting VCP setups...")
    setups = detect_setups(
        exchange_name=cfg.exchange,
        symbols_with_qv=symbols_with_qv,
        candles_map=candles_map,
        btc_candles=btc_candles,
        tf=tf_cfg
    )
    print(f"  → Found {len(setups)} VCP setups")
    
    # 5. Detect breakouts
    print(f"  → Detecting breakouts...")
    breakouts = detect_breakouts(setups, candles_map, tf_cfg)
    print(f"  → Found {len(breakouts)} breakout signals")
    
    # 6. Score signals
    print(f"  → Scoring signals...")
    scored = score_batch(breakouts, tf_cfg)
    
    # 7. Filter by minimum score
    min_score = cfg.min_score_4h if timeframe == "4h" else cfg.min_score_1d
    filtered = [s for s in scored if s.score >= min_score]
    print(f"  → {len(filtered)} signals passed score threshold ({min_score}+)")
    
    # 8. Save to database
    if save_to_db and filtered:
        print(f"  → Saving to database...")
        engine = make_engine(db_path)
        for sig in filtered:
            sig.event_id = f"{sig.symbol}_{sig.timeframe}_{sig.event_ts}"
            sig.dedupe_key = f"{sig.symbol}_{sig.timeframe}_{sig.signal_type}"
            insert_signal(engine, sig)
        print(f"  → Saved {len(filtered)} signals")
    
    print(f"[{datetime.now().isoformat()}] Scan complete!")
    return filtered


def run_full_scan(save_to_db: bool = True) -> Dict[str, List[SignalEvent]]:
    """Run scans for both 4h and 1d timeframes"""
    results = {}
    
    for tf in ["4h", "1d"]:
        signals = run_vcp_scan(timeframe=tf, save_to_db=save_to_db)
        results[tf] = signals
    
    return results


if __name__ == "__main__":
    results = run_full_scan()
    
    print("\n=== SCAN RESULTS ===")
    for tf, signals in results.items():
        print(f"\n{tf}:")
        for s in signals[:5]:  # Show top 5
            print(f"  {s.symbol} | {s.signal_type} | Score: {s.score} | Grade: {s.market_regime}")
