#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator - Central coordination for all system components

Responsibilities:
1. Schedule periodic scans (VCP 4h/1d)
2. Coordinate Market Gate checks
3. Manage signal deduplication and cooldowns
4. Trigger notifications
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .config import ScannerCfg
from .market_gate import run_market_gate_sync, MarketGateResult
from .run_scan import run_vcp_scan
from .storage import make_engine, get_state, upsert_state, get_recent_signals
from .models import SignalEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central orchestrator for the crypto analysis system.
    
    Manages:
    - Periodic VCP scans
    - Market Gate checks
    - Signal deduplication
    - Cooldown management
    """
    
    def __init__(
        self,
        cfg: ScannerCfg = None,
        db_path: str = "crypto_market/signals.sqlite3",
        scan_interval_minutes: int = 60
    ):
        self.cfg = cfg or ScannerCfg()
        self.db_path = db_path
        self.scan_interval = scan_interval_minutes * 60  # Convert to seconds
        self.engine = make_engine(db_path)
        self.last_gate_result: Optional[MarketGateResult] = None
        self.is_running = False
    
    def check_market_gate(self) -> MarketGateResult:
        """Check current market conditions"""
        logger.info("Checking Market Gate...")
        result = run_market_gate_sync()
        self.last_gate_result = result
        logger.info(f"Market Gate: {result.gate} (Score: {result.score})")
        return result
    
    def should_scan(self) -> bool:
        """Determine if we should run a scan based on Market Gate"""
        if self.last_gate_result is None:
            self.check_market_gate()
        
        gate = self.last_gate_result.gate
        
        if gate == "RED":
            logger.info("Market Gate RED - skipping scan")
            return False
        
        return True
    
    def is_in_cooldown(self, signal: SignalEvent) -> bool:
        """Check if a signal is in cooldown period"""
        state = get_state(self.engine, signal.dedupe_key)
        if not state:
            return False
        
        now_ts = int(time.time() * 1000)
        cooldown_until = state.get("cooldown_until_ts", 0)
        
        return now_ts < cooldown_until
    
    def apply_cooldown(self, signal: SignalEvent):
        """Apply cooldown after publishing a signal"""
        now_ts = int(time.time() * 1000)
        
        if signal.signal_type == "BREAKOUT":
            cooldown_hours = self.cfg.cooldown_hours_breakout
        else:
            cooldown_hours = self.cfg.cooldown_hours_retest
        
        cooldown_until = now_ts + (cooldown_hours * 60 * 60 * 1000)
        today = datetime.now().strftime("%Y-%m-%d")
        
        upsert_state(
            self.engine,
            dedupe_key=signal.dedupe_key,
            last_notified_ts=now_ts,
            cooldown_until_ts=cooldown_until,
            last_symbol_day=f"{signal.symbol}_{today}"
        )
    
    def filter_signals(self, signals: List[SignalEvent]) -> List[SignalEvent]:
        """Filter signals by quality, cooldown, and daily limits"""
        filtered = []
        
        for sig in signals:
            # Check cooldown
            if self.is_in_cooldown(sig):
                logger.debug(f"Skipping {sig.symbol} - in cooldown")
                continue
            
            # Check liquidity
            if sig.liquidity_bucket not in self.cfg.allow_liquidity:
                if sig.score < self.cfg.liquidity_exception_score:
                    logger.debug(f"Skipping {sig.symbol} - low liquidity")
                    continue
            
            filtered.append(sig)
        
        return filtered
    
    def run_scan_cycle(self) -> Dict[str, Any]:
        """Run a complete scan cycle"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "gate": None,
            "signals_4h": [],
            "signals_1d": [],
            "skipped": False
        }
        
        # 1. Check Market Gate
        gate_result = self.check_market_gate()
        results["gate"] = {
            "status": gate_result.gate,
            "score": gate_result.score,
            "reasons": gate_result.reasons
        }
        
        # 2. Decide if we should scan
        if not self.should_scan():
            results["skipped"] = True
            return results
        
        # 3. Run 4h scan
        logger.info("Running 4h VCP scan...")
        signals_4h = run_vcp_scan(self.cfg, timeframe="4h", save_to_db=True, db_path=self.db_path)
        filtered_4h = self.filter_signals(signals_4h)
        results["signals_4h"] = [
            {"symbol": s.symbol, "score": s.score, "type": s.signal_type}
            for s in filtered_4h
        ]
        
        # 4. Run 1d scan
        logger.info("Running 1d VCP scan...")
        signals_1d = run_vcp_scan(self.cfg, timeframe="1d", save_to_db=True, db_path=self.db_path)
        filtered_1d = self.filter_signals(signals_1d)
        results["signals_1d"] = [
            {"symbol": s.symbol, "score": s.score, "type": s.signal_type}
            for s in filtered_1d
        ]
        
        # 5. Apply cooldowns
        for sig in filtered_4h + filtered_1d:
            self.apply_cooldown(sig)
        
        logger.info(f"Scan complete: {len(filtered_4h)} 4h signals, {len(filtered_1d)} 1d signals")
        
        return results
    
    def run_loop(self):
        """Run continuous scan loop"""
        self.is_running = True
        logger.info(f"Starting orchestrator loop (interval: {self.scan_interval}s)")
        
        while self.is_running:
            try:
                results = self.run_scan_cycle()
                logger.info(f"Cycle complete: {results}")
            except Exception as e:
                logger.error(f"Scan cycle failed: {e}")
            
            # Wait for next cycle
            logger.info(f"Sleeping for {self.scan_interval}s...")
            time.sleep(self.scan_interval)
    
    def stop(self):
        """Stop the orchestrator loop"""
        self.is_running = False
        logger.info("Orchestrator stopped")


def run_orchestrator(interval_minutes: int = 60):
    """Run the orchestrator"""
    orch = Orchestrator(scan_interval_minutes=interval_minutes)
    try:
        orch.run_loop()
    except KeyboardInterrupt:
        orch.stop()


if __name__ == "__main__":
    run_orchestrator(interval_minutes=60)
