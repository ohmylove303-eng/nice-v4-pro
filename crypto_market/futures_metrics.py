#!/usr/bin/env python3
"""
Futures Metrics - Funding Rate & Open Interest from Binance Futures
"""
from __future__ import annotations
from typing import Optional, Tuple
import ccxt.async_support as ccxt


async def fetch_btc_funding_and_oi() -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch BTC funding rate and OI from Binance USDT-M Futures.
    
    Returns:
        (funding_rate, oi_z_score)
        - funding_rate: 0.0001 = 0.01%, 0.001 = 0.1%
    """
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    
    try:
        funding = None
        try:
            fr = await ex.fetch_funding_rate("BTC/USDT:USDT")
            funding = fr.get("fundingRate", None)
            if funding is not None:
                funding = float(funding)
        except Exception:
            try:
                fr = await ex.fetch_funding_rate("BTC/USDT")
                funding = fr.get("fundingRate", None)
                if funding is not None:
                    funding = float(funding)
            except Exception:
                funding = None

        return funding, None
    finally:
        await ex.close()


async def fetch_funding_only() -> Optional[float]:
    """Simplified: just fetch funding rate"""
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    
    try:
        try:
            fr = await ex.fetch_funding_rate("BTC/USDT:USDT")
            rate = fr.get("fundingRate", None)
            return float(rate) if rate else None
        except Exception:
            return None
    finally:
        await ex.close()
