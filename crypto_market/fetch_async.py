#!/usr/bin/env python3
import asyncio
import ccxt.async_support as ccxt
from typing import List, Dict, Tuple
from .models import Candle


async def fetch_ohlcv_batch(
    exchange_name: str,
    symbols: List[str],
    timeframe: str = "1d",
    limit: int = 200,
    max_concurrent: int = 10
) -> Dict[Tuple[str, str], List[Candle]]:
    """Async parallel OHLCV fetch"""
    ex = getattr(ccxt, exchange_name)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    
    async def fetch_one(symbol: str):
        async with semaphore:
            try:
                ohlcv = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)
                candles = [
                    Candle(ts=c[0], open=c[1], high=c[2], low=c[3], close=c[4], volume=c[5])
                    for c in ohlcv
                ]
                results[(symbol, timeframe)] = candles
            except Exception as e:
                pass
    
    await asyncio.gather(*[fetch_one(s) for s in symbols])
    await ex.close()
    
    return results
