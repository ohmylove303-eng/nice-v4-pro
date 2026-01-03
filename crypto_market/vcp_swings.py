#!/usr/bin/env python3
"""VCP Swing Point Extraction"""
import pandas as pd
from typing import Optional, Dict


def extract_vcp_from_swings(
    df: pd.DataFrame,
    k: int = 3,
    lookback: int = 200,
    min_r12: float = 1.2,
    min_r23: float = 1.1,
    require_descending_highs: bool = True,
    require_ascending_lows: bool = True,
) -> Optional[Dict]:
    """
    Extract VCP pattern from swing highs/lows
    """
    if len(df) < lookback:
        return None
    
    df = df.tail(lookback).copy()
    high = df["high"]
    low = df["low"]
    
    # Find swing highs
    swing_highs = []
    swing_lows = []
    
    for i in range(k, len(df) - k):
        if high.iloc[i] == high.iloc[i-k:i+k+1].max():
            swing_highs.append((i, high.iloc[i]))
        if low.iloc[i] == low.iloc[i-k:i+k+1].min():
            swing_lows.append((i, low.iloc[i]))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    
    # Calculate contractions
    contractions = []
    for i in range(min(len(swing_highs), len(swing_lows))):
        h = swing_highs[i][1] if i < len(swing_highs) else swing_highs[-1][1]
        l = swing_lows[i][1] if i < len(swing_lows) else swing_lows[-1][1]
        if l > 0:
            range_pct = (h - l) / l * 100
            contractions.append(range_pct)
    
    if len(contractions) < 2:
        return None
    
    c1, c2 = contractions[0], contractions[1]
    c3 = contractions[2] if len(contractions) > 2 else c2 * 0.8
    
    # Check decay ratios
    if c2 > 0 and c1 / c2 < min_r12:
        return None
    if c3 > 0 and c2 / c3 < min_r23:
        return None
    
    # Check structure if required
    if require_descending_highs:
        highs_only = [h[1] for h in swing_highs[-3:]]
        if len(highs_only) >= 2 and highs_only[-1] > highs_only[-2]:
            return None
    
    if require_ascending_lows:
        lows_only = [l[1] for l in swing_lows[-3:]]
        if len(lows_only) >= 2 and lows_only[-1] < lows_only[-2]:
            return None
    
    pivot_high = max(h[1] for h in swing_highs[-3:]) if swing_highs else high.max()
    
    return {
        "pivot_high": pivot_high,
        "c1": c1,
        "c2": c2,
        "c3": c3,
    }
