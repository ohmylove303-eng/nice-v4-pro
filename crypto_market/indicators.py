#!/usr/bin/env python3
"""Technical Indicators for Crypto VCP Analysis"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - volatility indicator"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def wick_ratio(open_: float, high: float, low: float, close: float) -> float:
    """
    Calculate wick ratio for a candle.
    Higher ratio = more wick (rejection), lower ratio = stronger close.
    """
    denom = high - low
    if denom <= 0:
        return 0.0
    return float((high - close) / denom)


def fractal_swings(df: pd.DataFrame, k: int = 3) -> list:
    """
    Fractal swing highs/lows detection.
    
    Args:
        df: DataFrame with 'high' and 'low' columns
        k: Number of bars on each side to confirm swing
        
    Returns:
        Ordered swings: [{'i': index, 'type': 'H'/'L', 'price': float}, ...]
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swings = []

    for i in range(k, n - k):
        hi = highs[i]
        lo = lows[i]

        # Swing High: higher than k bars on both sides
        if hi > np.max(highs[i-k:i]) and hi > np.max(highs[i+1:i+k+1]):
            swings.append({"i": i, "type": "H", "price": float(hi)})

        # Swing Low: lower than k bars on both sides
        if lo < np.min(lows[i-k:i]) and lo < np.min(lows[i+1:i+k+1]):
            swings.append({"i": i, "type": "L", "price": float(lo)})

    swings.sort(key=lambda x: x["i"])

    # Remove consecutive same-type (keep more extreme)
    cleaned = []
    for s in swings:
        if not cleaned:
            cleaned.append(s)
            continue
        last = cleaned[-1]
        if s["type"] != last["type"]:
            cleaned.append(s)
        else:
            if s["type"] == "H" and s["price"] >= last["price"]:
                cleaned[-1] = s
            elif s["type"] == "L" and s["price"] <= last["price"]:
                cleaned[-1] = s

    return cleaned


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(period).mean()


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average"""
    return df["volume"].rolling(period).mean()


def atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as percentage of close price"""
    atr_val = atr(df, period)
    return (atr_val / df["close"]) * 100


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple:
    """
    Bollinger Bands
    
    Returns:
        (upper_band, middle_band, lower_band)
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    MACD - Moving Average Convergence Divergence
    
    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
