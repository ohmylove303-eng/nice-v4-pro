#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lead-Lag Analysis - Data Fetcher Module
Fetches macro indicators from FRED, yfinance for analysis.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    ticker: str
    source: str  # 'yfinance', 'fred'
    description: str
    frequency: str  # 'daily', 'weekly', 'monthly'
    transform: str = 'price'


MARKET_SOURCES = [
    DataSource("BTC", "BTC-USD", "yfinance", "Bitcoin Price", "daily"),
    DataSource("ETH", "ETH-USD", "yfinance", "Ethereum Price", "daily"),
    DataSource("SPY", "SPY", "yfinance", "S&P 500 ETF", "daily"),
    DataSource("QQQ", "QQQ", "yfinance", "NASDAQ 100 ETF", "daily"),
    DataSource("DXY", "DX-Y.NYB", "yfinance", "US Dollar Index", "daily"),
    DataSource("GOLD", "GC=F", "yfinance", "Gold Futures", "daily"),
    DataSource("TLT", "TLT", "yfinance", "20+ Year Treasury Bond ETF", "daily"),
    DataSource("VIX", "^VIX", "yfinance", "Volatility Index", "daily"),
    DataSource("TNX", "^TNX", "yfinance", "10-Year Treasury Yield", "daily"),
    DataSource("OIL", "CL=F", "yfinance", "Crude Oil Futures", "daily"),
]


def fetch_yfinance_data(sources: List[DataSource], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch market data from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    
    tickers = [s.ticker for s in sources]
    names = [s.name for s in sources]
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return pd.DataFrame()
        
        if 'Close' in data.columns:
            if len(tickers) == 1:
                result = data['Close'].to_frame()
                result.columns = names
            else:
                result = data['Close']
                result.columns = names
        else:
            result = data
        
        return result
        
    except Exception as e:
        logger.error(f"yfinance fetch error: {e}")
        return pd.DataFrame()


def fetch_all_data(
    start_date: str = "2018-01-01",
    end_date: str = None,
    resample: str = "monthly",
    include_derivatives: bool = True
) -> pd.DataFrame:
    """
    Fetch all data sources and combine.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    market_df = fetch_yfinance_data(MARKET_SOURCES, start_date, end_date)
    
    if market_df.empty:
        return pd.DataFrame()
    
    combined = market_df
    
    # Resample
    if resample == "monthly":
        combined = combined.resample('M').last()
    elif resample == "weekly":
        combined = combined.resample('W').last()
    
    # Add derivatives (MoM, YoY, 3M)
    if include_derivatives:
        derivative_cols = {}
        
        price_cols = ['BTC', 'ETH', 'SPY', 'QQQ', 'DXY', 'GOLD', 'TLT', 'OIL', 'VIX', 'TNX']
        for col in price_cols:
            if col in combined.columns:
                derivative_cols[f'{col}_MoM'] = combined[col].pct_change(1) * 100
                derivative_cols[f'{col}_YoY'] = combined[col].pct_change(12) * 100
                derivative_cols[f'{col}_3M'] = combined[col].pct_change(3) * 100
        
        for col_name, series in derivative_cols.items():
            combined[col_name] = series
    
    combined = combined.ffill().dropna()
    
    return combined
