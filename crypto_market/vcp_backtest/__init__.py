#!/usr/bin/env python3
"""VCP Backtest Package"""
from .config import BacktestConfig
from .fee_model import (
    calculate_net_pnl,
    apply_slippage_to_entry,
    apply_slippage_to_exit,
    calculate_position_size,
    calculate_trade_metrics,
)
from .engine import BacktestEngine, BacktestResult, Position, Trade
