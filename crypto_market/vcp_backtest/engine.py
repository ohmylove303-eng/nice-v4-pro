#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCP Backtest Engine
Simulates trading based on VCP signals with realistic execution
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

from .config import BacktestConfig
from .fee_model import (
    calculate_net_pnl,
    apply_slippage_to_entry,
    apply_slippage_to_exit,
    calculate_position_size,
    calculate_trade_metrics
)


@dataclass
class Position:
    """Active position"""
    symbol: str
    entry_ts: int
    entry_price: float
    quantity: float
    stop_price: float
    target_price: Optional[float]
    trailing_stop: Optional[float]
    entry_bar: int
    signal_score: int
    signal_grade: str


@dataclass
class Trade:
    """Completed trade"""
    symbol: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    fees: float
    exit_reason: str  # "STOP", "TARGET", "TRAILING", "MAX_HOLD", "END"
    hold_bars: int
    signal_score: int


@dataclass
class BacktestResult:
    """Complete backtest results"""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict
    start_capital: float
    end_capital: float
    
    def summary(self) -> str:
        m = self.metrics
        return f"""
=== BACKTEST SUMMARY ===
Total Trades: {m['total_trades']}
Win Rate: {m['win_rate']:.1f}%
Profit Factor: {m['profit_factor']:.2f}
Total PnL: ${m['total_pnl']:,.2f}
Return: {(self.end_capital / self.start_capital - 1) * 100:.1f}%
Avg Hold Bars: {m['avg_hold_bars']:.1f}
"""


class BacktestEngine:
    """
    VCP Backtest Engine
    
    Features:
    - Realistic entry/exit with slippage
    - Multiple stop types (fixed, pivot-based, ATR)
    - Trailing stops
    - Position sizing
    - Portfolio management
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.capital = self.config.initial_capital
        self.equity_history: List[Tuple[int, float]] = []
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        return len(self.positions) < self.config.max_concurrent_positions
    
    def calculate_stop_price(self, entry_price: float, pivot_price: float, atr: float) -> float:
        """Calculate stop loss price based on config"""
        if self.config.stop_loss_type == "FIXED_PCT":
            return entry_price * (1 - self.config.stop_loss_value / 100)
        elif self.config.stop_loss_type == "PIVOT_BASED":
            return pivot_price * (1 - self.config.stop_loss_value / 100)
        elif self.config.stop_loss_type == "ATR_MULT":
            return entry_price - (atr * self.config.stop_loss_value)
        return entry_price * 0.95  # Default 5%
    
    def calculate_position_size(self, entry_price: float, stop_price: float) -> float:
        """Calculate position size based on config"""
        max_value = self.capital * (self.config.max_position_pct / 100)
        
        if self.config.position_sizing == "EQUAL":
            value = self.capital / self.config.max_concurrent_positions
        elif self.config.position_sizing == "VOLATILITY":
            risk_per_unit = abs(entry_price - stop_price)
            risk_amount = self.capital * 0.01  # 1% risk
            value = min(risk_amount / risk_per_unit * entry_price, max_value)
        else:
            value = self.capital / self.config.max_concurrent_positions
        
        value = min(value, max_value, self.capital * 0.5)  # Never more than 50%
        return value / entry_price
    
    def open_position(
        self,
        symbol: str,
        signal_ts: int,
        signal_price: float,
        pivot_price: float,
        atr: float,
        score: int,
        grade: str,
        bar_index: int
    ) -> Optional[Position]:
        """Open a new position"""
        if not self.can_open_position():
            return None
        
        if symbol in self.positions:
            return None
        
        # Apply slippage
        entry_price = apply_slippage_to_entry(signal_price, self.config)
        
        # Calculate stops
        stop_price = self.calculate_stop_price(entry_price, pivot_price, atr)
        target_price = entry_price * (1 + self.config.take_profit_pct / 100) if self.config.take_profit_pct else None
        
        # Position size
        quantity = self.calculate_position_size(entry_price, stop_price)
        if quantity <= 0:
            return None
        
        position = Position(
            symbol=symbol,
            entry_ts=signal_ts,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            target_price=target_price,
            trailing_stop=None,
            entry_bar=bar_index,
            signal_score=score,
            signal_grade=grade
        )
        
        self.positions[symbol] = position
        self.capital -= entry_price * quantity
        
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_ts: int,
        exit_price: float,
        bar_index: int,
        reason: str
    ) -> Optional[Trade]:
        """Close an existing position"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Apply slippage
        is_stop = reason in ["STOP", "TRAILING"]
        final_exit = apply_slippage_to_exit(exit_price, is_stop, self.config)
        
        # Calculate PnL
        gross, net, fees = calculate_net_pnl(
            pos.entry_price, final_exit, pos.quantity, self.config
        )
        
        trade = Trade(
            symbol=symbol,
            entry_ts=pos.entry_ts,
            exit_ts=exit_ts,
            entry_price=pos.entry_price,
            exit_price=final_exit,
            quantity=pos.quantity,
            gross_pnl=gross,
            net_pnl=net,
            fees=fees,
            exit_reason=reason,
            hold_bars=bar_index - pos.entry_bar,
            signal_score=pos.signal_score
        )
        
        self.trades.append(trade)
        self.capital += final_exit * pos.quantity
        del self.positions[symbol]
        
        return trade
    
    def update_trailing_stops(self, symbol: str, current_high: float):
        """Update trailing stop for a position"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        if self.config.trailing_stop_pct is None:
            return
        
        new_trailing = current_high * (1 - self.config.trailing_stop_pct / 100)
        
        if pos.trailing_stop is None or new_trailing > pos.trailing_stop:
            pos.trailing_stop = new_trailing
    
    def check_exits(
        self,
        symbol: str,
        ts: int,
        high: float,
        low: float,
        close: float,
        bar_index: int
    ) -> Optional[Trade]:
        """Check all exit conditions for a position"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Check stop loss
        if low <= pos.stop_price:
            return self.close_position(symbol, ts, pos.stop_price, bar_index, "STOP")
        
        # Check trailing stop
        if pos.trailing_stop and low <= pos.trailing_stop:
            return self.close_position(symbol, ts, pos.trailing_stop, bar_index, "TRAILING")
        
        # Check target
        if pos.target_price and high >= pos.target_price:
            return self.close_position(symbol, ts, pos.target_price, bar_index, "TARGET")
        
        # Check max hold
        if self.config.max_hold_bars:
            hold = bar_index - pos.entry_bar
            if hold >= self.config.max_hold_bars:
                return self.close_position(symbol, ts, close, bar_index, "MAX_HOLD")
        
        # Update trailing
        self.update_trailing_stops(symbol, high)
        
        return None
    
    def get_results(self) -> BacktestResult:
        """Get complete backtest results"""
        metrics = calculate_trade_metrics([
            {"net_pnl": t.net_pnl, "hold_bars": t.hold_bars}
            for t in self.trades
        ])
        
        equity_curve = pd.Series(
            [eq for _, eq in self.equity_history],
            index=[ts for ts, _ in self.equity_history]
        )
        
        return BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=equity_curve,
            metrics=metrics,
            start_capital=self.config.initial_capital,
            end_capital=self.capital
        )
