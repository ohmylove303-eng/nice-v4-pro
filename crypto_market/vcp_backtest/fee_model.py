#!/usr/bin/env python3
"""VCP Backtest Fee Model"""


def calculate_net_pnl(entry_price: float, exit_price: float, quantity: float, config) -> tuple:
    """
    Calculate net PnL after fees.
    
    Returns:
        (gross_pnl, net_pnl, total_fees)
    """
    entry_value = entry_price * quantity
    entry_commission = entry_value * (config.commission_pct / 100)
    exit_value = exit_price * quantity
    exit_commission = exit_value * (config.commission_pct / 100)
    gross_pnl = exit_value - entry_value
    total_fees = entry_commission + exit_commission
    net_pnl = gross_pnl - total_fees
    return gross_pnl, net_pnl, total_fees


def apply_slippage_to_entry(price: float, config) -> float:
    """Apply slippage to entry price (unfavorable direction)"""
    return price * (1 + config.slippage_pct / 100)


def apply_slippage_to_exit(price: float, is_stop: bool, config) -> float:
    """Apply slippage to exit price (unfavorable direction for stops)"""
    if is_stop:
        return price * (1 - config.slippage_pct / 100)
    return price * (1 - config.slippage_pct / 200)


def calculate_position_size(capital: float, risk_pct: float, entry_price: float, stop_price: float) -> float:
    """
    Calculate position size based on risk amount.
    
    Args:
        capital: Total capital
        risk_pct: Percentage of capital to risk (e.g., 1.0 for 1%)
        entry_price: Entry price
        stop_price: Stop loss price
        
    Returns:
        Quantity to buy
    """
    if entry_price <= stop_price:
        return 0.0
    
    risk_per_unit = entry_price - stop_price
    risk_amount = capital * (risk_pct / 100)
    quantity = risk_amount / risk_per_unit
    
    return quantity


def calculate_trade_metrics(trades: list) -> dict:
    """
    Calculate aggregate metrics from a list of trades.
    
    Args:
        trades: List of trade dicts with 'net_pnl', 'gross_pnl', 'hold_bars', etc.
        
    Returns:
        Dictionary with metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "avg_hold_bars": 0.0,
        }
    
    winners = [t for t in trades if t.get("net_pnl", 0) > 0]
    losers = [t for t in trades if t.get("net_pnl", 0) <= 0]
    
    total_win = sum(t.get("net_pnl", 0) for t in winners)
    total_loss = abs(sum(t.get("net_pnl", 0) for t in losers))
    
    return {
        "total_trades": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0.0,
        "avg_win": total_win / len(winners) if winners else 0.0,
        "avg_loss": total_loss / len(losers) if losers else 0.0,
        "profit_factor": total_win / total_loss if total_loss > 0 else float('inf'),
        "total_pnl": sum(t.get("net_pnl", 0) for t in trades),
        "avg_hold_bars": sum(t.get("hold_bars", 0) for t in trades) / len(trades) if trades else 0.0,
    }
