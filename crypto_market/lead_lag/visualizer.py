#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizer - Data visualization utilities
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass 
class ChartData:
    """Chart data for frontend rendering"""
    chart_type: str  # line, bar, scatter, heatmap
    title: str
    labels: List[str]
    datasets: List[Dict]
    options: Optional[Dict] = None
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.chart_type,
            "title": self.title,
            "data": {
                "labels": self.labels,
                "datasets": self.datasets
            },
            "options": self.options or {}
        })


def create_correlation_heatmap(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None
) -> ChartData:
    """
    Create correlation heatmap data for Chart.js Matrix plugin.
    
    Args:
        df: DataFrame with time series
        variables: Variables to include (default: all)
        
    Returns:
        ChartData for heatmap rendering
    """
    if variables is None:
        variables = list(df.columns)
    
    subset = df[variables].dropna()
    corr_matrix = subset.corr()
    
    # Convert to heatmap format
    data = []
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            data.append({
                "x": j,
                "y": i,
                "v": round(corr_matrix.loc[var1, var2], 2)
            })
    
    return ChartData(
        chart_type="matrix",
        title="Correlation Matrix",
        labels=variables,
        datasets=[{
            "data": data,
            "backgroundColor": "rgba(75, 192, 192, 0.6)"
        }]
    )


def create_lag_chart(
    cross_corr_results: List[Dict],
    max_items: int = 10
) -> ChartData:
    """
    Create lag analysis bar chart.
    
    Args:
        cross_corr_results: Cross-correlation results
        max_items: Maximum number of items to show
        
    Returns:
        ChartData for bar chart
    """
    results = cross_corr_results[:max_items]
    
    labels = [r.get("series1", "?") for r in results]
    lags = [r.get("best_lag", 0) for r in results]
    correlations = [r.get("best_correlation", 0) for r in results]
    
    return ChartData(
        chart_type="bar",
        title="Leading Indicators by Lag",
        labels=labels,
        datasets=[
            {
                "label": "Lag (months)",
                "data": lags,
                "backgroundColor": "rgba(54, 162, 235, 0.6)"
            },
            {
                "label": "Correlation",
                "data": correlations,
                "backgroundColor": "rgba(255, 99, 132, 0.6)"
            }
        ]
    )


def create_equity_curve(
    equity_data: pd.Series,
    benchmark: Optional[pd.Series] = None
) -> ChartData:
    """
    Create equity curve line chart.
    
    Args:
        equity_data: Equity values over time
        benchmark: Optional benchmark to compare
        
    Returns:
        ChartData for line chart
    """
    datasets = [{
        "label": "Portfolio",
        "data": [round(float(v), 2) for v in equity_data.values],
        "borderColor": "rgba(75, 192, 192, 1)",
        "fill": False
    }]
    
    if benchmark is not None:
        datasets.append({
            "label": "Benchmark",
            "data": [round(float(v), 2) for v in benchmark.values],
            "borderColor": "rgba(153, 102, 255, 1)",
            "fill": False
        })
    
    return ChartData(
        chart_type="line",
        title="Equity Curve",
        labels=[str(ts) for ts in equity_data.index],
        datasets=datasets
    )


def create_market_gate_gauge(score: int, gate: str) -> Dict:
    """
    Create Market Gate gauge data.
    
    Args:
        score: Gate score (0-100)
        gate: Gate status (GREEN/YELLOW/RED)
        
    Returns:
        Dict for gauge rendering
    """
    colors = {
        "GREEN": "#10b981",
        "YELLOW": "#f59e0b",
        "RED": "#ef4444"
    }
    
    return {
        "score": score,
        "max": 100,
        "gate": gate,
        "color": colors.get(gate, "#6b7280"),
        "thresholds": [
            {"value": 48, "color": "#ef4444"},
            {"value": 72, "color": "#f59e0b"},
            {"value": 100, "color": "#10b981"}
        ]
    }


def create_vcp_table_data(signals: List[Dict]) -> List[Dict]:
    """
    Format VCP signals for table display.
    
    Args:
        signals: List of signal dicts
        
    Returns:
        Formatted table data
    """
    return [
        {
            "symbol": s.get("symbol", "?"),
            "timeframe": s.get("timeframe", "?"),
            "type": s.get("signal_type", "?"),
            "score": s.get("score", 0),
            "grade": s.get("liquidity_bucket", "?"),
            "pivot": f"${s.get('pivot_high', 0):,.2f}",
            "breakout_pct": f"{s.get('breakout_close_pct', 0):.1f}%",
            "timestamp": s.get("event_ts", 0)
        }
        for s in signals
    ]
