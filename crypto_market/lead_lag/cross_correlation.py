#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Correlation Analysis
Measures similarity and lag between two time series
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CrossCorrelationResult:
    """Result of cross-correlation analysis"""
    series1: str
    series2: str
    max_lag: int
    best_lag: int
    best_correlation: float
    correlations: Dict[int, float]
    
    def get_interpretation(self) -> str:
        direction = "positive" if self.best_correlation > 0 else "negative"
        strength = abs(self.best_correlation)
        
        if strength > 0.7:
            strength_text = "strong"
        elif strength > 0.4:
            strength_text = "moderate"
        else:
            strength_text = "weak"
        
        return f"{self.series1} has {strength_text} {direction} correlation with {self.series2} at lag {self.best_lag}"


def cross_correlation(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 12
) -> CrossCorrelationResult:
    """
    Calculate cross-correlation between two series at different lags.
    
    Positive lag means series1 leads series2.
    Negative lag means series2 leads series1.
    
    Args:
        series1: First time series
        series2: Second time series
        max_lag: Maximum lag to test (both positive and negative)
        
    Returns:
        CrossCorrelationResult with correlations at each lag
    """
    correlations = {}
    
    # Align series
    aligned = pd.concat([series1, series2], axis=1).dropna()
    if len(aligned) < max_lag * 2:
        return CrossCorrelationResult(
            series1=series1.name or "series1",
            series2=series2.name or "series2",
            max_lag=max_lag,
            best_lag=0,
            best_correlation=0.0,
            correlations={}
        )
    
    s1 = aligned.iloc[:, 0]
    s2 = aligned.iloc[:, 1]
    
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = s1.iloc[:-lag].corr(s2.iloc[lag:])
        elif lag < 0:
            corr = s1.iloc[-lag:].corr(s2.iloc[:lag])
        else:
            corr = s1.corr(s2)
        
        if not np.isnan(corr):
            correlations[lag] = float(corr)
    
    # Find best lag
    if correlations:
        best_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_lag]
    else:
        best_lag = 0
        best_corr = 0.0
    
    return CrossCorrelationResult(
        series1=series1.name or "series1",
        series2=series2.name or "series2",
        max_lag=max_lag,
        best_lag=best_lag,
        best_correlation=best_corr,
        correlations=correlations
    )


def find_leading_indicators(
    df: pd.DataFrame,
    target: str,
    variables: Optional[List[str]] = None,
    max_lag: int = 12,
    min_correlation: float = 0.3
) -> List[CrossCorrelationResult]:
    """
    Find variables that have strong cross-correlation with target.
    
    Args:
        df: DataFrame with time series
        target: Target variable name
        variables: Variables to test (default: all except target)
        max_lag: Maximum lag to test
        min_correlation: Minimum abs correlation to include
        
    Returns:
        List of CrossCorrelationResults sorted by correlation strength
    """
    if target not in df.columns:
        raise ValueError(f"Target {target} not in DataFrame")
    
    if variables is None:
        variables = [c for c in df.columns if c != target]
    
    results = []
    target_series = df[target]
    
    for var in variables:
        if var == target or var not in df.columns:
            continue
        
        result = cross_correlation(df[var], target_series, max_lag)
        
        if abs(result.best_correlation) >= min_correlation:
            results.append(result)
    
    # Sort by absolute correlation
    results.sort(key=lambda r: abs(r.best_correlation), reverse=True)
    
    return results
