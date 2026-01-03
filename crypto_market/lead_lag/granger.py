#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lead-Lag Analysis - Granger Causality Module
Tests whether one time series has predictive power over another.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GrangerResult:
    """Result of Granger Causality test"""
    cause: str
    effect: str
    max_lag: int
    best_lag: int
    best_p_value: float
    is_significant: bool
    all_lags: Dict[int, Dict]
    
    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "best_lag": int(self.best_lag),
            "best_p_value": round(float(self.best_p_value), 4),
            "is_significant": bool(self.is_significant),
            "interpretation": self.get_interpretation()
        }
    
    def get_interpretation(self, lang: str = "ko") -> str:
        if lang == "ko":
            if self.is_significant:
                return f"{self.cause}은(는) {self.effect}을(를) {self.best_lag}기간 선행하여 예측 가능 (p={self.best_p_value:.4f})"
            else:
                return f"{self.cause}은(는) {self.effect}에 대한 예측력이 통계적으로 유의하지 않음"
        else:
            if self.is_significant:
                return f"{self.cause} Granger-causes {self.effect} at lag {self.best_lag} (p={self.best_p_value:.4f})"
            else:
                return f"{self.cause} does not Granger-cause {self.effect}"


def granger_causality_test(
    df: pd.DataFrame,
    cause_var: str,
    effect_var: str,
    max_lag: int = 6,
    significance_level: float = 0.05
) -> GrangerResult:
    """
    Perform Granger Causality test.
    
    H0: cause_var does NOT Granger-cause effect_var
    H1: cause_var Granger-causes effect_var
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return GrangerResult(
            cause=cause_var, effect=effect_var, max_lag=max_lag,
            best_lag=0, best_p_value=1.0, is_significant=False, all_lags={}
        )
    
    if cause_var not in df.columns or effect_var not in df.columns:
        raise ValueError(f"Variables not found: {cause_var}, {effect_var}")
    
    test_data = df[[effect_var, cause_var]].dropna()
    
    if len(test_data) < max_lag * 3:
        return GrangerResult(
            cause=cause_var, effect=effect_var, max_lag=max_lag,
            best_lag=0, best_p_value=1.0, is_significant=False, all_lags={}
        )
    
    try:
        results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        
        all_lags = {}
        for lag, result in results.items():
            f_test = result[0]['ssr_ftest']
            f_stat = f_test[0]
            p_value = f_test[1]
            
            all_lags[lag] = {
                "f_statistic": round(f_stat, 4),
                "p_value": round(p_value, 4),
                "is_significant": p_value < significance_level
            }
        
        best_lag = min(all_lags.keys(), key=lambda k: all_lags[k]['p_value'])
        best_p = all_lags[best_lag]['p_value']
        is_sig = best_p < significance_level
        
        return GrangerResult(
            cause=cause_var, effect=effect_var, max_lag=max_lag,
            best_lag=best_lag, best_p_value=best_p, is_significant=is_sig, all_lags=all_lags
        )
        
    except Exception as e:
        logger.error(f"Granger test failed: {e}")
        return GrangerResult(
            cause=cause_var, effect=effect_var, max_lag=max_lag,
            best_lag=0, best_p_value=1.0, is_significant=False, all_lags={}
        )


def bidirectional_granger_test(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    max_lag: int = 6
) -> Tuple[GrangerResult, GrangerResult]:
    """Test causality in both directions"""
    r1 = granger_causality_test(df, var1, var2, max_lag)
    r2 = granger_causality_test(df, var2, var1, max_lag)
    return r1, r2


def find_granger_causal_indicators(
    df: pd.DataFrame,
    target: str,
    variables: Optional[List[str]] = None,
    max_lag: int = 6
) -> List[GrangerResult]:
    """
    Find all variables that Granger-cause the target.
    """
    if variables is None:
        variables = [c for c in df.columns if c != target]
    
    significant_results = []
    
    for var in variables:
        if var == target:
            continue
        
        try:
            result = granger_causality_test(df, var, target, max_lag)
            if result.is_significant:
                significant_results.append(result)
        except Exception as e:
            logger.warning(f"Failed to test {var}: {e}")
    
    significant_results.sort(key=lambda r: r.best_p_value)
    
    return significant_results
