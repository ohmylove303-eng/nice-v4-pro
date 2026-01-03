#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Interpreter - AI-powered analysis interpretation
Uses Gemini API for generating insights from data
"""
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisInsight:
    """AI-generated insight"""
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    confidence: str  # high, medium, low


def get_gemini_client():
    """Get Gemini API client"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set")
            return None
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-pro")
    except ImportError:
        logger.warning("google-generativeai not installed")
        return None


def interpret_granger_results(results: List[Dict]) -> AnalysisInsight:
    """
    Interpret Granger causality results using LLM.
    
    Args:
        results: List of Granger test results
        
    Returns:
        AnalysisInsight with AI interpretation
    """
    if not results:
        return AnalysisInsight(
            summary="분석할 수 있는 유의한 선행 지표가 없습니다.",
            key_findings=[],
            recommendations=["더 긴 시계열 데이터를 수집하세요."],
            confidence="low"
        )
    
    model = get_gemini_client()
    
    if model is None:
        # Fallback to rule-based interpretation
        return _rule_based_interpretation(results)
    
    try:
        prompt = _build_granger_prompt(results)
        response = model.generate_content(prompt)
        return _parse_llm_response(response.text)
    except Exception as e:
        logger.error(f"LLM interpretation failed: {e}")
        return _rule_based_interpretation(results)


def interpret_market_gate(gate_result: Dict) -> AnalysisInsight:
    """
    Interpret Market Gate results using LLM.
    """
    model = get_gemini_client()
    
    if model is None:
        return _rule_based_gate_interpretation(gate_result)
    
    try:
        prompt = f"""
다음 비트코인 시장 상태 분석 결과를 해석해주세요:

- Gate: {gate_result.get('gate', 'N/A')}
- Score: {gate_result.get('score', 0)}/100
- Reasons: {', '.join(gate_result.get('reasons', []))}

한국어로 다음을 포함해 답변해주세요:
1. 현재 시장 상태 요약 (1-2문장)
2. 주요 발견사항 (3개)
3. 투자 권장사항 (2개)
"""
        response = model.generate_content(prompt)
        return _parse_llm_response(response.text)
    except Exception as e:
        logger.error(f"LLM interpretation failed: {e}")
        return _rule_based_gate_interpretation(gate_result)


def _build_granger_prompt(results: List[Dict]) -> str:
    """Build prompt for Granger interpretation"""
    findings = []
    for r in results[:5]:
        findings.append(
            f"- {r.get('cause', 'Unknown')} → BTC: lag {r.get('best_lag', 0)}, p-value {r.get('best_p_value', 1.0):.4f}"
        )
    
    return f"""
다음 Granger Causality 분석 결과를 해석해주세요:

{chr(10).join(findings)}

한국어로 다음을 포함해 답변해주세요:
1. 분석 결과 요약 (1-2문장)
2. 주요 발견사항 (3개 bullet point)
3. 투자 권장사항 (2개 bullet point)
"""


def _parse_llm_response(text: str) -> AnalysisInsight:
    """Parse LLM response into structured insight"""
    lines = text.strip().split('\n')
    
    summary = ""
    findings = []
    recommendations = []
    
    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if '요약' in line or '결과' in line:
            section = 'summary'
            continue
        elif '발견' in line or '주요' in line:
            section = 'findings'
            continue
        elif '권장' in line or '추천' in line:
            section = 'recommendations'
            continue
        
        if section == 'summary':
            summary += line + " "
        elif section == 'findings' and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
            findings.append(line.lstrip('-•0123456789. '))
        elif section == 'recommendations' and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
            recommendations.append(line.lstrip('-•0123456789. '))
    
    return AnalysisInsight(
        summary=summary.strip() or "분석 완료",
        key_findings=findings[:5],
        recommendations=recommendations[:3],
        confidence="medium"
    )


def _rule_based_interpretation(results: List[Dict]) -> AnalysisInsight:
    """Rule-based fallback interpretation"""
    top = results[0] if results else {}
    
    return AnalysisInsight(
        summary=f"{len(results)}개의 유의한 선행 지표가 발견되었습니다.",
        key_findings=[
            f"{r.get('cause', '?')}은 {r.get('best_lag', 0)}개월 선행하여 BTC를 예측합니다."
            for r in results[:3]
        ],
        recommendations=[
            "선행 지표의 변화를 모니터링하세요.",
            "다중 지표의 일치 여부를 확인하세요."
        ],
        confidence="medium"
    )


def _rule_based_gate_interpretation(gate_result: Dict) -> AnalysisInsight:
    """Rule-based Market Gate interpretation"""
    gate = gate_result.get('gate', 'YELLOW')
    score = gate_result.get('score', 50)
    
    interpretations = {
        "GREEN": {
            "summary": "시장이 강세 구간에 있으며 적극적인 진입이 가능합니다.",
            "recommendations": ["VCP 돌파 시그널에 주목하세요.", "포지션 크기를 늘릴 수 있습니다."]
        },
        "YELLOW": {
            "summary": "시장이 중립 구간에 있으며 선별적 접근이 필요합니다.",
            "recommendations": ["고품질 시그널만 선택하세요.", "포지션 크기를 보수적으로 유지하세요."]
        },
        "RED": {
            "summary": "시장이 약세 구간에 있으며 신규 진입을 자제하세요.",
            "recommendations": ["현금 비중을 늘리세요.", "기존 포지션의 손절을 강화하세요."]
        }
    }
    
    interp = interpretations.get(gate, interpretations["YELLOW"])
    
    return AnalysisInsight(
        summary=interp["summary"],
        key_findings=gate_result.get('reasons', [])[:3],
        recommendations=interp["recommendations"],
        confidence="high" if gate in ["GREEN", "RED"] else "medium"
    )
