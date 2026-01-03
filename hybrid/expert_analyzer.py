"""
Expert Perspective Analyzer
============================
블랙록 × JP모건 × 전문 트레이더 통합 분석

각 전문가 관점의 실제 분석 로직:
- 블랙록 (BlackRock): ETF/기관 자금 흐름 분석
- JP모건 (JP Morgan): 거시경제 분석
- 전문 트레이더: 기술적 분석
- 시장 분석가: OnChain/센티먼트 분석
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime


class Signal(Enum):
    STRONG_BUY = "강한 매수"
    BUY = "매수"
    NEUTRAL = "중립"
    SELL = "매도"
    STRONG_SELL = "강한 매도"


@dataclass
class ExpertJudgment:
    """전문가 판단 결과"""
    expert_name: str
    signal: Signal
    confidence: float  # 0-100
    reasoning: str
    key_factors: List[str]
    action: str
    
    def to_dict(self) -> Dict:
        return {
            'expert': self.expert_name,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'key_factors': self.key_factors,
            'action': self.action
        }


class BlackRockAnalyzer:
    """
    블랙록 관점: 기관 투자자
    ========================
    분석 대상:
    - ETF 대량 유입/유출
    - 수탁 자산 변화
    - 포트폴리오 리밸런싱
    - 기관 자금 흐름
    
    판단 기준:
    - ETF 유입 > $500M/day → 강한 매수 신호
    - ETF 유입 > $100M/day → 매수 신호
    - ETF 유입 < $50M/day → 중립
    - ETF 유출 → 매도 신호
    """
    
    NAME = "블랙록 (BlackRock)"
    
    def analyze(self, layer5_data: Dict) -> ExpertJudgment:
        """Layer 5 (기관/ETF) 데이터 기반 분석"""
        
        # 데이터 추출
        etf_inflow = layer5_data.get('etf_inflow', 0)  # 백만 달러 단위
        etf_cumulative = layer5_data.get('etf_cumulative', 0)  # 십억 달러 단위
        institutional_score = layer5_data.get('score', 0)
        max_score = layer5_data.get('max', 30)
        
        # 점수 백분율
        pct = (institutional_score / max_score * 100) if max_score > 0 else 0
        
        # 판단 로직
        key_factors = []
        
        # ETF 유입 분석
        if etf_inflow >= 500:
            key_factors.append(f"ETF 대량 유입: +${etf_inflow}M (매우 긍정)")
            etf_signal = 2
        elif etf_inflow >= 100:
            key_factors.append(f"ETF 유입: +${etf_inflow}M (긍정)")
            etf_signal = 1
        elif etf_inflow >= 0:
            key_factors.append(f"ETF 유입: +${etf_inflow}M (중립)")
            etf_signal = 0
        else:
            key_factors.append(f"ETF 유출: ${etf_inflow}M (부정)")
            etf_signal = -1
        
        # 누적 자금 분석
        if etf_cumulative >= 50:
            key_factors.append(f"누적 기관 자금: ${etf_cumulative}B (매우 높음)")
            cum_signal = 2
        elif etf_cumulative >= 30:
            key_factors.append(f"누적 기관 자금: ${etf_cumulative}B (높음)")
            cum_signal = 1
        else:
            key_factors.append(f"누적 기관 자금: ${etf_cumulative}B (보통)")
            cum_signal = 0
        
        # 점수 기반
        if pct >= 90:
            key_factors.append(f"기관 점수: {institutional_score}/{max_score} (최상위)")
            score_signal = 2
        elif pct >= 70:
            key_factors.append(f"기관 점수: {institutional_score}/{max_score} (양호)")
            score_signal = 1
        else:
            key_factors.append(f"기관 점수: {institutional_score}/{max_score} (보통)")
            score_signal = 0
        
        # 종합 신호
        total_signal = etf_signal + cum_signal + score_signal
        
        if total_signal >= 5:
            signal = Signal.STRONG_BUY
            confidence = 85
            reasoning = "기관 대량 진입 확인. 스팟 ETF 강한 유입과 누적 자금 증가로 중장기 상승 전망"
            action = "포트폴리오 비중 확대 권장 (Kelly 4%)"
        elif total_signal >= 3:
            signal = Signal.BUY
            confidence = 70
            reasoning = "기관 자금 유입 지속. ETF 유입 안정적이며 기관 관심 확인"
            action = "점진적 매수 권장 (Kelly 2-3%)"
        elif total_signal >= 1:
            signal = Signal.NEUTRAL
            confidence = 55
            reasoning = "기관 자금 흐름 중립. 추가 유입 확인 필요"
            action = "관망 권장, 추가 신호 대기"
        else:
            signal = Signal.SELL
            confidence = 65
            reasoning = "기관 자금 유출 감지. 리스크 관리 필요"
            action = "포지션 축소 권장"
        
        return ExpertJudgment(
            expert_name=self.NAME,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            action=action
        )


class JPMorganAnalyzer:
    """
    JP모건 관점: 투자은행
    ======================
    분석 대상:
    - 거시경제 지표 (Fed/CPI/GDP)
    - 통화 정책 변화
    - 글로벌 시장 연동성
    - 위험자산 선호도
    
    판단 기준:
    - Fed 금리 인하 기대 → 매수
    - CPI 하락(인플레 안정) → 매수
    - 달러 약세 → 암호화폐 매수
    - VIX 낮음 → 리스크온
    """
    
    NAME = "JP모건 (JP Morgan)"
    
    def analyze(self, layer4_data: Dict) -> ExpertJudgment:
        """Layer 4 (매크로) 데이터 기반 분석"""
        
        # 데이터 추출
        fed_rate = layer4_data.get('fed_rate', 4.5)
        cpi = layer4_data.get('cpi', 3.0)
        unemployment = layer4_data.get('unemployment', 4.0)
        dxy = layer4_data.get('dxy', 100)  # 달러 인덱스
        vix = layer4_data.get('vix', 20)
        macro_score = layer4_data.get('score', 0)
        max_score = layer4_data.get('max', 40)
        
        pct = (macro_score / max_score * 100) if max_score > 0 else 0
        
        key_factors = []
        signals = []
        
        # Fed 금리 분석
        if fed_rate <= 4.0:
            key_factors.append(f"Fed 금리 {fed_rate}%: 완화적 통화정책 (긍정)")
            signals.append(2)
        elif fed_rate <= 4.5:
            key_factors.append(f"Fed 금리 {fed_rate}%: 중립적 (관망)")
            signals.append(1)
        else:
            key_factors.append(f"Fed 금리 {fed_rate}%: 긴축적 (부정)")
            signals.append(-1)
        
        # CPI (인플레이션) 분석
        if cpi <= 2.5:
            key_factors.append(f"CPI {cpi}%: 물가 안정 (매우 긍정)")
            signals.append(2)
        elif cpi <= 3.0:
            key_factors.append(f"CPI {cpi}%: 물가 개선 중 (긍정)")
            signals.append(1)
        else:
            key_factors.append(f"CPI {cpi}%: 인플레 우려 (부정)")
            signals.append(-1)
        
        # 달러 인덱스 분석
        if dxy <= 100:
            key_factors.append(f"DXY {dxy}: 달러 약세 (암호화폐 호재)")
            signals.append(2)
        elif dxy <= 103:
            key_factors.append(f"DXY {dxy}: 달러 보합 (중립)")
            signals.append(0)
        else:
            key_factors.append(f"DXY {dxy}: 달러 강세 (암호화폐 악재)")
            signals.append(-1)
        
        # VIX (공포지수) 분석
        if vix <= 15:
            key_factors.append(f"VIX {vix}: 변동성 매우 낮음 (리스크온)")
            signals.append(2)
        elif vix <= 20:
            key_factors.append(f"VIX {vix}: 변동성 낮음 (양호)")
            signals.append(1)
        else:
            key_factors.append(f"VIX {vix}: 변동성 높음 (주의)")
            signals.append(-1)
        
        total_signal = sum(signals)
        
        if total_signal >= 6:
            signal = Signal.STRONG_BUY
            confidence = 85
            reasoning = "거시경제 환경 최적. 통화완화 기대, 달러 약세, 저변동성으로 위험자산 선호 환경"
            action = "적극 매수 권장 (Kelly 4%)"
        elif total_signal >= 3:
            signal = Signal.BUY
            confidence = 70
            reasoning = "거시경제 호조. 인플레 안정 추세와 완화적 통화정책 기대"
            action = "매수 권장 (Kelly 2-3%)"
        elif total_signal >= 0:
            signal = Signal.NEUTRAL
            confidence = 55
            reasoning = "거시경제 혼조. 일부 지표 개선되나 불확실성 존재"
            action = "관망 권장, 추가 지표 확인"
        else:
            signal = Signal.SELL
            confidence = 65
            reasoning = "거시경제 악화. 긴축 정책과 달러 강세로 위험자산 압박"
            action = "포지션 축소 권장"
        
        return ExpertJudgment(
            expert_name=self.NAME,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            action=action
        )


class ProfessionalTraderAnalyzer:
    """
    전문 트레이더 관점: 단기 거래
    =============================
    분석 대상:
    - 기술 지표 (RSI/MACD)
    - 거래소 유동성 변화
    - 호가창 깊이 분석
    - 거래 신호 (5분봉)
    
    판단 기준:
    - RSI > 85 + MACD 상향 → Type A 신호
    - RSI > 70 + 거래량 증가 → 매수
    - RSI < 30 → 과매도 반등 기대
    """
    
    NAME = "전문 트레이더"
    
    def analyze(self, layer1_data: Dict) -> ExpertJudgment:
        """Layer 1 (기술분석) 데이터 기반 분석"""
        
        rsi = layer1_data.get('rsi', 50)
        macd = layer1_data.get('macd', 'neutral')  # 'up', 'down', 'neutral'
        volume_change = layer1_data.get('volume_change', 0)  # 퍼센트
        technical_score = layer1_data.get('score', 0)
        max_score = layer1_data.get('max', 100)
        
        pct = (technical_score / max_score * 100) if max_score > 0 else 0
        
        key_factors = []
        signals = []
        
        # RSI 분석
        if rsi >= 85:
            key_factors.append(f"RSI {rsi}: 과매수 (Type A 신호)")
            signals.append(2 if macd == 'up' else 1)
        elif rsi >= 70:
            key_factors.append(f"RSI {rsi}: 강세 (매수 우위)")
            signals.append(1)
        elif rsi >= 50:
            key_factors.append(f"RSI {rsi}: 중립 (추세 확인 필요)")
            signals.append(0)
        elif rsi >= 30:
            key_factors.append(f"RSI {rsi}: 약세 (관망)")
            signals.append(-1)
        else:
            key_factors.append(f"RSI {rsi}: 과매도 (반등 기대)")
            signals.append(1)
        
        # MACD 분석
        if macd == 'up':
            key_factors.append("MACD: 상향 골든크로스 (강한 매수 신호)")
            signals.append(2)
        elif macd == 'neutral':
            key_factors.append("MACD: 중립 (추세 전환 대기)")
            signals.append(0)
        else:
            key_factors.append("MACD: 하향 데드크로스 (매도 신호)")
            signals.append(-2)
        
        # 거래량 분석
        if volume_change >= 100:
            key_factors.append(f"거래량: +{volume_change}% (폭증, 신호 신뢰도 높음)")
            signals.append(2)
        elif volume_change >= 50:
            key_factors.append(f"거래량: +{volume_change}% (증가)")
            signals.append(1)
        elif volume_change >= 0:
            key_factors.append(f"거래량: +{volume_change}% (보합)")
            signals.append(0)
        else:
            key_factors.append(f"거래량: {volume_change}% (감소, 신호 약함)")
            signals.append(-1)
        
        # 기술 점수
        if pct >= 80:
            key_factors.append(f"기술 점수: {technical_score}/{max_score} (강한 상승 추세)")
            signals.append(2)
        elif pct >= 65:
            key_factors.append(f"기술 점수: {technical_score}/{max_score} (상승 추세)")
            signals.append(1)
        else:
            key_factors.append(f"기술 점수: {technical_score}/{max_score} (약함)")
            signals.append(0)
        
        total_signal = sum(signals)
        
        if total_signal >= 6:
            signal = Signal.STRONG_BUY
            confidence = 80
            reasoning = "Type A 기술 신호. RSI 과매수 + MACD 상향 + 거래량 폭증으로 즉시 진입 권장"
            action = "즉시 매수 (Kelly 4%, SL -2%, TP +4%, Time-Stop 30분)"
        elif total_signal >= 3:
            signal = Signal.BUY
            confidence = 65
            reasoning = "기술 신호 양호. 추세 상향 확인되나 확인 매수 권장"
            action = "매수 (Kelly 2%, 추세 확인 후)"
        elif total_signal >= 0:
            signal = Signal.NEUTRAL
            confidence = 50
            reasoning = "기술 신호 혼조. 명확한 방향성 부재"
            action = "관망 권장, 신호 대기"
        else:
            signal = Signal.SELL
            confidence = 65
            reasoning = "하락 신호. MACD 데드크로스 또는 약세 패턴"
            action = "매도 또는 포지션 정리"
        
        return ExpertJudgment(
            expert_name=self.NAME,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            action=action
        )


class MarketAnalystAnalyzer:
    """
    시장 분석가 관점: OnChain/센티먼트
    ==================================
    분석 대상:
    - OnChain 데이터 (고래/MVRV)
    - 센티먼트 분석 (Fear & Greed)
    - 시장 심리
    - 유동성 구조
    
    판단 기준:
    - 고래 유입 + MVRV 정상 → 매수
    - Fear & Greed > 70 → 과열 주의
    - Fear & Greed < 30 → 공포 매수 기회
    """
    
    NAME = "시장 분석가"
    
    def analyze(self, layer2_data: Dict, layer3_data: Dict) -> ExpertJudgment:
        """Layer 2/3 (OnChain/심리) 데이터 기반 분석"""
        
        # OnChain 데이터
        whale_inflow = layer2_data.get('whale_inflow', 0)  # 고래 유입 개수
        mvrv = layer2_data.get('mvrv', 1.0)
        onchain_score = layer2_data.get('score', 0)
        onchain_max = layer2_data.get('max', 30)
        
        # 심리 데이터
        fear_greed = layer3_data.get('fear_greed', 50)
        sentiment_score = layer3_data.get('score', 0)
        sentiment_max = layer3_data.get('max', 100)
        
        key_factors = []
        signals = []
        
        # 고래 유입 분석
        if whale_inflow >= 10:
            key_factors.append(f"고래 유입: +{whale_inflow}개 (대량 진입)")
            signals.append(2)
        elif whale_inflow >= 5:
            key_factors.append(f"고래 유입: +{whale_inflow}개 (진입 중)")
            signals.append(1)
        elif whale_inflow >= 0:
            key_factors.append(f"고래 유입: +{whale_inflow}개 (보합)")
            signals.append(0)
        else:
            key_factors.append(f"고래 유출: {whale_inflow}개 (이탈)")
            signals.append(-2)
        
        # MVRV 분석
        if 1.5 <= mvrv <= 2.5:
            key_factors.append(f"MVRV {mvrv}: 적정 범위 (건강한 시장)")
            signals.append(1)
        elif mvrv < 1.0:
            key_factors.append(f"MVRV {mvrv}: 저평가 (매수 기회)")
            signals.append(2)
        elif mvrv > 3.0:
            key_factors.append(f"MVRV {mvrv}: 과열 (차익실현 구간)")
            signals.append(-1)
        else:
            key_factors.append(f"MVRV {mvrv}: 보통")
            signals.append(0)
        
        # Fear & Greed 분석
        if fear_greed >= 75:
            key_factors.append(f"Fear & Greed {fear_greed}: 극도의 탐욕 (과열 경고)")
            signals.append(-1)
        elif fear_greed >= 55:
            key_factors.append(f"Fear & Greed {fear_greed}: 탐욕 (상승 추세)")
            signals.append(1)
        elif fear_greed >= 45:
            key_factors.append(f"Fear & Greed {fear_greed}: 중립")
            signals.append(0)
        elif fear_greed >= 25:
            key_factors.append(f"Fear & Greed {fear_greed}: 공포 (매수 기회)")
            signals.append(2)
        else:
            key_factors.append(f"Fear & Greed {fear_greed}: 극도의 공포 (강한 매수 기회)")
            signals.append(3)
        
        # OnChain 점수
        onchain_pct = (onchain_score / onchain_max * 100) if onchain_max > 0 else 0
        if onchain_pct >= 80:
            key_factors.append(f"OnChain 점수: {onchain_score}/{onchain_max} (매우 긍정)")
            signals.append(2)
        elif onchain_pct >= 60:
            key_factors.append(f"OnChain 점수: {onchain_score}/{onchain_max} (긍정)")
            signals.append(1)
        else:
            key_factors.append(f"OnChain 점수: {onchain_score}/{onchain_max}")
            signals.append(0)
        
        total_signal = sum(signals)
        
        if total_signal >= 5:
            signal = Signal.STRONG_BUY
            confidence = 80
            reasoning = "시장 심층 분석 긍정. 기관/고래 진입 확인, 센티먼트 개선 중"
            action = "적극 매수 (중기 보유 권장)"
        elif total_signal >= 2:
            signal = Signal.BUY
            confidence = 65
            reasoning = "OnChain 지표 양호. 고래 유입 있으나 확인 필요"
            action = "분할 매수 권장"
        elif total_signal >= 0:
            signal = Signal.NEUTRAL
            confidence = 50
            reasoning = "시장 심리 혼조. 방향성 확인 필요"
            action = "관망"
        else:
            signal = Signal.SELL
            confidence = 60
            reasoning = "부정적 신호. 고래 이탈 또는 과열 구간"
            action = "포지션 축소"
        
        return ExpertJudgment(
            expert_name=self.NAME,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            action=action
        )


class ExpertPerspectiveOrchestrator:
    """
    전문가 관점 통합 분석기
    ======================
    4가지 전문가 관점을 종합하여 최종 판단 생성
    """
    
    def __init__(self):
        self.blackrock = BlackRockAnalyzer()
        self.jpmorgan = JPMorganAnalyzer()
        self.trader = ProfessionalTraderAnalyzer()
        self.analyst = MarketAnalystAnalyzer()
    
    def analyze_all(self, layer_data: Dict) -> Dict:
        """
        모든 전문가 관점 분석
        
        Parameters:
        - layer_data: {
            'layer1': { 'score': int, 'max': int, 'rsi': float, 'macd': str, 'volume_change': float },
            'layer2': { 'score': int, 'max': int, 'whale_inflow': int, 'mvrv': float },
            'layer3': { 'score': int, 'max': int, 'fear_greed': int },
            'layer4': { 'score': int, 'max': int, 'fed_rate': float, 'cpi': float, 'dxy': float, 'vix': float },
            'layer5': { 'score': int, 'max': int, 'etf_inflow': float, 'etf_cumulative': float }
        }
        """
        
        layer1 = layer_data.get('layer1', {})
        layer2 = layer_data.get('layer2', {})
        layer3 = layer_data.get('layer3', {})
        layer4 = layer_data.get('layer4', {})
        layer5 = layer_data.get('layer5', {})
        
        # 각 전문가 분석
        blackrock_result = self.blackrock.analyze(layer5)
        jpmorgan_result = self.jpmorgan.analyze(layer4)
        trader_result = self.trader.analyze(layer1)
        analyst_result = self.analyst.analyze(layer2, layer3)
        
        # 종합 신호 계산
        signal_weights = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.NEUTRAL: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2
        }
        
        weighted_sum = (
            signal_weights[blackrock_result.signal] * 0.25 +
            signal_weights[jpmorgan_result.signal] * 0.20 +
            signal_weights[trader_result.signal] * 0.30 +
            signal_weights[analyst_result.signal] * 0.25
        )
        
        # 평균 신뢰도
        avg_confidence = (
            blackrock_result.confidence * 0.25 +
            jpmorgan_result.confidence * 0.20 +
            trader_result.confidence * 0.30 +
            analyst_result.confidence * 0.25
        )
        
        # 최종 신호
        if weighted_sum >= 1.5:
            consensus = Signal.STRONG_BUY
            consensus_text = "전원 합의: 강한 매수"
        elif weighted_sum >= 0.5:
            consensus = Signal.BUY
            consensus_text = "다수 합의: 매수"
        elif weighted_sum >= -0.5:
            consensus = Signal.NEUTRAL
            consensus_text = "의견 분분: 관망"
        elif weighted_sum >= -1.5:
            consensus = Signal.SELL
            consensus_text = "다수 합의: 매도"
        else:
            consensus = Signal.STRONG_SELL
            consensus_text = "전원 합의: 강한 매도"
        
        return {
            'experts': [
                blackrock_result.to_dict(),
                jpmorgan_result.to_dict(),
                trader_result.to_dict(),
                analyst_result.to_dict()
            ],
            'consensus': {
                'signal': consensus.value,
                'text': consensus_text,
                'confidence': round(avg_confidence, 1),
                'weighted_score': round(weighted_sum, 2)
            },
            'timestamp': datetime.now().isoformat()
        }


# 테스트
if __name__ == '__main__':
    orchestrator = ExpertPerspectiveOrchestrator()
    
    # 샘플 데이터
    sample_data = {
        'layer1': {'score': 85, 'max': 100, 'rsi': 87, 'macd': 'up', 'volume_change': 145},
        'layer2': {'score': 26, 'max': 30, 'whale_inflow': 15, 'mvrv': 2.1},
        'layer3': {'score': 45, 'max': 100, 'fear_greed': 45},
        'layer4': {'score': 36, 'max': 40, 'fed_rate': 4.25, 'cpi': 2.6, 'dxy': 102.5, 'vix': 18.5},
        'layer5': {'score': 29, 'max': 30, 'etf_inflow': 1800, 'etf_cumulative': 52}
    }
    
    result = orchestrator.analyze_all(sample_data)
    
    print("=== 전문가 관점 통합 분석 ===\n")
    
    for expert in result['experts']:
        print(f"【 {expert['expert']} 】")
        print(f"  신호: {expert['signal']} (신뢰도 {expert['confidence']}%)")
        print(f"  판단: {expert['reasoning']}")
        print(f"  요인: {', '.join(expert['key_factors'])}")
        print(f"  행동: {expert['action']}")
        print()
    
    print(f"=== 전문가 합의 ===")
    print(f"  {result['consensus']['text']}")
    print(f"  신뢰도: {result['consensus']['confidence']}%")
