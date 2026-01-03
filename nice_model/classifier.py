"""
NICE v4 Type A/B/C Classifier
=============================
원본 NICE 모델 분류 로직 (100% 보존)

Type A (75% 신뢰): 즉시 거래 → Kelly 4%
Type B (60% 신뢰): 신중히 → Kelly 2%  
Type C (45% 신뢰): 거래 금지
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class SignalType(Enum):
    """거래 신호 타입"""
    TYPE_A = 'A'  # 75% 신뢰 - 즉시 거래
    TYPE_B = 'B'  # 60% 신뢰 - 신중히
    TYPE_C = 'C'  # 45% 신뢰 - 거래 금지


@dataclass
class NICESignal:
    """NICE 거래 신호"""
    signal_type: SignalType
    confidence: float  # 0-100%
    action: str
    kelly_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    time_stop_minutes: int
    reasons: list
    
    def to_dict(self) -> Dict:
        return {
            'type': self.signal_type.value,
            'confidence': self.confidence,
            'action': self.action,
            'kelly_pct': self.kelly_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'time_stop_minutes': self.time_stop_minutes,
            'reasons': self.reasons
        }


class NICEClassifier:
    """
    NICE v4 Type A/B/C 분류기
    
    점수 기반 분류:
    - 75+ → Type A (즉시 거래)
    - 55-74 → Type B (신중히)
    - <55 → Type C (거래 금지)
    
    사용법:
    >>> classifier = NICEClassifier()
    >>> signal = classifier.classify(score=78.5, details={})
    >>> print(signal.signal_type)  # SignalType.TYPE_A
    """
    
    # 임계값 설정 (원본 NICE 로직)
    THRESHOLD_A = 75.0  # Type A 최소 점수
    THRESHOLD_B = 55.0  # Type B 최소 점수
    
    # Type별 파라미터
    PARAMS = {
        SignalType.TYPE_A: {
            'confidence': 75.0,
            'action': '즉시 거래',
            'kelly_pct': 4.0,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0,
            'time_stop_minutes': 30
        },
        SignalType.TYPE_B: {
            'confidence': 60.0,
            'action': '신중히 거래',
            'kelly_pct': 2.0,
            'stop_loss_pct': 1.5,
            'take_profit_pct': 3.0,
            'time_stop_minutes': 20
        },
        SignalType.TYPE_C: {
            'confidence': 45.0,
            'action': '거래 금지',
            'kelly_pct': 0.0,
            'stop_loss_pct': 0.0,
            'take_profit_pct': 0.0,
            'time_stop_minutes': 0
        }
    }
    
    def classify(self, score: float, layer_details: Dict = None) -> NICESignal:
        """
        점수를 기반으로 Type A/B/C 분류
        
        Args:
            score: NICE 정규화 점수 (0-100)
            layer_details: 레이어별 상세 점수
            
        Returns:
            NICESignal: 거래 신호 객체
        """
        layer_details = layer_details or {}
        
        # Type 결정
        if score >= self.THRESHOLD_A:
            signal_type = SignalType.TYPE_A
        elif score >= self.THRESHOLD_B:
            signal_type = SignalType.TYPE_B
        else:
            signal_type = SignalType.TYPE_C
        
        # 파라미터 가져오기
        params = self.PARAMS[signal_type]
        
        # 판정 이유 생성
        reasons = self._generate_reasons(score, signal_type, layer_details)
        
        return NICESignal(
            signal_type=signal_type,
            confidence=params['confidence'],
            action=params['action'],
            kelly_pct=params['kelly_pct'],
            stop_loss_pct=params['stop_loss_pct'],
            take_profit_pct=params['take_profit_pct'],
            time_stop_minutes=params['time_stop_minutes'],
            reasons=reasons
        )
    
    def _generate_reasons(self, score: float, signal_type: SignalType, layer_details: Dict) -> list:
        """판정 이유 생성"""
        reasons = []
        
        if signal_type == SignalType.TYPE_A:
            reasons.append(f"종합 점수 {score:.1f}점으로 Type A 기준(75점) 충족")
            if layer_details.get('etf', {}).get('btc_etf', 0) >= 12:
                reasons.append("BTC ETF 강한 유입 감지")
            if layer_details.get('technical', {}).get('macd', 0) >= 8:
                reasons.append("MACD 상향 신호 확인")
                
        elif signal_type == SignalType.TYPE_B:
            reasons.append(f"종합 점수 {score:.1f}점으로 Type B 구간")
            reasons.append("일부 지표 불확실 - 신호 명확 시에만 거래")
            
        else:  # TYPE_C
            reasons.append(f"종합 점수 {score:.1f}점으로 거래 부적합")
            reasons.append("기댓값 음수 - 모니터링만 권장")
        
        return reasons
    
    def get_entry_checklist(self, signal: NICESignal) -> list:
        """
        진입 전 체크리스트 생성 (원본 NICE 체크리스트)
        """
        if signal.signal_type == SignalType.TYPE_C:
            return ["❌ 거래 금지 상태입니다"]
        
        checklist = [
            f"[ ] {signal.signal_type.value} 신호 확인",
            "[ ] RSI > 85 확인" if signal.signal_type == SignalType.TYPE_A else "[ ] RSI > 70 확인",
            "[ ] MACD 상향 확인",
            "[ ] 거래량 폭증 확인",
            "[ ] 상위 5개 코인 확인 (BTC, ETH, SOL, AVAX, LINK)",
            f"[ ] Kelly % 계산: 자본 × {signal.kelly_pct}%",
            f"[ ] 손절가 설정: 진입가 - {signal.stop_loss_pct}%",
            f"[ ] 익절가 설정: 진입가 + {signal.take_profit_pct}%",
            "[ ] 현재 포지션 2개 이하 확인",
            f"[ ] Time-Stop 설정: {signal.time_stop_minutes}분"
        ]
        
        return checklist


# 테스트용
if __name__ == '__main__':
    classifier = NICEClassifier()
    
    # Type A 테스트
    signal_a = classifier.classify(score=78.5)
    print(f"Score 78.5 → Type {signal_a.signal_type.value}")
    print(f"  Action: {signal_a.action}")
    print(f"  Kelly: {signal_a.kelly_pct}%")
    
    # Type B 테스트
    signal_b = classifier.classify(score=62.0)
    print(f"\nScore 62.0 → Type {signal_b.signal_type.value}")
    print(f"  Action: {signal_b.action}")
    
    # Type C 테스트
    signal_c = classifier.classify(score=45.0)
    print(f"\nScore 45.0 → Type {signal_c.signal_type.value}")
    print(f"  Action: {signal_c.action}")
