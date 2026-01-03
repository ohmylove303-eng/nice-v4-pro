"""
Hybrid Orchestrator
===================
NICE Hybrid System 전체 파이프라인 관리

파이프라인:
1. DataAggregator → 데이터 수집
2. NICEScorer → 5레이어 점수 계산
3. NICEClassifier → Type A/B/C 분류
4. KellyCalculator → 포지션 크기 계산
5. 최종 결과 반환
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class HybridResult:
    """하이브리드 분석 결과"""
    score: float                    # 종합 점수 (0-100)
    signal_type: str               # 'A', 'B', 'C'
    confidence: float              # 신뢰도 %
    action: str                    # 액션 (즉시 거래, 신중히, 금지)
    kelly_pct: float               # Kelly %
    position_size: float           # 포지션 크기 ($)
    layers: Dict                   # 레이어별 점수
    checklist: list                # 진입 체크리스트
    reasons: list                  # 판정 이유
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'score': round(self.score, 1),
            'signal': {
                'type': self.signal_type,
                'confidence': self.confidence,
                'action': self.action
            },
            'position': {
                'kelly_pct': self.kelly_pct,
                'size_usd': round(self.position_size, 2)
            },
            'layers': self.layers,
            'checklist': self.checklist,
            'reasons': self.reasons,
            'timestamp': self.timestamp.isoformat()
        }


class HybridOrchestrator:
    """
    NICE Hybrid System 오케스트레이터
    
    사용법:
    >>> orch = HybridOrchestrator(capital=10000)
    >>> result = orch.run()
    >>> print(result.signal_type)  # 'A', 'B', or 'C'
    """
    
    def __init__(self, capital: float = 10000.0):
        self.capital = capital
        
        # 모듈 임포트
        from .data_aggregator import DataAggregator
        from nice_model.scorer import NICEScorer
        from nice_model.classifier import NICEClassifier
        from nice_model.kelly import KellyCalculator
        
        self.aggregator = DataAggregator()
        self.scorer = NICEScorer()
        self.classifier = NICEClassifier()
        self.kelly = KellyCalculator(capital=capital)
    
    def run(self, custom_data: Optional[Dict] = None) -> HybridResult:
        """
        전체 파이프라인 실행
        
        Args:
            custom_data: 커스텀 데이터 (없으면 자동 수집)
            
        Returns:
            HybridResult: 하이브리드 분석 결과
        """
        # 1. 데이터 수집
        if custom_data:
            data = custom_data
        else:
            data = self.aggregator.collect_all()
        
        # 2. NICE 점수 계산
        score_result = self.scorer.calculate(data)
        
        # 3. Type 분류
        signal = self.classifier.classify(
            score=score_result.total_normalized,
            layer_details=score_result.to_dict()['layers']
        )
        
        # 4. Kelly 계산
        kelly_result = self.kelly.calculate(signal.signal_type.value)
        
        # 5. 체크리스트 생성
        checklist = self.classifier.get_entry_checklist(signal)
        
        # 6. 결과 조합
        return HybridResult(
            score=score_result.total_normalized,
            signal_type=signal.signal_type.value,
            confidence=signal.confidence,
            action=signal.action,
            kelly_pct=signal.kelly_pct,
            position_size=kelly_result.position_size,
            layers=score_result.to_dict()['layers'],
            checklist=checklist,
            reasons=signal.reasons,
            timestamp=datetime.now()
        )
    
    def get_summary(self) -> str:
        """
        사람이 읽기 쉬운 요약 생성
        """
        result = self.run()
        
        lines = [
            "=" * 50,
            "🎯 NICE Hybrid System - 분석 결과",
            "=" * 50,
            "",
            f"📊 종합 점수: {result.score:.1f}/100",
            f"🚦 신호 타입: Type {result.signal_type} ({result.confidence}% 신뢰)",
            f"💡 액션: {result.action}",
            "",
            f"💰 Kelly %: {result.kelly_pct}%",
            f"📈 포지션 크기: ${result.position_size:,.2f}",
            "",
            "📋 레이어별 점수:",
        ]
        
        for layer_name, layer_data in result.layers.items():
            lines.append(f"  - {layer_name}: {layer_data['score']}/{layer_data['max']}")
        
        lines.extend([
            "",
            "📝 판정 이유:",
        ])
        for reason in result.reasons:
            lines.append(f"  • {reason}")
        
        lines.extend([
            "",
            "=" * 50,
            f"⏰ {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
        ])
        
        return "\n".join(lines)


def run_hybrid(capital: float = 10000.0) -> Dict:
    """
    간편 실행 함수
    
    사용법:
    >>> from hybrid.orchestrator import run_hybrid
    >>> result = run_hybrid(capital=10000)
    """
    orch = HybridOrchestrator(capital=capital)
    return orch.run().to_dict()


# 테스트용
if __name__ == '__main__':
    print("=== NICE Hybrid System Test ===\n")
    
    orch = HybridOrchestrator(capital=10000)
    print(orch.get_summary())
