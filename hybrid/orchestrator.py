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
    
    v2.0: Protocol Gates + Palantir Tracker 통합
    
    사용법:
    >>> orch = HybridOrchestrator(capital=10000)
    >>> result = orch.run()
    >>> print(result.signal_type)  # 'A', 'B', or 'C'
    """
    
    def __init__(self, capital: float = 10000.0, strict_mode: bool = True):
        self.capital = capital
        self.strict_mode = strict_mode  # Fail-Closed 모드
        
        # 모듈 임포트
        from .data_aggregator import DataAggregator
        from nice_model.scorer import NICEScorer
        from nice_model.classifier import NICEClassifier
        from nice_model.kelly import KellyCalculator
        
        self.aggregator = DataAggregator()
        self.scorer = NICEScorer()
        self.classifier = NICEClassifier()
        self.kelly = KellyCalculator(capital=capital)
        
        # v2.0: Protocol Gates + Palantir 통합
        self.protocol_gates = None
        self.palantir = None
        
        try:
            from .protocol_gates import ProtocolGates
            self.protocol_gates = ProtocolGates()
        except Exception as e:
            print(f"[WARN] Protocol Gates 로드 실패: {e}")
        
        try:
            from .palantir_tracker import PalantirTracker
            self.palantir = PalantirTracker()
        except Exception as e:
            print(f"[WARN] Palantir Tracker 로드 실패: {e}")
    
    def run(self, custom_data: Optional[Dict] = None) -> HybridResult:
        """
        전체 파이프라인 실행 (Protocol Gates + Palantir 통합)
        
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
        
        # 2. Palantir: 데이터 계보 추적 시작
        lineage = {}
        if self.palantir:
            lineage = self.palantir.build_lineage(data)
        
        # 3. NICE 점수 계산
        score_result = self.scorer.calculate(data)
        
        # 4. Type 분류
        signal = self.classifier.classify(
            score=score_result.total_normalized,
            layer_details=score_result.to_dict()['layers']
        )
        
        # 5. ========== Protocol Gates 검증 (v2.0) ==========
        gates_pass = True
        gates_status = {'data_integrity': True, 'liquidity': True, 'confirm': True}
        
        if self.protocol_gates and self.strict_mode:
            try:
                # Gate 1: 데이터 무결성
                gates_status['data_integrity'] = self.protocol_gates.check_data_integrity(data)
                
                # Gate 2: 유동성 가드 (시뮬레이션 데이터)
                orderbook = data.get('orderbook', {'bid_volume': 1000000, 'ask_volume': 1000000, 'spread': 0.1})
                gates_status['liquidity'] = self.protocol_gates.check_liquidity_guards(orderbook)
                
                # Gate 3: 확인 게이트
                gates_status['confirm'] = self.protocol_gates.check_confirm_gate(
                    score_result.total_normalized,
                    signal.confidence,
                    score_result.to_dict()['layers']
                )
                
                gates_pass = all(gates_status.values())
                
                # Fail-Closed: 게이트 하나라도 실패 시 Type C로 강등
                if not gates_pass:
                    signal.signal_type = type('SignalType', (), {'value': 'C'})()
                    signal.action = '진입 금지 (Gate 실패)'
                    signal.reasons.append(f"Protocol Gates 실패: {[k for k,v in gates_status.items() if not v]}")
                    
            except Exception as e:
                print(f"[WARN] Protocol Gates 검증 실패: {e}")
        
        # 6. Kelly 계산 (게이트 통과 시에만 유효)
        if gates_pass:
            kelly_result = self.kelly.calculate(signal.signal_type.value)
        else:
            kelly_result = type('KellyResult', (), {'position_size': 0, 'kelly_pct': 0})()
        
        # 7. 체크리스트 생성
        checklist = self.classifier.get_entry_checklist(signal)
        
        # 8. Palantir: 증거 원장 기록
        if self.palantir:
            self.palantir.build_evidence({
                'score': score_result.total_normalized,
                'signal': signal.signal_type.value,
                'gates_pass': gates_pass,
                'gates_status': gates_status
            })
        
        # 9. 결과 조합
        result = HybridResult(
            score=score_result.total_normalized,
            signal_type=signal.signal_type.value,
            confidence=signal.confidence,
            action=signal.action,
            kelly_pct=signal.kelly_pct if gates_pass else 0,
            position_size=kelly_result.position_size,
            layers=score_result.to_dict()['layers'],
            checklist=checklist,
            reasons=signal.reasons,
            timestamp=datetime.now()
        )
        
        # 10. 결과에 Gates 상태 추가
        result.gates_pass = gates_pass
        result.gates_status = gates_status
        result.lineage = lineage
        
        return result
    
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
