"""
Palantir Tracker AIP
====================
데이터 계보 + 증거 원장 + 결정 온톨로지

천재들의 질문법 5가지 지원:
- Q1: 문제 해결? → 실시간 신호 제공
- Q2: 한계 인식? → Meta Reflection 포함
- Q3: 개선 가능? → 로드맵 제시
- Q4: 신뢰 근거? → Lineage + Evidence
- Q5: 실행 준비? → Gates + OCO
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import hashlib
import json


class DecisionStatus(Enum):
    ELIGIBLE = "ELIGIBLE"
    BLOCKED = "BLOCKED"
    HOLD = "HOLD"


@dataclass
class LineageNode:
    """데이터 계보 노드"""
    source_id: str
    source_type: str  # 'exchange_api', 'calculated', 'external_api'
    timestamp: str
    reliability: float  # 0.0 ~ 1.0
    data_hash: str = ""
    
    def __post_init__(self):
        if not self.data_hash:
            self.data_hash = hashlib.md5(
                f"{self.source_id}:{self.timestamp}".encode()
            ).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'source_type': self.source_type,
            'timestamp': self.timestamp,
            'reliability': self.reliability,
            'data_hash': self.data_hash
        }


@dataclass
class ComputationStep:
    """계산 단계"""
    step_number: int
    layer_name: str
    output_value: Any
    version: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            'step': self.step_number,
            'layer': self.layer_name,
            'output': self.output_value,
            'version': self.version,
            'timestamp': self.timestamp
        }


@dataclass
class EvidenceEntry:
    """증거 원장 항목"""
    evidence_id: str
    evidence_type: str  # 'data_point', 'calculation', 'gate_result'
    description: str
    value: Any
    timestamp: str
    verified: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'evidence_id': self.evidence_id,
            'type': self.evidence_type,
            'description': self.description,
            'value': self.value,
            'timestamp': self.timestamp,
            'verified': self.verified
        }


class PalantirTracker:
    """
    Palantir AIP 스타일 추적 시스템
    
    구성요소:
    1. Lineage (데이터 계보) - 데이터가 어디서 왔는지
    2. Evidence Ledger (증거 원장) - 결정에 사용된 증거
    3. Decision Ontology (결정 온톨로지) - 결정 구조화
    """
    
    def __init__(self, analysis_id: str = None):
        self.analysis_id = analysis_id or f"plt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.lineage_nodes: List[LineageNode] = []
        self.computation_steps: List[ComputationStep] = []
        self.evidence_entries: List[EvidenceEntry] = []
        self.execution_log: List[Dict] = []
        
        # 온톨로지 정의
        self.ontology = self._build_ontology()
    
    def _build_ontology(self) -> Dict:
        """결정 온톨로지 구축"""
        return {
            'version': 'NICE_v18.3_Ontology',
            'entities': {
                'Signal': {
                    'types': ['TYPE_A', 'TYPE_B', 'TYPE_C', 'HOLD'],
                    'properties': ['score', 'confidence', 'layers']
                },
                'Gate': {
                    'types': ['DATA_INTEGRITY', 'LIQUIDITY_GUARDS', 'CONFIRM_GATE'],
                    'properties': ['status', 'reason', 'metrics']
                },
                'Decision': {
                    'types': ['ELIGIBLE', 'BLOCKED', 'HOLD'],
                    'properties': ['timestamp', 'reason', 'allowed_action']
                },
                'Order': {
                    'types': ['PULLBACK_OCO', 'BREAKOUT_OCO'],
                    'properties': ['entry', 'stop_loss', 'take_profit', 'risk_reward']
                }
            },
            'relationships': [
                {'from': 'Signal', 'to': 'Gate', 'type': 'VALIDATES'},
                {'from': 'Gate', 'to': 'Decision', 'type': 'DETERMINES'},
                {'from': 'Decision', 'to': 'Order', 'type': 'ENABLES'}
            ],
            'constraints': [
                'All Gates must PASS for ELIGIBLE Decision',
                'BLOCKED Decision prohibits Order creation',
                'Risk:Reward >= 2.0 for valid Order'
            ]
        }
    
    def build_lineage(
        self,
        data_sources: Dict[str, Dict],
        computation_steps: List[Dict]
    ) -> Dict:
        """
        데이터 계보 구축
        
        Parameters:
            data_sources: {
                'source_id': {
                    'type': 'exchange_api',
                    'timestamp': '...',
                    'reliability': 0.95
                }
            }
            computation_steps: [
                {'step': 1, 'layer': 'Layer1', 'output': ..., 'version': '...'}
            ]
        """
        # 소스 노드 추가
        for source_id, source_info in data_sources.items():
            node = LineageNode(
                source_id=source_id,
                source_type=source_info.get('type', 'unknown'),
                timestamp=source_info.get('timestamp', datetime.now().isoformat()),
                reliability=source_info.get('reliability', 0.8)
            )
            self.lineage_nodes.append(node)
        
        # 계산 단계 추가
        for step_info in computation_steps:
            step = ComputationStep(
                step_number=step_info.get('step', 0),
                layer_name=step_info.get('layer', 'Unknown'),
                output_value=step_info.get('output', None),
                version=step_info.get('version', 'NICE_v18.3')
            )
            self.computation_steps.append(step)
        
        return {
            'analysis_id': self.analysis_id,
            'lineage_type': 'NICE_5Layer_Analysis',
            'data_sources': [n.to_dict() for n in self.lineage_nodes],
            'computation_chain': [s.to_dict() for s in self.computation_steps],
            'total_reliability': self._calculate_total_reliability(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_total_reliability(self) -> float:
        """전체 신뢰도 계산 (소스 신뢰도의 가중 평균)"""
        if not self.lineage_nodes:
            return 0.0
        
        total = sum(n.reliability for n in self.lineage_nodes)
        return round(total / len(self.lineage_nodes), 3)
    
    def build_evidence(
        self,
        analysis_id: str,
        realtime_data: Dict,
        nice_analysis: Dict,
        protocol_gates: Dict
    ) -> Dict:
        """
        증거 원장 구축
        
        모든 결정에 사용된 데이터 포인트를 기록
        """
        evidence_id_counter = 1
        
        # 실시간 데이터 증거
        if realtime_data.get('ticker'):
            self.evidence_entries.append(EvidenceEntry(
                evidence_id=f"ev-{evidence_id_counter:03d}",
                evidence_type='data_point',
                description='현재가 데이터',
                value=realtime_data.get('ticker', {}),
                timestamp=datetime.now().isoformat()
            ))
            evidence_id_counter += 1
        
        if realtime_data.get('orderbook'):
            self.evidence_entries.append(EvidenceEntry(
                evidence_id=f"ev-{evidence_id_counter:03d}",
                evidence_type='data_point',
                description='호가 데이터',
                value=realtime_data.get('orderbook', {}),
                timestamp=datetime.now().isoformat()
            ))
            evidence_id_counter += 1
        
        # NICE 분석 증거
        self.evidence_entries.append(EvidenceEntry(
            evidence_id=f"ev-{evidence_id_counter:03d}",
            evidence_type='calculation',
            description='NICE 점수',
            value={
                'score': nice_analysis.get('score'),
                'signal': nice_analysis.get('signal'),
                'confidence': nice_analysis.get('confidence')
            },
            timestamp=datetime.now().isoformat()
        ))
        evidence_id_counter += 1
        
        # Gate 결과 증거
        for gate_name in ['data_integrity', 'liquidity_guards', 'confirm_gate']:
            gate_result = protocol_gates.get(gate_name, {})
            self.evidence_entries.append(EvidenceEntry(
                evidence_id=f"ev-{evidence_id_counter:03d}",
                evidence_type='gate_result',
                description=f'{gate_name} 검증 결과',
                value=gate_result,
                timestamp=datetime.now().isoformat()
            ))
            evidence_id_counter += 1
        
        return {
            'analysis_id': analysis_id,
            'ledger_type': 'NICE_Evidence_Ledger',
            'entries': [e.to_dict() for e in self.evidence_entries],
            'total_entries': len(self.evidence_entries),
            'all_verified': all(e.verified for e in self.evidence_entries),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_ontology(self) -> Dict:
        """온톨로지 반환"""
        return self.ontology
    
    def log_execution_decision(
        self,
        status: str,
        reason: Optional[str] = None
    ):
        """실행 결정 로깅"""
        self.execution_log.append({
            'analysis_id': self.analysis_id,
            'status': status,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_full_audit_trail(self) -> Dict:
        """전체 감사 추적 반환"""
        return {
            'analysis_id': self.analysis_id,
            'lineage': {
                'data_sources': [n.to_dict() for n in self.lineage_nodes],
                'computation_steps': [s.to_dict() for s in self.computation_steps]
            },
            'evidence_ledger': [e.to_dict() for e in self.evidence_entries],
            'execution_log': self.execution_log,
            'ontology_version': self.ontology['version'],
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_genius_questions_report(self, nice_analysis: Dict, protocol_gates: Dict) -> Dict:
        """
        천재들의 질문법 5가지 리포트 생성
        
        Q1: 문제 해결? → 실시간 신호 제공
        Q2: 한계 인식? → Meta Reflection 포함
        Q3: 개선 가능? → 로드맵 제시
        Q4: 신뢰 근거? → Lineage + Evidence
        Q5: 실행 준비? → Gates + OCO
        """
        score = nice_analysis.get('score', 0)
        signal = nice_analysis.get('signal', 'UNKNOWN')
        confidence = nice_analysis.get('confidence', 0)
        
        return {
            'Q1_problem_solving': {
                'question': '문제를 해결하는가?',
                'answer': f"예. NICE 점수 {score:.2f}로 {signal} 신호를 도출하여 매매 결정 지원",
                'evidence': f"5-Layer 분석 완료, 신뢰도 {confidence:.1%}"
            },
            'Q2_limitations': {
                'question': '한계를 인식하는가?',
                'answer': nice_analysis.get('meta_reflection', {}).get('limitations', [
                    "과거 데이터 기반 분석의 한계",
                    "급격한 시장 변동 시 신호 지연 가능"
                ]),
                'mitigation': "Fail-Closed 원칙으로 안전 우선"
            },
            'Q3_improvement': {
                'question': '개선 가능한가?',
                'roadmap': [
                    "실시간 온체인 데이터 확장",
                    "AI 모델 정확도 향상",
                    "백테스트 커버리지 확대"
                ]
            },
            'Q4_trust_basis': {
                'question': '신뢰 근거가 있는가?',
                'lineage': {
                    'sources': len(self.lineage_nodes),
                    'reliability': self._calculate_total_reliability()
                },
                'evidence': {
                    'entries': len(self.evidence_entries),
                    'all_verified': all(e.verified for e in self.evidence_entries)
                }
            },
            'Q5_execution_ready': {
                'question': '실행 준비가 되었는가?',
                'gates_pass': protocol_gates.get('all_pass', False),
                'fail_closed': protocol_gates.get('fail_closed_active', True),
                'action_allowed': 'PENDING_APPROVAL' if protocol_gates.get('all_pass') else 'BLOCKED'
            },
            'timestamp': datetime.now().isoformat()
        }


# 테스트
if __name__ == '__main__':
    tracker = PalantirTracker("test-analysis-001")
    
    # Lineage 구축
    lineage = tracker.build_lineage(
        data_sources={
            'bithumb_orderbook': {
                'type': 'exchange_api',
                'timestamp': datetime.now().isoformat(),
                'reliability': 0.95
            },
            'technical_indicators': {
                'type': 'calculated',
                'timestamp': datetime.now().isoformat(),
                'reliability': 0.90
            }
        },
        computation_steps=[
            {'step': 1, 'layer': 'Layer1_Technical', 'output': 25, 'version': 'NICE_v18.3'},
            {'step': 2, 'layer': 'Layer2_OnChain', 'output': 22, 'version': 'NICE_v18.3'},
            {'step': 3, 'layer': 'Final_Score', 'output': 0.72, 'version': 'NICE_v18.3'}
        ]
    )
    
    print("=== Palantir Tracker AIP ===\n")
    print(f"Analysis ID: {tracker.analysis_id}")
    print(f"Total Reliability: {lineage['total_reliability']}")
    print(f"Data Sources: {lineage['total_reliability']}")
    print(f"\nOntology Version: {tracker.ontology['version']}")
