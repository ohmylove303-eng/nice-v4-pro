"""
Protocol Gates v2.6.1
=====================
Fail-Closed 검증 시스템

Gates:
1. Data Integrity - 데이터 무결성 검증
2. Liquidity Guards - 유동성 가드
3. Confirm Gate - 최종 확인 게이트

원칙: 모든 Gate가 PASS 상태가 아니면 진입 금지 (Fail-Closed)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


@dataclass
class GateResult:
    """게이트 검증 결과"""
    status: GateStatus
    reason: str
    missing_fields: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'reason': self.reason,
            'missing_fields': self.missing_fields,
            'metrics': self.metrics
        }


class ProtocolGates:
    """
    Protocol v2.6.1 Gate 검증 시스템
    
    Fail-Closed 원칙:
    - 모든 게이트가 PASS 상태일 때만 진입 허용
    - 하나라도 FAIL이면 자동 차단
    - 데이터 불완전 시 FAIL 처리
    """
    
    # 필수 데이터 필드
    REQUIRED_ORDERBOOK_FIELDS = ['bid_price', 'ask_price', 'bid_volume', 'ask_volume']
    REQUIRED_INDICATOR_FIELDS = ['rsi', 'macd', 'macd_signal']
    REQUIRED_ONCHAIN_FIELDS = ['mvrv', 'fear_greed']
    
    # 유동성 임계값
    MIN_BID_ASK_RATIO = 0.5  # 매수/매도 비율 최소값
    MAX_SPREAD_PCT = 0.5     # 스프레드 최대 0.5%
    MIN_VOLUME_24H = 10000   # 최소 24시간 거래량 (USD)
    
    def __init__(self):
        self.gate_history: List[Dict] = []
    
    def check_all_gates(self, realtime_data: Dict, nice_analysis: Dict) -> Dict:
        """
        모든 게이트 검증
        
        Returns:
            {
                'all_pass': bool,
                'fail_closed_active': bool,
                'data_integrity': GateResult,
                'liquidity_guards': GateResult,
                'confirm_gate': GateResult,
                'timestamp': str
            }
        """
        # 각 게이트 검증
        data_integrity = self.check_data_integrity(realtime_data)
        liquidity_guards = self.check_liquidity_guards(realtime_data)
        confirm_gate = self.check_confirm_gate(realtime_data, nice_analysis)
        
        # 전체 통과 여부
        all_pass = all([
            data_integrity.status == GateStatus.PASS,
            liquidity_guards.status == GateStatus.PASS,
            confirm_gate.status == GateStatus.PASS
        ])
        
        # Fail-Closed 활성화 여부
        fail_closed = not all_pass
        
        result = {
            'all_pass': all_pass,
            'fail_closed_active': fail_closed,
            'data_integrity': data_integrity.to_dict(),
            'liquidity_guards': liquidity_guards.to_dict(),
            'confirm_gate': confirm_gate.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 히스토리 저장
        self.gate_history.append(result)
        
        return result
    
    def check_data_integrity(self, realtime_data: Dict) -> GateResult:
        """
        데이터 무결성 검증
        
        검증 항목:
        - 호가 데이터 완전성
        - 지표 데이터 완전성
        - 온체인 데이터 완전성
        - 타임스탬프 유효성
        """
        missing_fields = []
        
        # 호가 데이터 검증
        orderbook = realtime_data.get('orderbook', {})
        for field in self.REQUIRED_ORDERBOOK_FIELDS:
            if field not in orderbook or orderbook.get(field) is None:
                missing_fields.append(f'orderbook.{field}')
        
        # 지표 데이터 검증
        indicators = realtime_data.get('indicators', {})
        for field in self.REQUIRED_INDICATOR_FIELDS:
            if field not in indicators or indicators.get(field) is None:
                missing_fields.append(f'indicators.{field}')
        
        # 온체인 데이터 검증
        onchain = realtime_data.get('onchain', {})
        for field in self.REQUIRED_ONCHAIN_FIELDS:
            if field not in onchain or onchain.get(field) is None:
                missing_fields.append(f'onchain.{field}')
        
        # 타임스탬프 검증
        timestamp = realtime_data.get('timestamp')
        if not timestamp:
            missing_fields.append('timestamp')
        
        if missing_fields:
            return GateResult(
                status=GateStatus.FAIL,
                reason=f"필수 데이터 누락: {', '.join(missing_fields[:3])}{'...' if len(missing_fields) > 3 else ''}",
                missing_fields=missing_fields,
                metrics={'total_missing': len(missing_fields)}
            )
        
        return GateResult(
            status=GateStatus.PASS,
            reason="모든 필수 데이터 확인 완료",
            metrics={'fields_validated': len(self.REQUIRED_ORDERBOOK_FIELDS) + 
                     len(self.REQUIRED_INDICATOR_FIELDS) + 
                     len(self.REQUIRED_ONCHAIN_FIELDS)}
        )
    
    def check_liquidity_guards(self, realtime_data: Dict) -> GateResult:
        """
        유동성 가드 검증
        
        검증 항목:
        - 매수/매도 비율
        - 스프레드
        - 24시간 거래량
        """
        orderbook = realtime_data.get('orderbook', {})
        ticker = realtime_data.get('ticker', {})
        
        # 데이터 추출
        bid_volume = orderbook.get('bid_volume', 0)
        ask_volume = orderbook.get('ask_volume', 0)
        bid_price = orderbook.get('bid_price', 0)
        ask_price = orderbook.get('ask_price', 0)
        volume_24h = ticker.get('volume_24h', 0)
        
        issues = []
        metrics = {}
        
        # 매수/매도 비율 검증
        if ask_volume > 0:
            bid_ask_ratio = bid_volume / ask_volume
            metrics['bid_ask_ratio'] = round(bid_ask_ratio, 2)
            if bid_ask_ratio < self.MIN_BID_ASK_RATIO:
                issues.append(f"매수/매도 비율 낮음: {bid_ask_ratio:.2f} < {self.MIN_BID_ASK_RATIO}")
        else:
            issues.append("매도 호가 없음")
        
        # 스프레드 검증
        if bid_price > 0:
            spread_pct = ((ask_price - bid_price) / bid_price) * 100
            metrics['spread_pct'] = round(spread_pct, 3)
            if spread_pct > self.MAX_SPREAD_PCT:
                issues.append(f"스프레드 과다: {spread_pct:.2f}% > {self.MAX_SPREAD_PCT}%")
        
        # 거래량 검증
        metrics['volume_24h'] = volume_24h
        if volume_24h < self.MIN_VOLUME_24H:
            issues.append(f"거래량 부족: ${volume_24h:,.0f} < ${self.MIN_VOLUME_24H:,.0f}")
        
        if issues:
            return GateResult(
                status=GateStatus.FAIL,
                reason=" | ".join(issues),
                metrics=metrics
            )
        
        return GateResult(
            status=GateStatus.PASS,
            reason="유동성 조건 충족",
            metrics=metrics
        )
    
    def check_confirm_gate(self, realtime_data: Dict, nice_analysis: Dict) -> GateResult:
        """
        최종 확인 게이트
        
        검증 항목:
        - NICE 점수 임계값
        - 신뢰도 임계값
        - 레이어 일관성
        """
        nice_score = nice_analysis.get('score', 0)
        confidence = nice_analysis.get('confidence', 0)
        layers = nice_analysis.get('layers', {})
        
        issues = []
        metrics = {
            'nice_score': nice_score,
            'confidence': confidence
        }
        
        # NICE 점수 검증 (0.40 이상)
        if nice_score < 0.40:
            issues.append(f"NICE 점수 미달: {nice_score:.2f} < 0.40")
        
        # 신뢰도 검증 (50% 이상)
        if confidence < 0.50:
            issues.append(f"신뢰도 미달: {confidence:.1%} < 50%")
        
        # 레이어 일관성 검증 (최소 3개 레이어가 긍정)
        positive_layers = 0
        for layer_name, layer_data in layers.items():
            if isinstance(layer_data, dict):
                score = layer_data.get('score', 0)
                max_score = layer_data.get('max', 30)
                if max_score > 0 and (score / max_score) >= 0.5:
                    positive_layers += 1
        
        metrics['positive_layers'] = positive_layers
        if positive_layers < 3:
            issues.append(f"긍정 레이어 부족: {positive_layers}/5")
        
        if issues:
            return GateResult(
                status=GateStatus.FAIL,
                reason=" | ".join(issues),
                metrics=metrics
            )
        
        return GateResult(
            status=GateStatus.PASS,
            reason="모든 확인 조건 충족",
            metrics=metrics
        )
    
    def calculate_oco_orders(
        self,
        symbol: str,
        strategy: str,  # 'pullback' or 'breakout'
        current_price: float,
        support: float,
        resistance: float,
        atr: float,
        tick_size: float = 1.0
    ) -> Dict:
        """
        OCO (One-Cancels-Other) 주문 계산
        
        조건:
        - Risk:Reward >= 2.0
        - Stop Loss는 ATR 기반
        - Take Profit은 지지/저항 기반
        """
        if strategy == 'pullback':
            # 눌림목 전략: 지지선 근처에서 매수
            entry_price = support * 1.005  # 지지선 +0.5%
            stop_loss = support * 0.98     # 지지선 -2%
            take_profit = resistance * 0.98  # 저항선 -2%
        else:
            # 돌파 전략: 저항선 돌파 시 매수
            entry_price = resistance * 1.005  # 저항선 +0.5%
            stop_loss = resistance * 0.97     # 저항선 -3%
            take_profit = resistance * 1.06   # 저항선 +6%
        
        # Tick size 정렬
        entry_price = round(entry_price / tick_size) * tick_size
        stop_loss = round(stop_loss / tick_size) * tick_size
        take_profit = round(take_profit / tick_size) * tick_size
        
        # Risk:Reward 계산
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        # RR >= 2.0 검증
        valid = rr_ratio >= 2.0
        
        return {
            'symbol': symbol,
            'strategy': strategy,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_usd': risk,
            'reward_usd': reward,
            'risk_reward_ratio': round(rr_ratio, 2),
            'valid': valid,
            'reason': f"RR {rr_ratio:.2f}" if valid else f"RR {rr_ratio:.2f} < 2.0 (불가)"
        }


# 테스트
if __name__ == '__main__':
    gates = ProtocolGates()
    
    # 샘플 데이터
    sample_realtime = {
        'timestamp': datetime.now().isoformat(),
        'orderbook': {
            'bid_price': 97500,
            'ask_price': 97600,
            'bid_volume': 100,
            'ask_volume': 80
        },
        'ticker': {
            'volume_24h': 25000000
        },
        'indicators': {
            'rsi': 65,
            'macd': 150,
            'macd_signal': 100
        },
        'onchain': {
            'mvrv': 2.1,
            'fear_greed': 55
        }
    }
    
    sample_nice = {
        'score': 0.72,
        'confidence': 0.68,
        'layers': {
            'technical': {'score': 25, 'max': 30},
            'onchain': {'score': 22, 'max': 30},
            'sentiment': {'score': 18, 'max': 30},
            'macro': {'score': 20, 'max': 30},
            'institutional': {'score': 25, 'max': 30}
        }
    }
    
    result = gates.check_all_gates(sample_realtime, sample_nice)
    
    print("=== Protocol Gates v2.6.1 ===\n")
    print(f"All Pass: {result['all_pass']}")
    print(f"Fail-Closed Active: {result['fail_closed_active']}")
    print(f"\nData Integrity: {result['data_integrity']['status']}")
    print(f"Liquidity Guards: {result['liquidity_guards']['status']}")
    print(f"Confirm Gate: {result['confirm_gate']['status']}")
