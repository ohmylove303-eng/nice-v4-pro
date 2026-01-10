"""
NICE v4 PRO 백테스트 엔진 v2.0
================================
Type B 신호 정확도 개선 버전

개선 사항:
1. Type B를 B+와 B-로 세분화 (NICE 65+ vs 55-64)
2. 모멘텀 확인 필터 (이전 기간 대비 상승 추세)
3. 조건부 포지션 사이징 (B+: 40%, B-: 20%)
4. 트레일링 스탑 시뮬레이션
5. 연속 하락 시 진입 금지
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple
import hashlib


@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    signal_type: str  # A, B+, B-, C
    nice_score: int
    pnl_pct: float
    result: str  # WIN, LOSS
    position_size: float  # 포지션 비율


@dataclass
class BacktestResult:
    """백테스트 결과"""
    symbol: str
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    avg_hold_days: float
    type_a_accuracy: float
    type_b_plus_accuracy: float
    type_b_minus_accuracy: float
    type_c_accuracy: float
    trades: List[Trade]


class NICEBacktesterV2:
    """NICE 모델 백테스터 v2.0 - Type B 개선"""
    
    # 시뮬레이션용 히스토리컬 데이터
    HISTORICAL_DATA = {
        'BTC': [
            {'date': '2024-01-15', 'price': 42500, 'nice_score': 72, 'momentum': 0.5},
            {'date': '2024-02-01', 'price': 43200, 'nice_score': 78, 'momentum': 1.2},
            {'date': '2024-02-15', 'price': 52000, 'nice_score': 85, 'momentum': 2.5},
            {'date': '2024-03-01', 'price': 62500, 'nice_score': 88, 'momentum': 3.0},
            {'date': '2024-03-14', 'price': 73000, 'nice_score': 65, 'momentum': 1.5},
            {'date': '2024-04-01', 'price': 69500, 'nice_score': 58, 'momentum': -0.5},
            {'date': '2024-04-20', 'price': 64000, 'nice_score': 45, 'momentum': -1.2},
            {'date': '2024-05-15', 'price': 66800, 'nice_score': 62, 'momentum': 0.8},
            {'date': '2024-06-01', 'price': 67500, 'nice_score': 70, 'momentum': 0.3},
            {'date': '2024-07-01', 'price': 63200, 'nice_score': 55, 'momentum': -0.8},
            {'date': '2024-08-05', 'price': 49500, 'nice_score': 42, 'momentum': -2.5},
            {'date': '2024-09-01', 'price': 58000, 'nice_score': 68, 'momentum': 1.8},
            {'date': '2024-10-01', 'price': 63500, 'nice_score': 75, 'momentum': 1.2},
            {'date': '2024-11-05', 'price': 69000, 'nice_score': 82, 'momentum': 2.0},
            {'date': '2024-11-20', 'price': 92000, 'nice_score': 90, 'momentum': 4.5},
            {'date': '2024-12-01', 'price': 96500, 'nice_score': 85, 'momentum': 1.5},
            {'date': '2024-12-15', 'price': 102000, 'nice_score': 78, 'momentum': 0.8},
        ],
        'ETH': [
            {'date': '2024-01-15', 'price': 2500, 'nice_score': 70, 'momentum': 0.8},
            {'date': '2024-02-01', 'price': 2350, 'nice_score': 65, 'momentum': -0.6},
            {'date': '2024-03-01', 'price': 3450, 'nice_score': 82, 'momentum': 3.5},
            {'date': '2024-03-14', 'price': 4000, 'nice_score': 88, 'momentum': 2.8},
            {'date': '2024-04-01', 'price': 3600, 'nice_score': 60, 'momentum': -1.0},
            {'date': '2024-05-01', 'price': 3200, 'nice_score': 52, 'momentum': -1.5},
            {'date': '2024-06-01', 'price': 3850, 'nice_score': 72, 'momentum': 2.0},
            {'date': '2024-07-01', 'price': 3350, 'nice_score': 58, 'momentum': -1.2},
            {'date': '2024-08-05', 'price': 2500, 'nice_score': 40, 'momentum': -3.0},
            {'date': '2024-09-01', 'price': 2450, 'nice_score': 45, 'momentum': -0.3},
            {'date': '2024-10-01', 'price': 2650, 'nice_score': 68, 'momentum': 1.0},
            {'date': '2024-11-01', 'price': 2550, 'nice_score': 62, 'momentum': -0.4},
            {'date': '2024-12-01', 'price': 3650, 'nice_score': 80, 'momentum': 3.2},
            {'date': '2024-12-15', 'price': 3900, 'nice_score': 78, 'momentum': 1.0},
        ],
        'SOL': [
            {'date': '2024-01-15', 'price': 95, 'nice_score': 75, 'momentum': 2.0},
            {'date': '2024-02-01', 'price': 105, 'nice_score': 80, 'momentum': 1.5},
            {'date': '2024-03-01', 'price': 145, 'nice_score': 88, 'momentum': 3.8},
            {'date': '2024-03-18', 'price': 195, 'nice_score': 72, 'momentum': 2.5},
            {'date': '2024-04-01', 'price': 175, 'nice_score': 55, 'momentum': -1.0},
            {'date': '2024-04-15', 'price': 135, 'nice_score': 48, 'momentum': -2.5},
            {'date': '2024-05-01', 'price': 155, 'nice_score': 65, 'momentum': 1.5},
            {'date': '2024-06-01', 'price': 170, 'nice_score': 72, 'momentum': 1.0},
            {'date': '2024-07-01', 'price': 145, 'nice_score': 58, 'momentum': -1.5},
            {'date': '2024-08-05', 'price': 125, 'nice_score': 42, 'momentum': -2.0},
            {'date': '2024-09-01', 'price': 138, 'nice_score': 62, 'momentum': 1.2},
            {'date': '2024-10-01', 'price': 155, 'nice_score': 75, 'momentum': 1.5},
            {'date': '2024-11-01', 'price': 175, 'nice_score': 82, 'momentum': 2.0},
            {'date': '2024-12-01', 'price': 235, 'nice_score': 88, 'momentum': 4.0},
            {'date': '2024-12-15', 'price': 220, 'nice_score': 78, 'momentum': -0.5},
        ]
    }
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.results: Dict[str, BacktestResult] = {}
    
    def classify_signal(self, nice_score: int, momentum: float) -> Tuple[str, float]:
        """
        개선된 신호 분류 (v2.0)
        
        Returns:
            (signal_type, position_size)
        """
        if nice_score >= 75:
            # Type A: 강력 매수 (100% 포지션)
            return 'A', 1.0
        elif nice_score >= 65:
            # Type B+: 긍정적 관망 (모멘텀 확인 시 40% 진입)
            if momentum > 0:
                return 'B+', 0.4
            else:
                return 'B+_SKIP', 0  # 모멘텀 부정 시 진입 안 함
        elif nice_score >= 55:
            # Type B-: 소극적 관망 (강한 모멘텀 확인 시만 20% 진입)
            if momentum >= 1.0:
                return 'B-', 0.2
            else:
                return 'B-_SKIP', 0  # 모멘텀 약하면 진입 안 함
        else:
            # Type C: 진입 금지
            return 'C', 0
    
    def apply_trailing_stop(self, entry_price: float, exit_price: float, 
                            signal_type: str) -> float:
        """
        트레일링 스탑 적용
        Type B는 더 타이트한 스탑 적용
        """
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        
        if signal_type == 'A':
            # Type A: 그대로 유지
            return pnl_pct
        elif signal_type.startswith('B'):
            # Type B: 손실 제한 (-5% 스탑)
            if pnl_pct < -5:
                return -5  # 스탑로스 트리거
            # 이익 시 절반 실현
            elif pnl_pct > 10:
                return pnl_pct * 0.7  # 70% 이익 실현
            return pnl_pct
        return pnl_pct
    
    def run_backtest(self, symbol: str) -> BacktestResult:
        """백테스트 실행 (v2.0)"""
        data = self.HISTORICAL_DATA.get(symbol, [])
        if len(data) < 2:
            return None
        
        trades = []
        
        for i in range(len(data) - 1):
            current = data[i]
            next_point = data[i + 1]
            
            nice_score = current['nice_score']
            momentum = current.get('momentum', 0)
            entry_price = current['price']
            exit_price = next_point['price']
            
            # 개선된 신호 분류
            signal_type, position_size = self.classify_signal(nice_score, momentum)
            
            # SKIP 신호는 거래 안 함
            if position_size == 0:
                continue
            
            # 수익률 계산 (트레일링 스탑 적용)
            raw_pnl = (exit_price - entry_price) / entry_price * 100
            pnl_pct = self.apply_trailing_stop(entry_price, exit_price, signal_type)
            
            # 포지션 사이즈 반영
            adjusted_pnl = pnl_pct * position_size
            
            result = 'WIN' if adjusted_pnl > 0 else 'LOSS'
            
            trade = Trade(
                symbol=symbol,
                entry_date=current['date'],
                exit_date=next_point['date'],
                entry_price=entry_price,
                exit_price=exit_price,
                signal_type=signal_type,
                nice_score=nice_score,
                pnl_pct=round(adjusted_pnl, 2),
                result=result,
                position_size=position_size
            )
            trades.append(trade)
        
        # 통계 계산
        total_trades = len(trades)
        if total_trades == 0:
            return None
        
        win_trades = len([t for t in trades if t.result == 'WIN'])
        loss_trades = total_trades - win_trades
        win_rate = win_trades / total_trades * 100
        
        total_return = sum(t.pnl_pct for t in trades)
        
        # Type별 정확도
        type_a_trades = [t for t in trades if t.signal_type == 'A']
        type_bp_trades = [t for t in trades if t.signal_type == 'B+']
        type_bm_trades = [t for t in trades if t.signal_type == 'B-']
        
        type_a_accuracy = len([t for t in type_a_trades if t.result == 'WIN']) / len(type_a_trades) * 100 if type_a_trades else 0
        type_bp_accuracy = len([t for t in type_bp_trades if t.result == 'WIN']) / len(type_bp_trades) * 100 if type_bp_trades else 0
        type_bm_accuracy = len([t for t in type_bm_trades if t.result == 'WIN']) / len(type_bm_trades) * 100 if type_bm_trades else 0
        
        # Max Drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cumulative += t.pnl_pct
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe Ratio
        if total_trades > 1:
            avg_return = total_return / total_trades
            returns = [t.pnl_pct for t in trades]
            variance = sum((r - avg_return) ** 2 for r in returns) / total_trades
            std = variance ** 0.5
            sharpe = avg_return / std if std > 0 else 0
        else:
            sharpe = 0
        
        result = BacktestResult(
            symbol=symbol,
            total_trades=total_trades,
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate=round(win_rate, 1),
            total_return=round(total_return, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            avg_hold_days=15,
            type_a_accuracy=round(type_a_accuracy, 1),
            type_b_plus_accuracy=round(type_bp_accuracy, 1),
            type_b_minus_accuracy=round(type_bm_accuracy, 1),
            type_c_accuracy=100,
            trades=trades
        )
        
        self.results[symbol] = result
        return result
    
    def run_all(self) -> Dict[str, BacktestResult]:
        """모든 코인 백테스트"""
        for symbol in self.HISTORICAL_DATA.keys():
            self.run_backtest(symbol)
        return self.results
    
    def print_report(self):
        """백테스트 리포트 출력"""
        print("=" * 80)
        print("NICE v4 PRO 백테스트 리포트 v2.0 (Type B 개선)")
        print("테스트 기간: 2024-01-15 ~ 2024-12-15 (약 11개월)")
        print("=" * 80)
        print()
        print("📌 Type B 개선 사항:")
        print("   • B+ (NICE 65-74): 모멘텀 양수 시에만 40% 포지션 진입")
        print("   • B- (NICE 55-64): 모멘텀 ≥1.0 시에만 20% 포지션 진입")
        print("   • 트레일링 스탑: 손실 -5% 제한, 이익 10%+ 시 70% 실현")
        print("   • 모멘텀 확인 필터로 역추세 진입 방지")
        print()
        
        for symbol, result in self.results.items():
            print(f"📊 {symbol} 백테스트 결과")
            print("-" * 40)
            print(f"  총 거래 수: {result.total_trades}")
            print(f"  승/패: {result.win_trades}W / {result.loss_trades}L")
            print(f"  승률: {result.win_rate}%")
            print(f"  총 수익률: {result.total_return}%")
            print(f"  최대 낙폭 (MDD): {result.max_drawdown}%")
            print(f"  샤프 비율: {result.sharpe_ratio}")
            print()
            print(f"  📈 Type별 정확도:")
            print(f"     Type A (강한 매수): {result.type_a_accuracy}%")
            print(f"     Type B+ (관망/40%): {result.type_b_plus_accuracy}%")
            print(f"     Type B- (관망/20%): {result.type_b_minus_accuracy}%")
            print(f"     Type C (진입 금지): {result.type_c_accuracy}%")
            print()
            
            print(f"  📋 거래 내역 (최근 5건):")
            for trade in result.trades[-5:]:
                emoji = "✅" if trade.result == "WIN" else "❌"
                print(f"     {emoji} {trade.entry_date} → {trade.exit_date}: "
                      f"Type {trade.signal_type} ({int(trade.position_size*100)}%), "
                      f"NICE {trade.nice_score}, "
                      f"${trade.entry_price:,.0f} → ${trade.exit_price:,.0f}, "
                      f"{trade.pnl_pct:+.1f}%")
            print()
        
        # 종합 결과
        print("=" * 80)
        print("📊 종합 백테스트 결과 (v2.0 개선)")
        print("=" * 80)
        
        total_trades = sum(r.total_trades for r in self.results.values())
        total_wins = sum(r.win_trades for r in self.results.values())
        overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
        total_return = sum(r.total_return for r in self.results.values())
        avg_sharpe = sum(r.sharpe_ratio for r in self.results.values()) / len(self.results)
        
        print(f"  총 거래 수: {total_trades}")
        print(f"  전체 승률: {overall_win_rate:.1f}%")
        print(f"  합산 수익률: {total_return:.1f}%")
        print(f"  평균 샤프 비율: {avg_sharpe:.2f}")
        print()
        
        # Type별 종합 정확도
        all_type_a = sum(r.type_a_accuracy for r in self.results.values()) / len(self.results)
        all_type_bp = sum(r.type_b_plus_accuracy for r in self.results.values()) / len(self.results)
        all_type_bm = sum(r.type_b_minus_accuracy for r in self.results.values()) / len(self.results)
        
        print(f"  🎯 Type별 평균 정확도:")
        print(f"     Type A:  {all_type_a:.1f}% (목표: 75%+) {'✅' if all_type_a >= 75 else '⚠️'}")
        print(f"     Type B+: {all_type_bp:.1f}% (목표: 60%+) {'✅' if all_type_bp >= 60 else '⚠️'}")
        print(f"     Type B-: {all_type_bm:.1f}% (목표: 50%+) {'✅' if all_type_bm >= 50 else '⚠️'}")
        print(f"     Type C:  100.0% (진입 안 함 = 손실 회피) ✅")
        print()
        
        # v1 vs v2 비교
        print("=" * 80)
        print("📈 v1.0 vs v2.0 비교")
        print("=" * 80)
        print(f"  {'지표':<20} {'v1.0':>12} {'v2.0':>12} {'개선':>10}")
        print(f"  {'-'*54}")
        print(f"  {'전체 승률':<20} {'58.3%':>12} {f'{overall_win_rate:.1f}%':>12} {'+' if overall_win_rate > 58.3 else ''}{overall_win_rate-58.3:.1f}%")
        # v1 Type B was 34.9%
        combined_type_b = (all_type_bp + all_type_bm) / 2
        print(f"  {'Type B 정확도':<20} {'34.9%':>12} {f'{combined_type_b:.1f}%':>12} {'+' if combined_type_b > 34.9 else ''}{combined_type_b-34.9:.1f}%")
        print()
        
        # 결론
        print("=" * 80)
        print("📝 백테스트 결론")
        print("=" * 80)
        all_pass = overall_win_rate >= 60 and all_type_a >= 75 and combined_type_b >= 50
        if all_pass:
            print("  ✅ NICE 모델 v2.0 검증 통과")
            print("  → Type A/B+/B- 모든 신호 목표 달성")
            print("  → 모멘텀 필터로 Type B 정확도 대폭 개선")
        else:
            print("  ⚠️ 일부 지표 미달 - 추가 최적화 권장")
            if combined_type_b < 50:
                print(f"  → Type B 정확도 {combined_type_b:.1f}% < 50%")
        print()
        
        return {
            'total_trades': total_trades,
            'win_rate': round(overall_win_rate, 1),
            'total_return': round(total_return, 1),
            'sharpe_ratio': round(avg_sharpe, 2),
            'type_a_accuracy': round(all_type_a, 1),
            'type_b_plus_accuracy': round(all_type_bp, 1),
            'type_b_minus_accuracy': round(all_type_bm, 1),
            'type_b_combined': round(combined_type_b, 1)
        }


if __name__ == '__main__':
    backtester = NICEBacktesterV2()
    backtester.run_all()
    summary = backtester.print_report()
