"""
NICE v4 Kelly % Calculator
==========================
원본 NICE 모델 Kelly 공식 (100% 보존)

Kelly % = (p × b - q) / b
- p = 승률 (Type A: 75%, Type B: 60%, Type C: 45%)
- q = 패률 (1 - p)
- b = 승/패 비율 (RR ratio)

실제 적용:
- Type A: 4% (보수적 적용)
- Type B: 2%
- Type C: 0% (거래 금지)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class KellyResult:
    """Kelly % 계산 결과"""
    kelly_full: float      # 원래 Kelly %
    kelly_safe: float      # 보수적 Kelly % (1/4 적용)
    recommended: float     # NICE 권장 % (Type별 고정값)
    position_size: float   # 실제 포지션 크기 ($)
    capital: float
    
    def to_dict(self) -> dict:
        return {
            'kelly_full_pct': round(self.kelly_full, 2),
            'kelly_safe_pct': round(self.kelly_safe, 2),
            'nice_recommended_pct': self.recommended,
            'position_size_usd': round(self.position_size, 2),
            'capital_usd': self.capital
        }


class KellyCalculator:
    """
    NICE v4 Kelly % 계산기
    
    사용법:
    >>> calc = KellyCalculator(capital=10000)
    >>> result = calc.calculate(signal_type='A')
    >>> print(result.position_size)  # $400
    """
    
    # Type별 승률 (NICE 원본 값)
    WIN_RATES = {
        'A': 0.75,  # 75%
        'B': 0.60,  # 60%
        'C': 0.45   # 45%
    }
    
    # Type별 RR (Risk-Reward) 비율
    RR_RATIOS = {
        'A': 2.0,  # 익절 4% / 손절 2%
        'B': 2.0,  # 익절 3% / 손절 1.5%
        'C': 1.0   # 거래 금지
    }
    
    # NICE 권장 Kelly % (보수적 고정값)
    NICE_KELLY = {
        'A': 4.0,
        'B': 2.0,
        'C': 0.0
    }
    
    def __init__(self, capital: float = 10000.0):
        """
        Args:
            capital: 총 자본금 ($)
        """
        self.capital = capital
    
    def calculate(self, signal_type: str, custom_win_rate: Optional[float] = None) -> KellyResult:
        """
        Kelly % 계산
        
        Args:
            signal_type: 'A', 'B', or 'C'
            custom_win_rate: 커스텀 승률 (없으면 Type별 기본값)
            
        Returns:
            KellyResult: 계산 결과
        """
        signal_type = signal_type.upper()
        
        if signal_type not in self.WIN_RATES:
            raise ValueError(f"Invalid signal type: {signal_type}")
        
        # 승률 및 RR 설정
        p = custom_win_rate if custom_win_rate else self.WIN_RATES[signal_type]
        q = 1 - p
        b = self.RR_RATIOS[signal_type]
        
        # Kelly 공식: (p × b - q) / b
        if b == 0:
            kelly_full = 0.0
        else:
            kelly_full = ((p * b) - q) / b
        
        # 보수적 Kelly (1/4 적용)
        kelly_safe = kelly_full * 0.25
        
        # NICE 권장값 (고정)
        recommended = self.NICE_KELLY[signal_type]
        
        # 포지션 크기 계산
        position_size = (recommended / 100) * self.capital
        
        return KellyResult(
            kelly_full=kelly_full * 100,  # % 변환
            kelly_safe=kelly_safe * 100,
            recommended=recommended,
            position_size=position_size,
            capital=self.capital
        )
    
    def calculate_position(self, signal_type: str, entry_price: float) -> dict:
        """
        포지션 상세 계산 (진입가 기준)
        
        Args:
            signal_type: 'A', 'B', or 'C'
            entry_price: 진입 가격
            
        Returns:
            dict: 포지션 정보 (크기, 손절가, 익절가)
        """
        kelly_result = self.calculate(signal_type)
        
        # 손절/익절 비율
        stop_loss_pct = {'A': 2.0, 'B': 1.5, 'C': 0.0}[signal_type.upper()]
        take_profit_pct = {'A': 4.0, 'B': 3.0, 'C': 0.0}[signal_type.upper()]
        
        return {
            'signal_type': signal_type.upper(),
            'capital': self.capital,
            'kelly_pct': kelly_result.recommended,
            'position_size_usd': kelly_result.position_size,
            'entry_price': entry_price,
            'stop_loss_price': entry_price * (1 - stop_loss_pct / 100),
            'take_profit_price': entry_price * (1 + take_profit_pct / 100),
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'rr_ratio': take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
        }


# 테스트용
if __name__ == '__main__':
    calc = KellyCalculator(capital=10000)
    
    print("=== Kelly % Calculator Test ===\n")
    
    for sig_type in ['A', 'B', 'C']:
        result = calc.calculate(sig_type)
        print(f"Type {sig_type}:")
        print(f"  Full Kelly: {result.kelly_full:.2f}%")
        print(f"  Safe Kelly: {result.kelly_safe:.2f}%")
        print(f"  NICE Recommended: {result.recommended}%")
        print(f"  Position Size: ${result.position_size:.2f}")
        print()
    
    # 포지션 예시 (BTC $96,000 진입)
    print("=== Position Example (BTC @ $96,000) ===\n")
    pos = calc.calculate_position('A', entry_price=96000)
    print(f"Entry: ${pos['entry_price']:,.0f}")
    print(f"Position: ${pos['position_size_usd']:.0f}")
    print(f"Stop Loss: ${pos['stop_loss_price']:,.0f} (-{pos['stop_loss_pct']}%)")
    print(f"Take Profit: ${pos['take_profit_price']:,.0f} (+{pos['take_profit_pct']}%)")
    print(f"RR Ratio: {pos['rr_ratio']:.1f}")
