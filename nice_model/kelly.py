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
    
    def calculate_by_coin_type(
        self, 
        signal_type: str, 
        coin_type: str = 'major',
        volatility: str = 'medium',
        confidence: float = 100.0
    ) -> dict:
        """
        메이저/기타 코인별 차등 Kelly % 계산
        
        신규 파일(NICE_v4_Backend_API.py) 로직 적용:
        - 메이저: 승률 68%, RR 1.3 (안정적)
        - 기타: 승률 55%, RR 1.8 (고수익/고위험)
        
        Args:
            signal_type: 'A', 'B', or 'C'
            coin_type: 'major' or 'other'
            volatility: 'low', 'medium', 'high'
            confidence: 신호 신뢰도 (0-100%)
            
        Returns:
            dict: Kelly 계산 결과
        """
        signal_type = signal_type.upper()
        coin_type = coin_type.lower()
        
        # 코인 타입별 승률 및 RR  
        win_rates = {'major': 0.68, 'other': 0.55}
        avg_ratios = {'major': 1.3, 'other': 1.8}
        
        p = win_rates.get(coin_type, 0.60)
        b = avg_ratios.get(coin_type, 1.5)
        q = 1 - p
        
        # 기본 Kelly 계산
        base_kelly = (b * p - q) / b if b > 0 else 0
        
        # 신호 타입별 배수 (A=100%, B=70%, C=40%)
        signal_mult = {'A': 1.0, 'B': 0.7, 'C': 0.4, 'WAIT': 0.0}.get(signal_type, 0)
        
        # 신뢰도 배수 (0-100% → 0-1)
        conf_mult = min(confidence, 100) / 100
        
        # 변동성 배수 (높으면 포지션 축소)
        vol_mult = {'low': 1.0, 'medium': 0.8, 'high': 0.6}.get(volatility, 0.8)
        
        # 최종 Kelly (최대 4% 제한)
        final_kelly = min(base_kelly * 0.25 * signal_mult * conf_mult * vol_mult, 0.04)
        
        return {
            'coin_type': coin_type,
            'signal_type': signal_type,
            'base_kelly_pct': round(base_kelly * 100, 2),
            'final_kelly_pct': round(final_kelly * 100, 2),
            'position_size_usd': round(self.capital * final_kelly, 2),
            'win_rate': p,
            'rr_ratio': b,
            'volatility': volatility,
            'confidence': confidence
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
