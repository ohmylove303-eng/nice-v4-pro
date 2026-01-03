"""
NICE v4 Coin Analyzer
====================
개별 코인에 대한 NICE 분석 통합 모듈

주요 기능:
1. 코인별 기술분석 데이터 수집
2. 5레이어 NICE 점수 계산
3. Type A/B/C 분류
4. 거래 추천 생성
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import random

from .scorer import NICEScorer, NICEScore
from .classifier import NICEClassifier, NICESignal, SignalType
from .kelly import KellyCalculator, KellyResult
from .data_collector import NICEDataCollector, NICEData


@dataclass
class CoinNICEResult:
    """코인 NICE 분석 결과"""
    symbol: str
    name: str
    price: float
    change_24h: float
    
    # NICE 점수
    nice_score: NICEScore
    normalized_score: float
    
    # 분류 결과
    signal: NICESignal
    signal_type: str  # A, B, C
    
    # Kelly 결과
    kelly: KellyResult
    
    # 원본 데이터
    data: NICEData
    
    # 거래 추천
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # 메타
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'price': self.price,
            'change_24h': self.change_24h,
            
            'nice': {
                'total_score': round(self.normalized_score, 1),
                'raw_score': round(self.nice_score.total_raw, 1),
                'layers': self.nice_score.to_dict()['layers']
            },
            
            'signal': {
                'type': self.signal_type,
                'confidence': self.signal.confidence,
                'action': self.signal.action
            },
            
            'kelly': {
                'full_pct': round(self.kelly.kelly_full, 2),
                'safe_pct': round(self.kelly.kelly_safe, 2),
                'recommended_pct': self.kelly.recommended,
                'position_size_usd': round(self.kelly.position_size, 2)
            },
            
            'trading': {
                'entry_price': self.entry_price,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'stop_loss_pct': self.signal.stop_loss_pct,
                'take_profit_pct': self.signal.take_profit_pct,
                'time_stop_minutes': self.signal.time_stop_minutes
            },
            
            'reasons': self.signal.reasons,
            'data': self.data.to_dict(),
            'timestamp': self.timestamp.isoformat()
        }


class CoinNICEAnalyzer:
    """
    코인별 NICE 분석기
    
    사용법:
    >>> analyzer = CoinNICEAnalyzer(capital=10000)
    >>> result = analyzer.analyze('BTC', price=98000, change_24h=2.5)
    >>> print(result.normalized_score)  # 0-100
    >>> print(result.signal_type)  # A, B, C
    """
    
    # 메이저 코인 (시총 상위)
    MAJOR_COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'AVAX', 'DOGE', 'DOT', 'LINK']
    
    # 코인 이름
    COIN_NAMES = {
        'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'XRP': 'Ripple',
        'BNB': 'BNB', 'ADA': 'Cardano', 'AVAX': 'Avalanche', 'DOGE': 'Dogecoin',
        'DOT': 'Polkadot', 'LINK': 'Chainlink', 'MATIC': 'Polygon', 'ATOM': 'Cosmos',
        'UNI': 'Uniswap', 'PEPE': 'Pepe', 'SHIB': 'Shiba Inu', 'APT': 'Aptos',
        'SUI': 'Sui', 'OP': 'Optimism', 'ARB': 'Arbitrum', 'NEAR': 'Near'
    }
    
    def __init__(self, capital: float = 10000.0):
        """
        Args:
            capital: 총 자본금 ($)
        """
        self.capital = capital
        self.scorer = NICEScorer()
        self.classifier = NICEClassifier()
        self.kelly_calc = KellyCalculator(capital=capital)
        
    def analyze(self, symbol: str, price: float = None, change_24h: float = None,
                data: NICEData = None) -> CoinNICEResult:
        """
        코인 NICE 분석 실행
        
        Args:
            symbol: 코인 심볼 (BTC, ETH 등)
            price: 현재가 (없으면 자동 수집)
            change_24h: 24시간 변동률 (없으면 랜덤)
            data: 외부 NICEData (없으면 자동 수집)
            
        Returns:
            CoinNICEResult: 분석 결과
        """
        symbol = symbol.upper()
        
        # 1. 데이터 수집
        if data is None:
            collector = NICEDataCollector(symbol=symbol)
            data = collector.collect_all()
        
        # 가격 설정
        if price is None:
            price = data.technical.price or self._get_default_price(symbol)
        
        if change_24h is None:
            change_24h = random.uniform(-5, 10)
        
        # 2. NICE 점수 계산
        scorer_data = data.to_scorer_format()
        nice_score = self.scorer.calculate(scorer_data)
        normalized = nice_score.total_normalized
        
        # 3. Type 분류
        layer_details = nice_score.to_dict()['layers']
        signal = self.classifier.classify(score=normalized, layer_details=layer_details)
        
        # 4. Kelly 계산
        kelly = self.kelly_calc.calculate(signal_type=signal.signal_type.value)
        
        # 5. 거래가 계산
        entry = price * 0.995  # 현재가 기준 진입
        sl = price * (1 - signal.stop_loss_pct / 100)
        tp = price * (1 + signal.take_profit_pct / 100)
        
        return CoinNICEResult(
            symbol=symbol,
            name=self.COIN_NAMES.get(symbol, symbol),
            price=price,
            change_24h=change_24h,
            nice_score=nice_score,
            normalized_score=normalized,
            signal=signal,
            signal_type=signal.signal_type.value,
            kelly=kelly,
            data=data,
            entry_price=round(entry, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6)
        )
    
    def analyze_multiple(self, symbols: List[str]) -> List[CoinNICEResult]:
        """
        여러 코인 동시 분석
        
        Args:
            symbols: 코인 심볼 리스트
            
        Returns:
            List[CoinNICEResult]: 분석 결과 리스트 (점수 내림차순)
        """
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze(symbol)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        # 점수 내림차순 정렬
        results.sort(key=lambda x: x.normalized_score, reverse=True)
        
        return results
    
    def get_top_signals(self, limit: int = 5) -> List[CoinNICEResult]:
        """
        상위 N개 신호 코인 반환
        
        Args:
            limit: 반환할 코인 수
            
        Returns:
            List[CoinNICEResult]: 상위 코인 분석 결과
        """
        results = self.analyze_multiple(self.MAJOR_COINS)
        return results[:limit]
    
    def get_type_a_coins(self) -> List[CoinNICEResult]:
        """Type A 코인만 필터링"""
        results = self.analyze_multiple(self.MAJOR_COINS)
        return [r for r in results if r.signal_type == 'A']
    
    def _get_default_price(self, symbol: str) -> float:
        """기본 가격 반환"""
        prices = {
            'BTC': 98000, 'ETH': 3500, 'SOL': 195, 'XRP': 2.35, 'DOGE': 0.38,
            'BNB': 680, 'ADA': 1.05, 'AVAX': 42, 'LINK': 28, 'DOT': 9.5,
            'MATIC': 0.95, 'ATOM': 11, 'UNI': 14, 'PEPE': 0.0000195, 'SHIB': 0.0000285
        }
        return prices.get(symbol, 100)


class NICEMarketAnalyzer:
    """
    NICE 기반 시장 전체 분석기
    
    시장 상태 판단 및 전체 점수 계산
    """
    
    def __init__(self):
        self.collector = NICEDataCollector(symbol='BTC')
        self.scorer = NICEScorer()
        self.classifier = NICEClassifier()
    
    def analyze_market(self) -> Dict:
        """
        시장 전체 NICE 분석
        
        Returns:
            Dict: 시장 분석 결과
        """
        # BTC 기준 데이터 수집
        data = self.collector.collect_all()
        
        # 점수 계산
        score = self.scorer.calculate(data.to_scorer_format())
        normalized = score.total_normalized
        
        # 분류
        signal = self.classifier.classify(normalized)
        
        # 시장 상태 결정
        if normalized >= 75:
            market_state = 'STRONG_BULL'
            recommendation = '적극 매수'
        elif normalized >= 60:
            market_state = 'BULL'
            recommendation = '매수 고려'
        elif normalized >= 45:
            market_state = 'NEUTRAL'
            recommendation = '관망'
        elif normalized >= 30:
            market_state = 'BEAR'
            recommendation = '매도 고려'
        else:
            market_state = 'STRONG_BEAR'
            recommendation = '매수 금지'
        
        return {
            'market_state': market_state,
            'recommendation': recommendation,
            'total_score': round(normalized, 1),
            'signal_type': signal.signal_type.value,
            'signal_confidence': signal.confidence,
            'layers': score.to_dict()['layers'],
            'data': {
                'fear_greed': data.sentiment.fear_greed,
                'btc_etf_flow': data.etf.btc_etf_flow_m,
                'whale_flow': data.onchain.net_flow,
                'fed_stance': data.macro.fed_stance
            },
            'timestamp': datetime.now().isoformat()
        }


# 테스트용
if __name__ == '__main__':
    print("=== NICE Coin Analyzer Test ===\n")
    
    analyzer = CoinNICEAnalyzer(capital=10000)
    
    # 단일 코인 분석
    btc_result = analyzer.analyze('BTC', price=98000, change_24h=2.5)
    print(f"BTC Analysis:")
    print(f"  NICE Score: {btc_result.normalized_score:.1f}/100")
    print(f"  Signal: Type {btc_result.signal_type}")
    print(f"  Kelly: {btc_result.kelly.recommended}% (${btc_result.kelly.position_size:.0f})")
    print(f"  Entry: ${btc_result.entry_price:,.2f}")
    print(f"  SL: ${btc_result.stop_loss:,.2f} | TP: ${btc_result.take_profit:,.2f}")
    print(f"  Reasons: {btc_result.signal.reasons[:2]}")
    
    print("\n=== Top 5 Signals ===\n")
    
    top_signals = analyzer.get_top_signals(limit=5)
    for i, result in enumerate(top_signals, 1):
        type_emoji = '🟢' if result.signal_type == 'A' else ('🟡' if result.signal_type == 'B' else '🔴')
        print(f"{i}. {result.symbol} {type_emoji} - {result.normalized_score:.1f}점 (Type {result.signal_type})")
    
    print("\n=== Market Analysis ===\n")
    
    market_analyzer = NICEMarketAnalyzer()
    market = market_analyzer.analyze_market()
    print(f"Market State: {market['market_state']}")
    print(f"Recommendation: {market['recommendation']}")
    print(f"Total Score: {market['total_score']}/100")
