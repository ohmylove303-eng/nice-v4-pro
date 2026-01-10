"""
Whale Analyzer - 주포 분석 모듈 (결정적 버전)
==============================================
고래(주포) 포진 분석, 프렉탈 핸들링, 유통량 분석
모든 random 제거 - 같은 입력 → 같은 출력 보장
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import hashlib


@dataclass
class WhalePosition:
    """고래 포지션 정보"""
    address_count: int  # 보유 지갑 수
    total_holding: float  # 총 보유량 (%)
    avg_entry_price: float  # 평균 진입가
    sentiment: str  # 'accumulating', 'distributing', 'holding'
    
    
@dataclass
class FractalPattern:
    """프렉탈 패턴 정보"""
    pattern_type: str  # 'higher_high', 'lower_low', 'double_top', 'double_bottom'
    strength: float  # 0-100
    direction: str  # 'bullish', 'bearish', 'neutral'
    key_levels: Dict[str, float]  # support, resistance


@dataclass
class CoinAnalysis:
    """코인 종합 분석 결과"""
    symbol: str
    name: str
    price: float
    change_24h: float
    volume_24h: float
    market_cap: float
    
    # 유통량 분석
    circulating_supply: float
    total_supply: float
    circulation_ratio: float  # 유통 비율 (%)
    
    # 주포 분석
    whale_position: WhalePosition
    
    # 프렉탈 분석
    fractal: FractalPattern
    
    # NICE 점수
    nice_score: int
    nice_type: str  # A, B, C
    
    # 거래 추천
    entry_price: str
    stop_loss: str
    take_profit: str
    kelly_pct: float
    
    # 섹터 (선택)
    sector: str = 'Other'
    
    # ========== API 호환용 속성 ==========
    @property
    def circulating_pct(self) -> float:
        """유통 비율 (%)"""
        return self.circulation_ratio
    
    @property
    def max_supply(self) -> Optional[float]:
        """최대 공급량 (총 공급량 기준)"""
        return self.total_supply
    
    @property
    def nice_signal(self) -> str:
        """NICE 신호 텍스트"""
        if self.nice_type == 'A':
            return '강력 매수'
        elif self.nice_type == 'B':
            return '관망 / 간 보기'
        else:
            return '매수 금지'
    
    @property
    def whale_wallets(self) -> int:
        """고래 지갑 수"""
        return self.whale_position.address_count
    
    @property
    def whale_holding_pct(self) -> float:
        """고래 보유 비율 (%)"""
        return self.whale_position.total_holding
    
    @property
    def whale_strength(self) -> str:
        """고래 포지션 강도"""
        return self.whale_position.sentiment
    
    @property
    def fractal_pattern(self) -> str:
        """프렉탈 패턴 유형"""
        return self.fractal.pattern_type.replace('_', ' ').title()
    
    @property
    def fractal_strength(self) -> float:
        """프렉탈 패턴 강도"""
        return self.fractal.strength
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'price': self.price,
            'change_24h': self.change_24h,
            'volume_24h': self.volume_24h,
            'market_cap': self.market_cap,
            'circulation': {
                'circulating': self.circulating_supply,
                'total': self.total_supply,
                'ratio': self.circulation_ratio
            },
            'whale': {
                'address_count': self.whale_position.address_count,
                'holding_pct': self.whale_position.total_holding,
                'avg_entry': self.whale_position.avg_entry_price,
                'sentiment': self.whale_position.sentiment
            },
            'fractal': {
                'pattern': self.fractal.pattern_type,
                'strength': self.fractal.strength,
                'direction': self.fractal.direction,
                'levels': self.fractal.key_levels
            },
            'nice': {
                'score': self.nice_score,
                'type': self.nice_type,
                'signal': self.nice_signal
            },
            'trading': {
                'entry': self.entry_price,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'kelly_pct': self.kelly_pct
            },
            'sector': self.sector
        }


def _deterministic_hash(symbol: str, seed: str = "") -> int:
    """심볼 기반 결정적 해시 생성 (0-999 범위)"""
    h = hashlib.md5(f"{symbol.upper()}{seed}".encode()).hexdigest()
    return int(h[:8], 16) % 1000


class WhaleAnalyzer:
    """고래(주포) 분석기 - 결정적 버전"""
    
    # 메이저 코인 목록 (시총 상위)
    MAJOR_COINS = [
        'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'AVAX', 'DOGE', 
        'DOT', 'LINK', 'MATIC', 'ATOM', 'UNI', 'LTC', 'ETC'
    ]
    
    # 섹터 분류
    SECTORS = {
        'Layer1': ['BTC', 'ETH', 'SOL', 'AVAX', 'ATOM', 'DOT', 'NEAR', 'SUI', 'APT'],
        'DeFi': ['UNI', 'AAVE', 'LINK', 'CRV', 'MKR', 'COMP', 'SNX', 'SUSHI'],
        'Meme': ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF', 'MEME'],
        'AI': ['FET', 'AGIX', 'OCEAN', 'RNDR', 'TAO', 'ARKM'],
        'Gaming': ['AXS', 'SAND', 'MANA', 'IMX', 'GALA', 'ENJ', 'ILV'],
        'Layer2': ['OP', 'ARB', 'MATIC', 'ZK', 'STRK', 'MANTA'],
        'Exchange': ['BNB', 'CRO', 'KCS', 'OKB', 'GT', 'LEO'],
        'Privacy': ['ZEC', 'XMR', 'DASH', 'ZEN']
    }
    
    # 코인별 고정 데이터 (결정적)
    COIN_META = {
        'BTC': {'wallets': 250, 'holding': 38.5, 'pattern': 'higher_high'},
        'ETH': {'wallets': 220, 'holding': 35.2, 'pattern': 'higher_low'},
        'SOL': {'wallets': 180, 'holding': 42.1, 'pattern': 'higher_high'},
        'XRP': {'wallets': 150, 'holding': 48.3, 'pattern': 'ascending_triangle'},
        'DOGE': {'wallets': 140, 'holding': 45.0, 'pattern': 'higher_low'},
        'BNB': {'wallets': 160, 'holding': 40.5, 'pattern': 'higher_high'},
        'ADA': {'wallets': 130, 'holding': 35.8, 'pattern': 'double_bottom'},
        'AVAX': {'wallets': 120, 'holding': 38.0, 'pattern': 'higher_low'},
        'LINK': {'wallets': 110, 'holding': 41.2, 'pattern': 'ascending_triangle'},
        'DOT': {'wallets': 100, 'holding': 36.5, 'pattern': 'lower_high'},
        'PEPE': {'wallets': 85, 'holding': 55.0, 'pattern': 'higher_high'},
        'SHIB': {'wallets': 90, 'holding': 52.0, 'pattern': 'double_bottom'},
        'APT': {'wallets': 75, 'holding': 48.0, 'pattern': 'higher_low'},
        'SUI': {'wallets': 70, 'holding': 50.0, 'pattern': 'higher_high'},
        'NEAR': {'wallets': 65, 'holding': 45.0, 'pattern': 'ascending_triangle'},
        'WIF': {'wallets': 55, 'holding': 58.0, 'pattern': 'higher_high'},
        'OP': {'wallets': 60, 'holding': 42.0, 'pattern': 'higher_low'},
        'ARB': {'wallets': 58, 'holding': 44.0, 'pattern': 'lower_high'},
        'ZEC': {'wallets': 45, 'holding': 40.0, 'pattern': 'ascending_triangle'},
        'LTC': {'wallets': 95, 'holding': 35.0, 'pattern': 'double_bottom'},
    }
    
    def __init__(self):
        self.cache = {}
        
    def get_sector(self, symbol: str) -> str:
        """코인의 섹터 분류"""
        symbol = symbol.upper()
        for sector, coins in self.SECTORS.items():
            if symbol in coins:
                return sector
        return 'Other'
    
    def is_major(self, symbol: str) -> bool:
        """메이저 코인 여부"""
        return symbol.upper() in self.MAJOR_COINS
    
    def analyze_whale_position(self, symbol: str, price: float, change_24h: float = 0) -> WhalePosition:
        """
        고래 포지션 분석 (결정적)
        - 심볼 기반 고정 데이터 사용
        - change_24h 기반 sentiment 결정
        """
        symbol = symbol.upper()
        meta = self.COIN_META.get(symbol, None)
        
        if meta:
            address_count = meta['wallets']
            holding_pct = meta['holding']
        else:
            # 알려지지 않은 코인: 해시 기반 결정적 값
            h = _deterministic_hash(symbol, 'whale')
            is_major = self.is_major(symbol)
            if is_major:
                address_count = 50 + (h % 150)  # 50-200
                holding_pct = 25 + (h % 200) / 10  # 25-45
            else:
                address_count = 10 + (h % 70)  # 10-80
                holding_pct = 35 + (h % 300) / 10  # 35-65
        
        # 평균 진입가 (현재가 대비 change_24h 기반)
        if change_24h >= 5:
            entry_variance = -0.10  # 수익 중
        elif change_24h >= 0:
            entry_variance = -0.02  # 약간 수익
        elif change_24h >= -5:
            entry_variance = 0.05  # 약간 손실
        else:
            entry_variance = 0.12  # 손실 중
        
        avg_entry = price * (1 + entry_variance)
        
        # 센티먼트 결정 (change_24h 기반)
        if change_24h >= 5:
            sentiment = 'accumulating'  # 축적 중
        elif change_24h >= 0:
            sentiment = 'holding'  # 보유 중
        else:
            sentiment = 'distributing'  # 분배 중
            
        return WhalePosition(
            address_count=address_count,
            total_holding=round(holding_pct, 1),
            avg_entry_price=round(avg_entry, 2),
            sentiment=sentiment
        )
    
    def analyze_fractal(self, symbol: str, price: float, change_24h: float = 0) -> FractalPattern:
        """
        프렉탈 패턴 분석 (결정적)
        - 심볼 기반 기본 패턴 + change_24h 기반 조정
        """
        symbol = symbol.upper()
        meta = self.COIN_META.get(symbol, None)
        
        # 패턴 결정
        if meta:
            base_pattern = meta['pattern']
        else:
            # change_24h 기반 패턴 결정
            if change_24h >= 10:
                base_pattern = 'higher_high'
            elif change_24h >= 5:
                base_pattern = 'higher_low'
            elif change_24h >= 0:
                base_pattern = 'ascending_triangle'
            elif change_24h >= -5:
                base_pattern = 'double_bottom'
            else:
                base_pattern = 'lower_low'
        
        # 방향 결정
        bullish_patterns = ['higher_high', 'higher_low', 'double_bottom', 'ascending_triangle']
        bearish_patterns = ['lower_high', 'lower_low', 'double_top', 'descending_triangle']
        
        if base_pattern in bullish_patterns:
            direction = 'bullish'
        elif base_pattern in bearish_patterns:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # 강도 결정 (change_24h 절대값 기반)
        strength = min(95, max(55, 70 + abs(change_24h) * 2))
        
        # 지지/저항 레벨 계산 (결정적)
        support = price * 0.97   # 현재가 -3%
        resistance = price * 1.04  # 현재가 +4%
        
        return FractalPattern(
            pattern_type=base_pattern,
            strength=round(strength, 1),
            direction=direction,
            key_levels={
                'support': round(support, 6),
                'resistance': round(resistance, 6),
                'pivot': round(price, 6)
            }
        )
    
    def calculate_nice_score(self, symbol: str, change_24h: float, volume_ratio: float) -> tuple:
        """
        NICE 5레이어 점수 계산 (결정적)
        """
        # 기술 점수 (상승률 기반)
        tech_score = min(30, max(0, 15 + change_24h * 2))
        
        # 거래량 점수
        vol_score = min(20, max(0, volume_ratio * 10))
        
        # OnChain 점수 (메이저 코인 우대, 결정적)
        if self.is_major(symbol):
            onchain_score = 20
        else:
            h = _deterministic_hash(symbol, 'onchain')
            onchain_score = 10 + (h % 9)  # 10-18
        
        # 심리 점수 (시장 전반 - 결정적 기본값)
        sentiment_score = 12  # 중립
        
        # 매크로 점수 (시장 전반 - 결정적 기본값)
        macro_score = 8  # 중립
        
        # 기관 점수 (메이저 코인 우대)
        if self.is_major(symbol):
            institutional_score = 12
        else:
            h = _deterministic_hash(symbol, 'inst')
            institutional_score = 3 + (h % 6)  # 3-8
        
        total_score = int(tech_score + vol_score + onchain_score + sentiment_score + macro_score + institutional_score)
        total_score = min(100, max(0, total_score))
        
        # Type 결정
        if total_score >= 75:
            nice_type = 'A'
        elif total_score >= 55:
            nice_type = 'B'
        else:
            nice_type = 'C'
            
        return total_score, nice_type
    
    def analyze_coin(self, symbol: str, name: str = None, **kwargs) -> CoinAnalysis:
        """종합 코인 분석 (결정적)"""
        symbol = symbol.upper()
        
        # 기본값 또는 전달된 값 사용
        price = kwargs.get('price', 100)
        change_24h = kwargs.get('change_24h', 0)
        volume_24h = kwargs.get('volume_24h', 1e6)
        market_cap = kwargs.get('market_cap', 1e9)
        
        # 유통량 (전달된 값 우선 사용)
        total_supply = kwargs.get('total_supply', market_cap / price if price > 0 else 1e9)
        circulating = kwargs.get('circulating', total_supply * 0.8)
        
        if kwargs.get('max_supply') is None:
            # 무한 발행 코인
            circulation_ratio = 100.0
        else:
            max_supply = kwargs.get('max_supply', total_supply)
            circulation_ratio = (circulating / max_supply * 100) if max_supply > 0 else 100.0
        
        circulating_supply = circulating
        
        # 거래량 비율
        volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
        
        # 분석 수행 (결정적)
        whale_position = self.analyze_whale_position(symbol, price, change_24h)
        fractal = self.analyze_fractal(symbol, price, change_24h)
        nice_score, nice_type = self.calculate_nice_score(symbol, change_24h, volume_ratio)
        
        # 거래 추천 계산
        if nice_type == 'A':
            sl_pct, tp_pct, kelly = 0.02, 0.04, 4.0
        elif nice_type == 'B':
            sl_pct, tp_pct, kelly = 0.03, 0.05, 2.5
        else:
            sl_pct, tp_pct, kelly = 0.05, 0.08, 1.0
            
        entry_low = price * 0.995
        entry_high = price * 1.005
        stop_loss = price * (1 - sl_pct)
        take_profit = price * (1 + tp_pct)
        
        return CoinAnalysis(
            symbol=symbol,
            name=name or symbol,
            price=price,
            change_24h=change_24h,
            volume_24h=volume_24h,
            market_cap=market_cap,
            circulating_supply=circulating_supply,
            total_supply=total_supply,
            circulation_ratio=round(circulation_ratio, 1),
            whale_position=whale_position,
            fractal=fractal,
            nice_score=nice_score,
            nice_type=nice_type,
            entry_price=f"{entry_low:,.6f} - {entry_high:,.6f}",
            stop_loss=f"{stop_loss:,.6f}",
            take_profit=f"{take_profit:,.6f}",
            kelly_pct=kelly,
            sector=self.get_sector(symbol)
        )
    
    def rank_coins(self, coins_data: List[Dict], timeframe: str = 'scalp') -> Dict[str, List[Dict]]:
        """
        코인 순위 정렬 (타임프레임별 차별화)
        """
        major_coins = []
        other_coins = []
        
        for coin in coins_data:
            symbol = coin.get('symbol', '').upper()
            
            if timeframe == 'scalp':
                score_boost = (abs(coin.get('change_24h', 0)) * 2) + (coin.get('volume_24h', 0) / 1e9)
            elif timeframe == 'short':
                score_boost = coin.get('change_24h', 0) * 3
            elif timeframe == 'medium':
                score_boost = (coin.get('market_cap', 0) / 1e10) + 10
            else:
                score_boost = (coin.get('market_cap', 0) / 1e9) + 20
                
            # 분석 실행
            analysis = self.analyze_coin(
                symbol=symbol,
                name=coin.get('name', symbol),
                price=coin.get('price', 0),
                change_24h=coin.get('change_24h', 0),
                volume_24h=coin.get('volume_24h', 0),
                market_cap=coin.get('market_cap', 0)
            )
            
            # 점수 조정
            analysis.nice_score = min(99, int(analysis.nice_score * 0.8 + score_boost % 20))
            
            coin_dict = analysis.to_dict()
            coin_dict['sector'] = self.get_sector(symbol)
            coin_dict['is_major'] = self.is_major(symbol)
            
            if self.is_major(symbol):
                major_coins.append(coin_dict)
            else:
                other_coins.append(coin_dict)
        
        # 정렬 로직
        def sort_key(c):
            if timeframe == 'scalp':
                return (-abs(c['change_24h']), -c['volume_24h'], -c['nice']['score'])
            elif timeframe == 'short':
                return (-c['nice']['score'], -c['change_24h'])
            elif timeframe == 'medium':
                return (-c['nice']['score'], -c['market_cap'])
            else:
                return (-c['nice']['score'], -c['market_cap'])
        
        major_coins.sort(key=sort_key)
        other_coins.sort(key=sort_key)
        
        return {
            'major': major_coins[:5],
            'other': other_coins[:5],
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat()
        }


class CryptoFundFlow:
    """암호화폐 자금 흐름 분석"""
    
    def get_fund_flows(self) -> Dict:
        """자금 유입/유출 데이터 (결정적)"""
        
        inflows = [
            {'name': 'BTC ETF', 'type': 'ETF', 'amount': 1800, 'change': '+12%'},
            {'name': 'ETH Staking', 'type': 'Staking', 'amount': 520, 'change': '+8%'},
            {'name': 'SOL DeFi', 'type': 'DeFi', 'amount': 180, 'change': '+15%'},
            {'name': 'LINK Oracle', 'type': 'Oracle', 'amount': 95, 'change': '+6%'},
            {'name': 'AVAX L1', 'type': 'Layer1', 'amount': 72, 'change': '+4%'}
        ]
        
        outflows = [
            {'name': 'Exchange→Wallet', 'type': 'Transfer', 'amount': 450, 'change': '-8%'},
            {'name': 'USDT Flow', 'type': 'Stable', 'amount': 320, 'change': '-5%'},
            {'name': 'Meme Sell', 'type': 'Meme', 'amount': 180, 'change': '-12%'},
            {'name': 'DeFi Exit', 'type': 'DeFi', 'amount': 95, 'change': '-3%'},
            {'name': 'NFT Sales', 'type': 'NFT', 'amount': 45, 'change': '-15%'}
        ]
        
        net_flow = sum(i['amount'] for i in inflows) - sum(o['amount'] for o in outflows)
        
        return {
            'inflows': inflows,
            'outflows': outflows,
            'net_flow': net_flow,
            'market_sentiment': 'Bullish' if net_flow > 0 else 'Neutral',
            'sentiment_score': 65 if net_flow > 500 else 55,
            'timestamp': datetime.now().isoformat()
        }


# 테스트
if __name__ == '__main__':
    analyzer = WhaleAnalyzer()
    
    # 같은 코인 2번 분석 → 같은 결과 확인
    doge1 = analyzer.analyze_coin('DOGE', 'Dogecoin', price=0.38, change_24h=8.5, volume_24h=4e9, market_cap=55e9, max_supply=None)
    doge2 = analyzer.analyze_coin('DOGE', 'Dogecoin', price=0.38, change_24h=8.5, volume_24h=4e9, market_cap=55e9, max_supply=None)
    
    print(f"DOGE Test 1: {doge1.circulation_ratio}%, Whale: {doge1.whale_position.sentiment}, Pattern: {doge1.fractal.pattern_type}")
    print(f"DOGE Test 2: {doge2.circulation_ratio}%, Whale: {doge2.whale_position.sentiment}, Pattern: {doge2.fractal.pattern_type}")
    print(f"Consistent: {doge1.circulation_ratio == doge2.circulation_ratio and doge1.whale_position.sentiment == doge2.whale_position.sentiment}")
