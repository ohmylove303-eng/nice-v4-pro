"""
Whale Analyzer - 주포 분석 모듈
================================
고래(주포) 포진 분석, 프렉탈 핸들링, 유통량 분석
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import hashlib


def _det_hash(symbol: str, seed: str = '') -> int:
    """결정적 해시 생성 (0-999)"""
    h = hashlib.md5(f"{symbol.upper()}{seed}".encode()).hexdigest()
    return int(h[:8], 16) % 1000


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


class WhaleAnalyzer:
    """고래(주포) 분석기"""
    
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
        'Exchange': ['BNB', 'CRO', 'KCS', 'OKB', 'GT', 'LEO']
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
    
    def analyze_whale_position(self, symbol: str, price: float) -> WhalePosition:
        """
        고래 포지션 분석
        실제로는 Glassnode, Whale Alert 등 API 연동 필요
        """
        # 시뮬레이션 데이터 (실제 구현 시 API 연동)
        is_major = self.is_major(symbol)
        
        # 결정적: 심볼 기반 고정값
        h = _det_hash(symbol, 'whale')
        if is_major:
            address_count = 50 + (h % 150)  # 50-200
            holding_pct = 25 + (h % 200) / 10  # 25-45
        else:
            address_count = 10 + (h % 70)  # 10-80
            holding_pct = 35 + (h % 300) / 10  # 35-65
        
        # 평균 진입가 추정 (현재가 대비) - 결정적
        entry_variance = -0.15 + (h % 250) / 1000  # -0.15 to +0.10
        avg_entry = price * (1 + entry_variance)
        
        # 센티먼트 결정
        if avg_entry < price * 0.95:
            sentiment = 'accumulating'  # 축적 중
        elif avg_entry > price * 1.05:
            sentiment = 'distributing'  # 분배 중
        else:
            sentiment = 'holding'  # 보유 중
            
        return WhalePosition(
            address_count=address_count,
            total_holding=round(holding_pct, 1),
            avg_entry_price=round(avg_entry, 2),
            sentiment=sentiment
        )
    
    def analyze_fractal(self, symbol: str, price: float) -> FractalPattern:
        """
        프렉탈 패턴 분석
        고점/저점 구조 확인
        """
        # 시뮬레이션 (실제로는 가격 데이터 분석 필요)
        patterns = [
            ('higher_high', 'bullish'),
            ('higher_low', 'bullish'),
            ('lower_high', 'bearish'),
            ('lower_low', 'bearish'),
            ('double_top', 'bearish'),
            ('double_bottom', 'bullish'),
            ('ascending_triangle', 'bullish'),
            ('descending_triangle', 'bearish')
        ]
        
        # 결정적: 심볼 기반 패턴 선택
        h = _det_hash(symbol, 'fractal')
        pattern_type, direction = patterns[h % len(patterns)]
        strength = 55 + (h % 400) / 10  # 55-95
        
        # 지지/저항 레벨 계산 - 결정적
        support = price * (1 - 0.02 - (h % 30) / 1000)  # -2% to -5%
        resistance = price * (1 + 0.02 + (h % 30) / 1000)  # +2% to +5%
        
        return FractalPattern(
            pattern_type=pattern_type,
            strength=round(strength, 1),
            direction=direction,
            key_levels={
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'pivot': round(price, 2)
            }
        )
    
    def calculate_nice_score(
        self, 
        symbol: str, 
        change_24h: float, 
        volume_ratio: float,
        palantir_reliability: float = 0.85
    ) -> tuple:
        """
        NICE 5레이어 + Palantir 신뢰도 점수 계산
        
        Palantir 신뢰도가 높을수록 보너스 점수 부여 (최대 +5점)
        → 데이터 품질이 좋을 때만 높은 점수 획득 가능
        """
        # 1. 기술 점수 (상승률 기반) - 최대 30점
        tech_score = min(30, max(0, 15 + change_24h * 2))
        
        # 2. 거래량 점수 - 최대 20점
        vol_score = min(20, max(0, volume_ratio * 10))
        
        # 3. OnChain 점수 (메이저 코인 우대) - 결정적, 최대 20점
        h = _det_hash(symbol, 'nice')
        onchain_score = 20 if self.is_major(symbol) else 10 + (h % 9)  # 10-18
        
        # 4. 심리/매크로/기관 점수 (공통) - 결정적
        sentiment_score = 8 + (h % 8)  # 8-15
        macro_score = 5 + (h % 6)  # 5-10
        institutional_score = (5 + (h % 11)) if self.is_major(symbol) else (2 + (h % 7))  # 5-15 or 2-8
        
        # 5. Palantir 신뢰도 보너스 (NEW!) - 최대 +5점
        # 신뢰도 0.6 미만: 0점, 0.6~1.0: 0~5점
        palantir_bonus = max(0, (palantir_reliability - 0.6) * 12.5)  # (1.0-0.6)*12.5 = 5
        palantir_bonus = min(5, palantir_bonus)
        
        # 총점 계산
        base_score = tech_score + vol_score + onchain_score + sentiment_score + macro_score + institutional_score
        total_score = int(base_score + palantir_bonus)
        total_score = min(100, max(0, total_score))
        
        # Type 결정 (Palantir 보너스 포함)
        if total_score >= 75:
            nice_type = 'A'
        elif total_score >= 55:
            nice_type = 'B'
        else:
            nice_type = 'C'
            
        return total_score, nice_type
    
    def analyze_coin(self, symbol: str, name: str = None, **kwargs) -> CoinAnalysis:
        """종합 코인 분석"""
        
        # 기본값 또는 전달된 값 사용 - 결정적 기본값
        h = _det_hash(symbol, 'coin')
        price = kwargs.get('price', 100 + h * 10)  # 결정적 기본가
        change_24h = kwargs.get('change_24h', (h % 25) - 10)  # -10 to +15
        volume_24h = kwargs.get('volume_24h', 1e6 * (1 + h))  # 결정적
        market_cap = kwargs.get('market_cap', 1e8 * (1 + h % 100))  # 결정적
        
        # 유통량 - 결정적 (max_supply 기반)
        total_supply = kwargs.get('total_supply', 1e8 * (1 + h % 100))
        max_supply = kwargs.get('max_supply', total_supply)  # None이면 무한 발행
        if max_supply is None:
            circulation_ratio = 100.0  # 무한 발행 코인
        else:
            circulating = kwargs.get('circulating', total_supply * 0.7)
            circulation_ratio = (circulating / max_supply * 100) if max_supply > 0 else 100.0
        circulating_supply = total_supply * (circulation_ratio / 100)
        
        # 거래량 비율
        volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
        
        # 분석 수행
        whale_position = self.analyze_whale_position(symbol, price)
        fractal = self.analyze_fractal(symbol, price)
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
            circulation_ratio=round(circulation_ratio * 100, 1),
            whale_position=whale_position,
            fractal=fractal,
            nice_score=nice_score,
            nice_type=nice_type,
            entry_price=f"{entry_low:,.2f} - {entry_high:,.2f}",
            stop_loss=f"{stop_loss:,.2f}",
            take_profit=f"{take_profit:,.2f}",
            kelly_pct=kelly,
            sector=self.get_sector(symbol)
        )
    
    def rank_coins(self, coins_data: List[Dict], timeframe: str = 'scalp') -> Dict[str, List[Dict]]:
        """
        코인 순위 정렬 (타임프레임별 차별화)
        
        Args:
            timeframe: 'scalp', 'short', 'medium', 'long'
            
        Logic:
            - scalp: 변동성(Vol) + 모멘텀(Change) 중심
            - short: 기술적 지표(Fractal) + NICE 점수
            - medium: OnChain(Whale) + NICE 점수
            - long: ETF/기관 점수 + 펀더멘탈
        """
        major_coins = []
        other_coins = []
        
        for coin in coins_data:
            symbol = coin.get('symbol', '').upper()
            
            # 타임프레임별 가중치 조정 (시뮬레이션)
            base_score = coin.get('price', 0) % 100 # 랜덤성 부여를 위한 임시 로직
            
            if timeframe == 'scalp':
                # 단타: 변동성과 거래량 중요
                score_boost = (abs(coin.get('change_24h', 0)) * 2) + (coin.get('volume_24h', 0) / 1e9)
            elif timeframe == 'short':
                # 단기: 추세와 모멘텀
                score_boost = coin.get('change_24h', 0) * 3
            elif timeframe == 'medium':
                # 중기: 온체인/펀더멘탈 (시가총액 가중)
                score_boost = (coin.get('market_cap', 0) / 1e10) + 10
            else: # long
                # 장기: 시가총액과 안정성
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
            coin_dict['score_boost'] = score_boost # 디버깅용
            
            if self.is_major(symbol):
                major_coins.append(coin_dict)
            else:
                other_coins.append(coin_dict)
        
        # 정렬 로직 차별화
        def sort_key(c):
            if timeframe == 'scalp':
                # 단타: 상승량 → 거래량 → NICE 점수
                return (-abs(c['change_24h']), -c['volume_24h'], -c['nice']['score'])
            elif timeframe == 'short':
                # NICE 점수 > 상승률
                return (-c['nice']['score'], -c['change_24h'])
            elif timeframe == 'medium':
                # NICE 점수 > 고래 포지션(가정)
                return (-c['nice']['score'], -c['market_cap'])
            else: # long
                # 장기: NICE 점수 > 시총 (장기 투자는 펀더멘탈 중요)
                return (-c['nice']['score'], -c['market_cap'])
        
        major_coins.sort(key=sort_key)
        other_coins.sort(key=sort_key)
        
        # BTC와 ETH는 항상 메이저 목록에 포함 (변동률이 낮아도)
        must_include = ['BTC', 'ETH']
        major_result = []
        seen_symbols = set()
        
        # 먼저 BTC/ETH 추가
        for coin in major_coins:
            if coin['symbol'] in must_include:
                major_result.append(coin)
                seen_symbols.add(coin['symbol'])
        
        # 나머지 메이저 코인 추가 (상한 10개)
        for coin in major_coins:
            if coin['symbol'] not in seen_symbols:
                major_result.append(coin)
                seen_symbols.add(coin['symbol'])
                if len(major_result) >= 10:
                    break
        
        return {
            'major': major_result,  # BTC/ETH 항상 포함 + 최대 10개
            'other': other_coins[:10],  # 상한 10개로 증가
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat()
        }


class CryptoFundFlow:
    """암호화폐 자금 흐름 분석"""
    
    def get_fund_flows(self) -> Dict:
        """자금 유입/유출 데이터"""
        
        # 시뮬레이션 데이터 (실제로는 API 연동)
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
            'market_sentiment': 'Bullish' if net_flow > 0 else 'Neutral',  # 결정적
            'sentiment_score': 65 if net_flow > 500 else (55 if net_flow > 0 else 45),  # 결정적
            'timestamp': datetime.now().isoformat()
        }


# 테스트
if __name__ == '__main__':
    analyzer = WhaleAnalyzer()
    
    # 샘플 코인 분석
    btc = analyzer.analyze_coin('BTC', 'Bitcoin', price=45000, change_24h=2.5, volume_24h=25e9, market_cap=880e9)
    print(f"BTC Analysis: {btc.nice_score} (Type {btc.nice_type})")
    print(f"Whale: {btc.whale_position.sentiment}, {btc.whale_position.total_holding}%")
    print(f"Fractal: {btc.fractal.pattern_type} ({btc.fractal.direction})")
    
    # 자금 흐름
    flows = CryptoFundFlow()
    fund_data = flows.get_fund_flows()
    print(f"\nNet Flow: ${fund_data['net_flow']}M")
    print(f"Sentiment: {fund_data['market_sentiment']} ({fund_data['sentiment_score']})")
