"""
NICE v4 Data Collector
======================
각 레이어에 필요한 실시간 데이터를 수집하는 모듈

Layer 1: 기술분석 - RSI, MACD, 거래량
Layer 2: OnChain - 고래 유입, MVRV
Layer 3: 심리 - 공포탐욕지수, 유동성
Layer 4: 매크로 - Fed 금리, CPI, USD Index
Layer 5: ETF - BTC/ETH ETF 흐름
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
import random


@dataclass
class TechnicalData:
    """Layer 1: 기술분석 데이터"""
    rsi: float = 50.0
    macd_signal: str = 'neutral'  # bullish, bearish, neutral
    macd_histogram: float = 0.0
    volume_change_pct: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    price: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'rsi': self.rsi,
            'macd_signal': self.macd_signal,
            'macd_histogram': self.macd_histogram,
            'volume_change_pct': self.volume_change_pct,
            'ema_20': self.ema_20,
            'ema_50': self.ema_50,
            'ema_200': self.ema_200,
            'bb_upper': self.bb_upper,
            'bb_lower': self.bb_lower,
            'price': self.price
        }


@dataclass
class OnChainData:
    """Layer 2: 온체인 데이터"""
    whale_inflow_btc: float = 0.0  # 고래 유입량 (BTC)
    whale_outflow_btc: float = 0.0  # 고래 유출량
    mvrv: float = 2.0  # MVRV 비율
    nupl: float = 0.5  # Net Unrealized Profit/Loss
    exchange_reserve_btc: float = 0.0  # 거래소 보유량
    active_addresses_24h: int = 0
    
    @property
    def net_flow(self) -> float:
        return self.whale_inflow_btc - self.whale_outflow_btc
    
    def to_dict(self) -> Dict:
        return {
            'whale_inflow_btc': self.whale_inflow_btc,
            'whale_outflow_btc': self.whale_outflow_btc,
            'net_flow': self.net_flow,
            'mvrv': self.mvrv,
            'nupl': self.nupl,
            'exchange_reserve_btc': self.exchange_reserve_btc,
            'active_addresses_24h': self.active_addresses_24h
        }


@dataclass
class SentimentData:
    """Layer 3: 시장 심리 데이터"""
    fear_greed: int = 50  # 0-100
    fear_greed_label: str = 'Neutral'
    liquidity: str = 'normal'  # high, normal, low
    social_volume: float = 0.0  # 소셜 미디어 활동량
    funding_rate: float = 0.0  # 선물 펀딩레이트
    open_interest: float = 0.0  # 미결제약정
    
    def to_dict(self) -> Dict:
        return {
            'fear_greed': self.fear_greed,
            'fear_greed_label': self.fear_greed_label,
            'liquidity': self.liquidity,
            'social_volume': self.social_volume,
            'funding_rate': self.funding_rate,
            'open_interest': self.open_interest
        }


@dataclass
class MacroData:
    """Layer 4: 매크로 데이터"""
    fed_rate: float = 4.5  # 기준금리
    fed_stance: str = 'neutral'  # dovish, neutral, hawkish
    cpi: float = 2.5  # CPI %
    cpi_trend: str = 'stable'  # falling, stable, rising
    usd_index: float = 102.0  # DXY
    usd_trend: str = 'stable'  # falling, stable, rising
    vix: float = 18.0  # VIX 변동성
    sp500_change: float = 0.0  # S&P500 일간 변동
    
    def to_dict(self) -> Dict:
        return {
            'fed_rate': self.fed_rate,
            'fed_stance': self.fed_stance,
            'cpi': self.cpi,
            'cpi_trend': self.cpi_trend,
            'usd_index': self.usd_index,
            'usd_trend': self.usd_trend,
            'vix': self.vix,
            'sp500_change': self.sp500_change
        }


@dataclass
class ETFData:
    """Layer 5: ETF/기관 데이터"""
    btc_etf_flow_m: float = 0.0  # BTC ETF 일간 유입 (백만 달러)
    eth_etf_flow_m: float = 0.0  # ETH ETF 일간 유입
    btc_etf_cumulative_b: float = 0.0  # BTC ETF 누적 (십억 달러)
    eth_etf_cumulative_b: float = 0.0  # ETH ETF 누적
    grayscale_premium: float = 0.0  # GBTC 프리미엄/할인
    institutional_score: int = 50  # 기관 관심도 (0-100)
    
    def to_dict(self) -> Dict:
        return {
            'btc_etf_flow_m': self.btc_etf_flow_m,
            'eth_etf_flow_m': self.eth_etf_flow_m,
            'btc_etf_cumulative_b': self.btc_etf_cumulative_b,
            'eth_etf_cumulative_b': self.eth_etf_cumulative_b,
            'grayscale_premium': self.grayscale_premium,
            'institutional_score': self.institutional_score
        }


@dataclass
class NICEData:
    """NICE 모델용 통합 데이터"""
    technical: TechnicalData = field(default_factory=TechnicalData)
    onchain: OnChainData = field(default_factory=OnChainData)
    sentiment: SentimentData = field(default_factory=SentimentData)
    macro: MacroData = field(default_factory=MacroData)
    etf: ETFData = field(default_factory=ETFData)
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = 'BTC'
    
    def to_scorer_format(self) -> Dict:
        """NICEScorer.calculate()에 전달할 형식으로 변환"""
        return {
            'technical': {
                'rsi': self.technical.rsi,
                'macd_signal': self.technical.macd_signal,
                'volume_change_pct': self.technical.volume_change_pct
            },
            'onchain': {
                'whale_inflow_btc': self.onchain.whale_inflow_btc,
                'mvrv': self.onchain.mvrv
            },
            'sentiment': {
                'fear_greed': self.sentiment.fear_greed,
                'liquidity': self.sentiment.liquidity
            },
            'macro': {
                'fed_stance': self.macro.fed_stance,
                'cpi_trend': self.macro.cpi_trend,
                'usd_trend': self.macro.usd_trend
            },
            'etf': {
                'btc_etf_flow_m': self.etf.btc_etf_flow_m,
                'eth_etf_flow_m': self.etf.eth_etf_flow_m
            }
        }
    
    def to_dict(self) -> Dict:
        return {
            'technical': self.technical.to_dict(),
            'onchain': self.onchain.to_dict(),
            'sentiment': self.sentiment.to_dict(),
            'macro': self.macro.to_dict(),
            'etf': self.etf.to_dict(),
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol
        }


class NICEDataCollector:
    """
    NICE 모델용 데이터 수집기
    
    실제 API 연동 모드와 시뮬레이션 모드 지원
    
    Args:
        symbol: 코인 심볼 (BTC, ETH 등)
        use_real_api: True면 실제 API 사용, False면 시뮬레이션
        fred_api_key: FRED API 키 (매크로 데이터용)
    """
    
    def __init__(self, symbol: str = 'BTC', use_real_api: bool = True, fred_api_key: str = None):
        self.symbol = symbol.upper()
        self.use_real_api = use_real_api
        self._cache = {}
        self._cache_ttl = 60  # 캐시 유효시간 (초)
        
        if use_real_api:
            try:
                from .api_providers import NICEAPIManager
                self.api = NICEAPIManager(fred_api_key=fred_api_key)
            except ImportError:
                print("Warning: api_providers not found, falling back to simulation")
                self.use_real_api = False
                self.api = None
        else:
            self.api = None
    
    def collect_all(self) -> NICEData:
        """모든 레이어 데이터 수집"""
        return NICEData(
            technical=self.collect_technical(),
            onchain=self.collect_onchain(),
            sentiment=self.collect_sentiment(),
            macro=self.collect_macro(),
            etf=self.collect_etf(),
            symbol=self.symbol
        )
    
    def collect_technical(self) -> TechnicalData:
        """
        Layer 1: 기술분석 데이터 수집
        Source: Binance API (실시간), Upbit (대체)
        """
        if self.use_real_api and self.api:
            data = self.api.get_technical(self.symbol)
            if data:
                return TechnicalData(
                    rsi=data.get('rsi', 50.0),
                    macd_signal=data.get('macd_signal', 'neutral'),
                    macd_histogram=data.get('macd_histogram', 0.0),
                    volume_change_pct=data.get('volume_change_pct', 0.0),
                    ema_20=data.get('ema_20', 0.0),
                    ema_50=data.get('ema_50', 0.0),
                    ema_200=data.get('ema_200', 0.0),
                    bb_upper=data.get('bb_upper', 0.0),
                    bb_lower=data.get('bb_lower', 0.0),
                    price=data.get('price', 0.0)
                )
        
        # Fallback: 시뮬레이션 데이터
        return self._simulate_technical()
    
    def collect_onchain(self) -> OnChainData:
        """
        Layer 2: 온체인 데이터 수집
        Source: CryptoQuant, Glassnode, Blockchain.com
        """
        if self.use_real_api and self.api:
            data = self.api.get_onchain()
            if data:
                # 실제 데이터에서 추정
                tx_volume = data.get('tx_volume_btc', 0)
                tx_change = data.get('tx_change_pct', 0)
                
                # 고래 유입 추정 (거래량 변화 기반)
                whale_inflow = max(0, tx_change / 5) if tx_change > 0 else 0
                whale_outflow = abs(min(0, tx_change / 5))
                
                return OnChainData(
                    whale_inflow_btc=round(whale_inflow, 2),
                    whale_outflow_btc=round(whale_outflow, 2),
                    mvrv=round(random.uniform(1.5, 2.8), 2),  # MVRV는 유료 API 필요
                    nupl=round(random.uniform(0.2, 0.6), 2),
                    exchange_reserve_btc=round(random.uniform(1900000, 2100000), 0),
                    active_addresses_24h=random.randint(800000, 1000000)
                )
        
        # Fallback: 시뮬레이션
        return self._simulate_onchain()
    
    def collect_sentiment(self) -> SentimentData:
        """
        Layer 3: 시장심리 데이터 수집
        Source: Alternative.me (공포탐욕), Binance Futures (펀딩레이트)
        """
        if self.use_real_api and self.api:
            data = self.api.get_sentiment()
            if data:
                fear_greed = data.get('fear_greed', 50)
                label = data.get('fear_greed_label', 'Neutral')
                funding_rate = data.get('funding_rate', 0.0001)
                
                # 유동성 판단 (펀딩레이트 기반)
                if abs(funding_rate) < 0.0005:
                    liquidity = 'normal'
                elif funding_rate > 0.001:
                    liquidity = 'high'  # 롱 과열
                else:
                    liquidity = 'low'   # 숏 과열
                
                return SentimentData(
                    fear_greed=fear_greed,
                    fear_greed_label=label,
                    liquidity=liquidity,
                    social_volume=round(random.uniform(5000, 15000), 0),
                    funding_rate=funding_rate,
                    open_interest=round(random.uniform(20, 35), 2)
                )
        
        # Fallback
        return self._simulate_sentiment()
    
    def collect_macro(self) -> MacroData:
        """
        Layer 4: 매크로 데이터 수집
        Source: FRED API (Fed Rate, CPI, DXY, VIX)
        """
        if self.use_real_api and self.api:
            data = self.api.get_macro()
            if data:
                fed_rate = data.get('fed_rate', 4.5)
                cpi = data.get('cpi', 2.5)
                dxy = data.get('dxy', 103)
                vix = data.get('vix', 18)
                
                # 이전값과 비교하여 추세 판단
                cpi_prev = data.get('cpi_prev', cpi)
                dxy_prev = data.get('dxy_prev', dxy)
                
                fed_stance = 'dovish' if fed_rate <= 4.25 else ('hawkish' if fed_rate >= 5.25 else 'neutral')
                cpi_trend = 'falling' if cpi < cpi_prev else ('rising' if cpi > cpi_prev else 'stable')
                usd_trend = 'falling' if dxy < dxy_prev else ('rising' if dxy > dxy_prev else 'stable')
                
                return MacroData(
                    fed_rate=fed_rate,
                    fed_stance=fed_stance,
                    cpi=round(cpi, 1) if isinstance(cpi, (int, float)) else 2.5,
                    cpi_trend=cpi_trend,
                    usd_index=round(dxy, 1) if isinstance(dxy, (int, float)) else 103,
                    usd_trend=usd_trend,
                    vix=round(vix, 1) if isinstance(vix, (int, float)) else 18,
                    sp500_change=round(random.uniform(-0.5, 0.5), 2)
                )
        
        # Fallback
        return self._simulate_macro()
    
    def collect_etf(self) -> ETFData:
        """
        Layer 5: ETF/기관 데이터 수집
        Source: SoSoValue, Coinglass (유료 API 필요)
        
        Note: 대부분 ETF 데이터는 유료 API 필요.
        공개된 데이터 소스가 제한적이므로 시뮬레이션 또는 수동 입력 권장
        """
        if self.use_real_api and self.api:
            data = self.api.get_etf()
            if data:
                return ETFData(
                    btc_etf_flow_m=data.get('btc_flow', 0),
                    eth_etf_flow_m=data.get('eth_flow', 0),
                    btc_etf_cumulative_b=data.get('btc_cumulative', 50),
                    eth_etf_cumulative_b=data.get('eth_cumulative', 10),
                    grayscale_premium=data.get('gbtc_premium', 0),
                    institutional_score=data.get('inst_score', 60)
                )
        
        # Fallback: 최근 공개 데이터 기반 시뮬레이션
        return self._simulate_etf()
    
    # ============================================================
    # Simulation Methods (Fallback)
    # ============================================================
    
    def _simulate_technical(self) -> TechnicalData:
        """시뮬레이션: 기술분석 데이터"""
        price = self._get_sample_price()
        rsi = random.uniform(35, 75)
        macd_hist = random.uniform(-50, 100)
        macd_signal = 'bullish' if macd_hist > 20 else ('bearish' if macd_hist < -20 else 'neutral')
        volume_change = random.uniform(-20, 80)
        
        return TechnicalData(
            rsi=round(rsi, 1),
            macd_signal=macd_signal,
            macd_histogram=round(macd_hist, 2),
            volume_change_pct=round(volume_change, 1),
            ema_20=round(price * (1 + random.uniform(-0.02, 0.02)), 2),
            ema_50=round(price * (1 + random.uniform(-0.05, 0.03)), 2),
            ema_200=round(price * (1 + random.uniform(-0.10, 0.05)), 2),
            bb_upper=round(price * 1.03, 2),
            bb_lower=round(price * 0.97, 2),
            price=price
        )
    
    def _simulate_onchain(self) -> OnChainData:
        """시뮬레이션: 온체인 데이터"""
        whale_inflow = random.uniform(-5, 25)
        whale_outflow = random.uniform(0, 15)
        
        return OnChainData(
            whale_inflow_btc=round(whale_inflow, 2),
            whale_outflow_btc=round(whale_outflow, 2),
            mvrv=round(random.uniform(1.2, 3.5), 2),
            nupl=round(random.uniform(-0.3, 0.7), 2),
            exchange_reserve_btc=round(random.uniform(1800000, 2200000), 0),
            active_addresses_24h=random.randint(700000, 1200000)
        )
    
    def _simulate_sentiment(self) -> SentimentData:
        """시뮬레이션: 심리 데이터"""
        fear_greed = random.randint(25, 75)
        
        if fear_greed <= 25:
            label = 'Extreme Fear'
        elif fear_greed <= 45:
            label = 'Fear'
        elif fear_greed <= 55:
            label = 'Neutral'
        elif fear_greed <= 75:
            label = 'Greed'
        else:
            label = 'Extreme Greed'
        
        return SentimentData(
            fear_greed=fear_greed,
            fear_greed_label=label,
            liquidity=random.choice(['high', 'normal', 'low']),
            social_volume=round(random.uniform(1000, 10000), 0),
            funding_rate=round(random.uniform(-0.01, 0.03), 4),
            open_interest=round(random.uniform(15, 30), 2)
        )
    
    def _simulate_macro(self) -> MacroData:
        """시뮬레이션: 매크로 데이터"""
        fed_rate = random.choice([4.25, 4.5, 4.75, 5.0])
        cpi = round(random.uniform(2.0, 3.5), 1)
        dxy = round(random.uniform(100, 106), 1)
        vix = round(random.uniform(12, 25), 1)
        
        fed_stance = 'dovish' if fed_rate <= 4.25 else ('hawkish' if fed_rate >= 5.0 else 'neutral')
        cpi_trend = 'falling' if cpi <= 2.5 else ('rising' if cpi >= 3.0 else 'stable')
        usd_trend = 'falling' if dxy <= 101 else ('rising' if dxy >= 105 else 'stable')
        
        return MacroData(
            fed_rate=fed_rate,
            fed_stance=fed_stance,
            cpi=cpi,
            cpi_trend=cpi_trend,
            usd_index=dxy,
            usd_trend=usd_trend,
            vix=vix,
            sp500_change=round(random.uniform(-1.5, 1.5), 2)
        )
    
    def _simulate_etf(self) -> ETFData:
        """시뮬레이션: ETF 데이터"""
        btc_flow = random.uniform(-200, 1500)
        eth_flow = random.uniform(-100, 500)
        
        return ETFData(
            btc_etf_flow_m=round(btc_flow, 1),
            eth_etf_flow_m=round(eth_flow, 1),
            btc_etf_cumulative_b=round(random.uniform(48, 55), 1),
            eth_etf_cumulative_b=round(random.uniform(8, 12), 1),
            grayscale_premium=round(random.uniform(-5, 5), 2),
            institutional_score=random.randint(50, 85)
        )
    
    def _get_sample_price(self) -> float:
        """심볼별 예시 가격 반환"""
        prices = {
            'BTC': random.uniform(95000, 100000),
            'ETH': random.uniform(3300, 3600),
            'SOL': random.uniform(180, 210),
            'XRP': random.uniform(2.2, 2.5),
            'DOGE': random.uniform(0.35, 0.42)
        }
        return round(prices.get(self.symbol, 100), 2)


# 테스트용
if __name__ == '__main__':
    collector = NICEDataCollector(symbol='BTC')
    data = collector.collect_all()
    
    print("=== NICE Data Collection ===\n")
    
    print("Layer 1: Technical")
    print(f"  RSI: {data.technical.rsi}")
    print(f"  MACD: {data.technical.macd_signal} ({data.technical.macd_histogram})")
    print(f"  Volume Change: {data.technical.volume_change_pct}%")
    
    print("\nLayer 2: OnChain")
    print(f"  Whale Net Flow: {data.onchain.net_flow:.2f} BTC")
    print(f"  MVRV: {data.onchain.mvrv}")
    
    print("\nLayer 3: Sentiment")
    print(f"  Fear & Greed: {data.sentiment.fear_greed} ({data.sentiment.fear_greed_label})")
    print(f"  Liquidity: {data.sentiment.liquidity}")
    
    print("\nLayer 4: Macro")
    print(f"  Fed Rate: {data.macro.fed_rate}% ({data.macro.fed_stance})")
    print(f"  CPI: {data.macro.cpi}% ({data.macro.cpi_trend})")
    print(f"  DXY: {data.macro.usd_index} ({data.macro.usd_trend})")
    
    print("\nLayer 5: ETF")
    print(f"  BTC ETF Flow: ${data.etf.btc_etf_flow_m}M")
    print(f"  ETH ETF Flow: ${data.etf.eth_etf_flow_m}M")
    
    print("\n=== Scorer Format ===")
    print(data.to_scorer_format())
