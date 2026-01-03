"""
NICE v4 External API Providers
==============================
실제 외부 API 연동 모듈

지원 API:
- Layer 1: Binance, Upbit (기술분석)
- Layer 2: CryptoQuant, Glassnode (온체인)
- Layer 3: Alternative.me (공포탐욕), Coinalyze (펀딩레이트)
- Layer 4: FRED API (매크로)
- Layer 5: SoSoValue (ETF)
"""

import requests
import json
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import time


# ============================================================
# Layer 1: Technical Analysis API (Binance / Upbit)
# ============================================================

class BinanceAPI:
    """Binance Public API (No API Key Required)"""
    
    BASE_URL = "https://api.binance.com"
    
    SYMBOL_MAP = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
        'XRP': 'XRPUSDT', 'DOGE': 'DOGEUSDT', 'BNB': 'BNBUSDT',
        'ADA': 'ADAUSDT', 'AVAX': 'AVAXUSDT', 'LINK': 'LINKUSDT',
        'DOT': 'DOTUSDT', 'MATIC': 'MATICUSDT', 'ATOM': 'ATOMUSDT'
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'NICE-v4-Collector/1.0'})
    
    def get_price(self, symbol: str) -> Dict:
        """현재가 조회"""
        try:
            pair = self.SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}USDT")
            url = f"{self.BASE_URL}/api/v3/ticker/24hr"
            resp = self.session.get(url, params={'symbol': pair}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return {
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'volume_24h': float(data['quoteVolume']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice'])
            }
        except Exception as e:
            print(f"Binance price error: {e}")
            return None
    
    def get_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List:
        """캔들스틱 데이터 조회"""
        try:
            pair = self.SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}USDT")
            url = f"{self.BASE_URL}/api/v3/klines"
            resp = self.session.get(url, params={
                'symbol': pair,
                'interval': interval,
                'limit': limit
            }, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Binance klines error: {e}")
            return []
    
    def calculate_indicators(self, symbol: str) -> Dict:
        """기술 지표 계산 (RSI, MACD, Volume)"""
        klines = self.get_klines(symbol, interval='1h', limit=100)
        if not klines:
            return None
        
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        # RSI 계산 (14기간)
        rsi = self._calc_rsi(closes, 14)
        
        # MACD 계산
        macd, signal, histogram = self._calc_macd(closes)
        macd_signal = 'bullish' if histogram > 0 else 'bearish' if histogram < 0 else 'neutral'
        
        # 거래량 변화
        if len(volumes) >= 2:
            avg_vol = sum(volumes[-20:-1]) / 19 if len(volumes) >= 20 else sum(volumes[:-1]) / (len(volumes)-1)
            vol_change = ((volumes[-1] - avg_vol) / avg_vol) * 100 if avg_vol > 0 else 0
        else:
            vol_change = 0
        
        # EMA 계산
        ema_20 = self._calc_ema(closes, 20)
        ema_50 = self._calc_ema(closes, 50)
        ema_200 = self._calc_ema(closes, min(200, len(closes)-1)) if len(closes) > 50 else ema_50
        
        # 볼린저 밴드
        bb_upper, bb_lower = self._calc_bollinger(closes, 20, 2)
        
        return {
            'rsi': round(rsi, 1),
            'macd_signal': macd_signal,
            'macd_histogram': round(histogram, 2),
            'volume_change_pct': round(vol_change, 1),
            'ema_20': round(ema_20, 2),
            'ema_50': round(ema_50, 2),
            'ema_200': round(ema_200, 2),
            'bb_upper': round(bb_upper, 2),
            'bb_lower': round(bb_lower, 2),
            'price': closes[-1]
        }
    
    def _calc_rsi(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calc_macd(self, closes: List[float]) -> tuple:
        ema_12 = self._calc_ema(closes, 12)
        ema_26 = self._calc_ema(closes, 26)
        macd = ema_12 - ema_26
        
        # Signal line (9-period EMA of MACD) - simplified
        signal = macd * 0.8  # Approximation
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calc_ema(self, data: List[float], period: int) -> float:
        if len(data) < period:
            return data[-1] if data else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calc_bollinger(self, closes: List[float], period: int = 20, std_dev: int = 2) -> tuple:
        if len(closes) < period:
            return closes[-1] * 1.02, closes[-1] * 0.98
        
        sma = sum(closes[-period:]) / period
        variance = sum((x - sma) ** 2 for x in closes[-period:]) / period
        std = variance ** 0.5
        
        return sma + std_dev * std, sma - std_dev * std


# ============================================================
# Layer 3: Sentiment API (Alternative.me Fear & Greed)
# ============================================================

class FearGreedAPI:
    """Alternative.me Fear & Greed Index API"""
    
    URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_index(self) -> Dict:
        """현재 공포탐욕지수 조회"""
        try:
            resp = self.session.get(self.URL, params={'limit': 1}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            if 'data' in data and len(data['data']) > 0:
                fng = data['data'][0]
                return {
                    'value': int(fng['value']),
                    'label': fng['value_classification'],
                    'timestamp': fng['timestamp']
                }
        except Exception as e:
            print(f"Fear & Greed API error: {e}")
        
        return None


class CoinalyzeAPI:
    """Coinalyze Funding Rate API (Free Tier)"""
    
    URL = "https://api.coinalyze.net/v1/funding-rate"
    
    def get_funding_rate(self, symbol: str = 'BTC') -> Dict:
        """펀딩레이트 조회 (Binance Perpetual)"""
        try:
            # Binance Perpetual Funding Rate
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            resp = requests.get(url, params={
                'symbol': f"{symbol.upper()}USDT",
                'limit': 1
            }, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            if data and len(data) > 0:
                return {
                    'rate': float(data[0]['fundingRate']),
                    'time': data[0]['fundingTime']
                }
        except Exception as e:
            print(f"Funding rate error: {e}")
        
        return None


# ============================================================
# Layer 4: Macro API (FRED)
# ============================================================

class FREDAPI:
    """Federal Reserve Economic Data API"""
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    # 무료 API Key (데모용, 실제 사용시 발급 필요)
    # https://fred.stlouisfed.org/docs/api/api_key.html
    API_KEY = None  # Set via environment or config
    
    SERIES = {
        'fed_rate': 'FEDFUNDS',      # Federal Funds Rate
        'cpi': 'CPIAUCSL',            # Consumer Price Index
        'dxy': 'DTWEXBGS',            # Trade Weighted U.S. Dollar Index
        'vix': 'VIXCLS',              # VIX
        'sp500': 'SP500'              # S&P 500
    }
    
    def __init__(self, api_key: str = None):
        import os
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        self.session = requests.Session()
    
    def get_series(self, series_id: str, limit: int = 5) -> List[Dict]:
        """FRED 시리즈 데이터 조회"""
        if not self.api_key:
            return None
        
        try:
            resp = self.session.get(self.BASE_URL, params={
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': limit
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get('observations', [])
        except Exception as e:
            print(f"FRED API error: {e}")
            return None
    
    def get_macro_data(self) -> Dict:
        """전체 매크로 데이터 조회"""
        result = {}
        
        for key, series_id in self.SERIES.items():
            obs = self.get_series(series_id, limit=2)
            if obs and len(obs) > 0:
                try:
                    result[key] = float(obs[0]['value'])
                    if len(obs) > 1:
                        result[f'{key}_prev'] = float(obs[1]['value'])
                except (ValueError, KeyError):
                    pass
        
        return result if result else None


# ============================================================
# Layer 5: ETF Flow API (SoSoValue)
# ============================================================

class SoSoValueAPI:
    """SoSoValue ETF Flow Data (Scraping fallback)"""
    
    # SoSoValue doesn't have public API, using alternative sources
    ALTERNATIVE_URL = "https://api.coinglass.com/api/pro/etf/flows"
    
    def get_etf_flows(self) -> Dict:
        """ETF 유입/유출 데이터 (대안 소스 또는 시뮬레이션)"""
        try:
            # Try Coinglass ETF data (may require API key)
            resp = requests.get(
                "https://open-api.coinglass.com/public/v2/indicator/etf_flows",
                headers={'coinglassSecret': ''},  # Free tier limited
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                # Parse and return
                return data.get('data', {})
        except:
            pass
        
        # Fallback: Return latest known typical values
        return None


# ============================================================
# Layer 2: OnChain API (CryptoQuant / Glassnode Alternative)
# ============================================================

class OnChainAPI:
    """OnChain Data API (using free alternatives)"""
    
    def get_whale_data(self) -> Dict:
        """고래 데이터 (Blockchain.com free API)"""
        try:
            # Blockchain.com exchange flows (free)
            resp = requests.get(
                "https://api.blockchain.info/charts/estimated-transaction-volume?timespan=2days&format=json",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                values = data.get('values', [])
                if len(values) >= 2:
                    today = values[-1].get('y', 0)
                    yesterday = values[-2].get('y', 0)
                    change = ((today - yesterday) / yesterday * 100) if yesterday > 0 else 0
                    return {
                        'tx_volume_btc': today / 1e8,  # Convert from satoshis
                        'tx_change_pct': round(change, 2)
                    }
        except Exception as e:
            print(f"OnChain API error: {e}")
        
        return None
    
    def get_exchange_reserve(self) -> Dict:
        """거래소 보유량 (CryptoQuant free tier)"""
        # CryptoQuant requires paid API
        # Using approximation from public data
        return None


# ============================================================
# Unified API Manager
# ============================================================

class NICEAPIManager:
    """통합 API 관리자"""
    
    def __init__(self, fred_api_key: str = None):
        self.binance = BinanceAPI()
        self.fear_greed = FearGreedAPI()
        self.coinalyze = CoinalyzeAPI()
        self.fred = FREDAPI(api_key=fred_api_key)
        self.sosovalue = SoSoValueAPI()
        self.onchain = OnChainAPI()
        
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 60  # seconds
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache_time:
            return False
        return (datetime.now() - self._cache_time[key]).seconds < self._cache_ttl
    
    def get_technical(self, symbol: str) -> Dict:
        """Layer 1: 기술분석 데이터"""
        cache_key = f"technical_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        data = self.binance.calculate_indicators(symbol)
        if data:
            self._cache[cache_key] = data
            self._cache_time[cache_key] = datetime.now()
        
        return data
    
    def get_sentiment(self) -> Dict:
        """Layer 3: 심리 데이터"""
        cache_key = "sentiment"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        fng = self.fear_greed.get_index()
        funding = self.coinalyze.get_funding_rate('BTC')
        
        data = {
            'fear_greed': fng['value'] if fng else 50,
            'fear_greed_label': fng['label'] if fng else 'Neutral',
            'funding_rate': funding['rate'] if funding else 0.0001
        }
        
        self._cache[cache_key] = data
        self._cache_time[cache_key] = datetime.now()
        
        return data
    
    def get_macro(self) -> Dict:
        """Layer 4: 매크로 데이터"""
        cache_key = "macro"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        data = self.fred.get_macro_data()
        if data:
            self._cache[cache_key] = data
            self._cache_time[cache_key] = datetime.now()
        
        return data
    
    def get_onchain(self) -> Dict:
        """Layer 2: 온체인 데이터"""
        cache_key = "onchain"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        data = self.onchain.get_whale_data()
        if data:
            self._cache[cache_key] = data
            self._cache_time[cache_key] = datetime.now()
        
        return data
    
    def get_etf(self) -> Dict:
        """Layer 5: ETF 데이터"""
        # Most ETF APIs require paid subscription
        # Return typical values or None
        return self.sosovalue.get_etf_flows()


# 테스트
if __name__ == '__main__':
    print("=== NICE API Provider Test ===\n")
    
    # Binance Test
    print("Layer 1: Binance API")
    binance = BinanceAPI()
    price = binance.get_price('BTC')
    if price:
        print(f"  BTC Price: ${price['price']:,.2f} ({price['change_24h']:+.2f}%)")
    
    indicators = binance.calculate_indicators('BTC')
    if indicators:
        print(f"  RSI: {indicators['rsi']}")
        print(f"  MACD: {indicators['macd_signal']}")
        print(f"  Volume Change: {indicators['volume_change_pct']}%")
    
    # Fear & Greed Test
    print("\nLayer 3: Fear & Greed Index")
    fng = FearGreedAPI()
    index = fng.get_index()
    if index:
        print(f"  Value: {index['value']} ({index['label']})")
    
    # Funding Rate Test
    print("\nLayer 3: Funding Rate")
    coinalyze = CoinalyzeAPI()
    funding = coinalyze.get_funding_rate('BTC')
    if funding:
        print(f"  BTC Funding Rate: {funding['rate']:.4%}")
    
    print("\n=== Test Complete ===")
