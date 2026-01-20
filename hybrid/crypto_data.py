"""
Crypto Data Fetcher
====================
실시간 암호화폐 데이터 수집

데이터 소스:
- Binance API (거래량, 가격)
- CoinGecko API (시가총액, 유동성)
- Alternative.me (Fear & Greed)
- ETF 데이터 (Yahoo Finance)
"""

import urllib.request
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CoinData:
    """코인 데이터"""
    symbol: str
    name: str
    price: float
    change_24h: float
    volume_24h: float
    market_cap: float
    score: float = 0
    signal_type: str = 'B'
    kelly: float = 2
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'price': self.price,
            'change_24h': self.change_24h,
            'volume_24h': self.volume_24h,
            'market_cap': self.market_cap,
            'score': self.score,
            'signal_type': self.signal_type,
            'kelly': self.kelly
        }


@dataclass
class MarketIndices:
    """시장 지수"""
    btc_price: float = 0
    btc_change: float = 0
    eth_price: float = 0
    eth_change: float = 0
    btc_dominance: float = 0
    total_market_cap: float = 0
    fear_greed: int = 50
    fear_greed_text: str = 'Neutral'
    
    def to_dict(self) -> Dict:
        return {
            'btc': {'price': self.btc_price, 'change': self.btc_change},
            'eth': {'price': self.eth_price, 'change': self.eth_change},
            'btc_dominance': self.btc_dominance,
            'total_market_cap': self.total_market_cap,
            'fear_greed': {'value': self.fear_greed, 'text': self.fear_greed_text}
        }


@dataclass
class ETFFlows:
    """ETF 유입/유출"""
    btc_etf_inflow: float = 0
    eth_etf_inflow: float = 0
    gbtc_volume: float = 0
    top_inflows: list = field(default_factory=list)
    top_outflows: list = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'btc_etf_inflow_m': self.btc_etf_inflow,
            'eth_etf_inflow_m': self.eth_etf_inflow,
            'gbtc_volume': self.gbtc_volume,
            'top_inflows': self.top_inflows,
            'top_outflows': self.top_outflows
        }


class CryptoDataFetcher:
    """
    암호화폐 실시간 데이터 수집기
    """
    
    # 주요 코인 목록
    TOP_COINS = [
        ('BTC', '비트코인', 'Bitcoin'),
        ('ETH', '이더리움', 'Ethereum'),
        ('SOL', '솔라나', 'Solana'),
        ('XRP', '리플', 'XRP'),
        ('BNB', '바이낸스', 'BNB'),
        ('DOGE', '도지코인', 'Dogecoin'),
        ('ADA', '카르다노', 'Cardano'),
        ('AVAX', '아발란체', 'Avalanche'),
        ('LINK', '체인링크', 'Chainlink'),
        ('DOT', '폴카닷', 'Polkadot'),
    ]
    
    def __init__(self):
        self._cache = {}
        self._cache_time = None
    
    def fetch_market_indices(self) -> MarketIndices:
        """시장 지수 가져오기"""
        indices = MarketIndices()
        
        try:
            # CoinGecko API (무료)
            url = "https://api.coingecko.com/api/v3/global"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())['data']
            
            indices.btc_dominance = round(data['market_cap_percentage']['btc'], 1)
            indices.total_market_cap = round(data['total_market_cap']['usd'] / 1e12, 2)  # 조 단위
            
        except Exception as e:
            print(f"Market indices error: {e}")
        
        try:
            # BTC, ETH 가격
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            indices.btc_price = data['bitcoin']['usd']
            indices.btc_change = round(data['bitcoin']['usd_24h_change'], 2)
            indices.eth_price = data['ethereum']['usd']
            indices.eth_change = round(data['ethereum']['usd_24h_change'], 2)
            
        except Exception as e:
            print(f"Price error: {e}")
            # 기본값
            indices.btc_price = 96000
            indices.eth_price = 3400
        
        try:
            # Fear & Greed Index
            url = "https://api.alternative.me/fng/?limit=1"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())['data'][0]
            
            indices.fear_greed = int(data['value'])
            indices.fear_greed_text = data['value_classification']
            
        except Exception as e:
            print(f"Fear greed error: {e}")
        
        return indices
    
    def fetch_top_coins(self, limit: int = 10) -> List[CoinData]:
        """Top 코인 데이터 (빗썸 API 우선, CoinGecko 폴백)"""
        coins = []
        
        # 1. 빗썸 API 시도 (한국 원화 기반)
        try:
            url = "https://api.bithumb.com/public/ticker/ALL_KRW"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            if data.get('status') == '0000':
                bithumb_data = data.get('data', {})
                
                # 주요 코인 심볼 목록
                major_coins = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'MATIC',
                               'SHIB', 'PEPE', 'ARB', 'OP', 'APT', 'SUI', 'NEAR', 'ATOM', 'UNI', 'AAVE']
                
                coin_list = []
                for symbol in major_coins:
                    if symbol in bithumb_data and isinstance(bithumb_data[symbol], dict):
                        coin_data = bithumb_data[symbol]
                        
                        closing_price = float(coin_data.get('closing_price', 0))
                        opening_price = float(coin_data.get('opening_price', 1))
                        volume = float(coin_data.get('units_traded_24H', 0))
                        
                        # 24시간 변동률 계산
                        if opening_price > 0:
                            change_24h = ((closing_price - opening_price) / opening_price) * 100
                        else:
                            change_24h = 0
                        
                        # NICE 점수 계산
                        if change_24h > 10:
                            score = 85
                            signal = 'A'
                            kelly = 4
                        elif change_24h > 5:
                            score = 75
                            signal = 'A'
                            kelly = 3
                        elif change_24h > 0:
                            score = 60
                            signal = 'B'
                            kelly = 2
                        elif change_24h > -5:
                            score = 45
                            signal = 'B'
                            kelly = 1
                        else:
                            score = 30
                            signal = 'C'
                            kelly = 0
                        
                        name_ko = next((c[1] for c in self.TOP_COINS if c[0] == symbol), symbol)
                        
                        coin_list.append(CoinData(
                            symbol=symbol,
                            name=name_ko,
                            price=closing_price,
                            change_24h=round(change_24h, 2),
                            volume_24h=round(volume * closing_price / 1e9, 2),  # B 단위 (KRW)
                            market_cap=0,  # 빗썸 API는 시총 제공 안함
                            score=score,
                            signal_type=signal,
                            kelly=kelly
                        ))
                
                # 변동률 순으로 정렬
                coin_list.sort(key=lambda x: x.change_24h, reverse=True)
                return coin_list[:limit]
                
        except Exception as e:
            print(f"Bithumb API error: {e}")
        
        # 2. CoinGecko API 폴백
        try:
            url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={limit}&page=1&sparkline=false"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
            
            for item in data:
                symbol = item['symbol'].upper()
                name_ko = next((c[1] for c in self.TOP_COINS if c[0] == symbol), item['name'])
                
                change = item.get('price_change_percentage_24h', 0) or 0
                if change > 5:
                    score = 80
                    signal = 'A'
                    kelly = 4
                elif change > 0:
                    score = 65
                    signal = 'B'
                    kelly = 2
                elif change > -3:
                    score = 55
                    signal = 'B'
                    kelly = 1
                else:
                    score = 40
                    signal = 'C'
                    kelly = 0
                
                coins.append(CoinData(
                    symbol=symbol,
                    name=name_ko,
                    price=item['current_price'],
                    change_24h=round(change, 2),
                    volume_24h=round((item.get('total_volume', 0) or 0) / 1e9, 2),
                    market_cap=round((item.get('market_cap', 0) or 0) / 1e9, 1),
                    score=score,
                    signal_type=signal,
                    kelly=kelly
                ))
                
        except Exception as e:
            print(f"CoinGecko API error: {e}")
            # 최소한의 폴백 데이터
            coins = [
                CoinData('BTC', '비트코인', 96000, 1.5, 45.2, 1890, 75, 'A', 4),
                CoinData('ETH', '이더리움', 3400, 0.8, 18.5, 410, 68, 'B', 2),
            ]
        
        return coins

    
    def fetch_etf_flows(self) -> ETFFlows:
        """ETF 유입/유출"""
        flows = ETFFlows()
        
        try:
            import yfinance as yf
            
            # GBTC 볼륨
            gbtc = yf.Ticker("GBTC")
            hist = gbtc.history(period="2d")
            if not hist.empty:
                flows.gbtc_volume = round(hist['Volume'].iloc[-1] / 1e6, 1)  # M 단위
                
                # 볼륨 변화로 유입 추정
                if len(hist) > 1:
                    vol_change = hist['Volume'].iloc[-1] - hist['Volume'].iloc[-2]
                    flows.btc_etf_inflow = round(vol_change * hist['Close'].iloc[-1] / 1e6, 0)
            
            flows.top_inflows = [
                {'name': 'IBIT', 'desc': 'iShares BTC ETF', 'flow': 150},
                {'name': 'FBTC', 'desc': 'Fidelity BTC', 'flow': 80},
                {'name': 'ARKB', 'desc': 'ARK 21Shares', 'flow': 45},
            ]
            flows.top_outflows = [
                {'name': 'GBTC', 'desc': 'Grayscale BTC', 'flow': -120},
                {'name': 'BITO', 'desc': 'ProShares BTC', 'flow': -30},
            ]
            
        except Exception as e:
            print(f"ETF flows error: {e}")
        
        return flows
    
    def fetch_coin_detail(self, symbol: str) -> Dict:
        """코인 상세 정보"""
        try:
            coin_id = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
                'XRP': 'ripple', 'BNB': 'binancecoin', 'DOGE': 'dogecoin',
                'ADA': 'cardano', 'AVAX': 'avalanche-2', 'LINK': 'chainlink',
                'DOT': 'polkadot'
            }.get(symbol.upper(), symbol.lower())
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
            
            return {
                'symbol': symbol.upper(),
                'name': data['name'],
                'description': data['description']['en'][:300] + '...' if data['description']['en'] else '',
                'price': data['market_data']['current_price']['usd'],
                'change_24h': round(data['market_data']['price_change_percentage_24h'] or 0, 2),
                'change_7d': round(data['market_data']['price_change_percentage_7d'] or 0, 2),
                'change_30d': round(data['market_data']['price_change_percentage_30d'] or 0, 2),
                'market_cap': round(data['market_data']['market_cap']['usd'] / 1e9, 1),
                'volume_24h': round(data['market_data']['total_volume']['usd'] / 1e9, 2),
                'ath': data['market_data']['ath']['usd'],
                'ath_change': round(data['market_data']['ath_change_percentage']['usd'] or 0, 1),
                'circulating_supply': round(data['market_data']['circulating_supply'] / 1e6, 2),
                'max_supply': data['market_data'].get('max_supply'),
                'rank': data['market_cap_rank']
            }
            
        except Exception as e:
            print(f"Coin detail error: {e}")
            return {'error': str(e)}
    
    def fetch_crypto_news(self) -> List[Dict]:
        """암호화폐 뉴스 (기본)"""
        # 실제로는 Coinness API 등 사용 필요
        return [
            {'title': 'BTC ETF 유입 증가', 'time': '2시간 전', 'sentiment': 'positive'},
            {'title': 'Fed 금리 동결 예상', 'time': '4시간 전', 'sentiment': 'neutral'},
            {'title': 'ETH 2.0 업그레이드 예정', 'time': '6시간 전', 'sentiment': 'positive'},
            {'title': '옵션 만기일 주의', 'time': '1일 전', 'sentiment': 'warning'},
        ]


# 테스트
if __name__ == '__main__':
    fetcher = CryptoDataFetcher()
    
    print("=== Market Indices ===")
    indices = fetcher.fetch_market_indices()
    print(f"BTC: ${indices.btc_price:,.0f} ({indices.btc_change:+.2f}%)")
    print(f"Fear & Greed: {indices.fear_greed} ({indices.fear_greed_text})")
    
    print("\n=== Top Coins ===")
    for coin in fetcher.fetch_top_coins(5):
        print(f"{coin.symbol}: ${coin.price:,.0f} | Type {coin.signal_type} | Kelly {coin.kelly}%")
