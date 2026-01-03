"""
NICE v4 5-Layer Scorer
======================
원본 NICE 모델 점수 계산 로직 (100% 보존)

5개 레이어:
- Layer 1: 기술분석 (RSI, MACD, 거래량) → /30점
- Layer 2: OnChain (고래 유입, MVRV) → /30점  
- Layer 3: 시장심리 (공포탐욕, 유동성) → /30점
- Layer 4: 매크로 (Fed, CPI, USD) → /30점
- Layer 5: ETF/기관 (BTC ETF, ETH ETF) → /30점

총점: /150점 → 정규화하여 /100점으로 변환
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime


@dataclass
class LayerScore:
    """개별 레이어 점수"""
    name: str
    score: float  # 0-30
    max_score: float = 30.0
    details: Dict[str, float] = field(default_factory=dict)
    
    @property
    def percentage(self) -> float:
        return (self.score / self.max_score) * 100


@dataclass 
class NICEScore:
    """NICE 전체 점수"""
    layer1_technical: LayerScore
    layer2_onchain: LayerScore
    layer3_sentiment: LayerScore
    layer4_macro: LayerScore
    layer5_etf: LayerScore
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def total_raw(self) -> float:
        """원점수 합계 (0-150)"""
        return (
            self.layer1_technical.score +
            self.layer2_onchain.score +
            self.layer3_sentiment.score +
            self.layer4_macro.score +
            self.layer5_etf.score
        )
    
    @property
    def total_normalized(self) -> float:
        """정규화 점수 (0-100)"""
        return (self.total_raw / 150) * 100
    
    def to_dict(self) -> Dict:
        return {
            'total_score': round(self.total_normalized, 1),
            'total_raw': round(self.total_raw, 1),
            'layers': {
                'technical': {
                    'score': self.layer1_technical.score,
                    'max': 30,
                    'details': self.layer1_technical.details
                },
                'onchain': {
                    'score': self.layer2_onchain.score,
                    'max': 30,
                    'details': self.layer2_onchain.details
                },
                'sentiment': {
                    'score': self.layer3_sentiment.score,
                    'max': 30,
                    'details': self.layer3_sentiment.details
                },
                'macro': {
                    'score': self.layer4_macro.score,
                    'max': 30,
                    'details': self.layer4_macro.details
                },
                'etf': {
                    'score': self.layer5_etf.score,
                    'max': 30,
                    'details': self.layer5_etf.details
                }
            },
            'timestamp': self.timestamp.isoformat()
        }


class NICEScorer:
    """
    NICE v4 5레이어 점수 계산기
    
    사용법:
    >>> scorer = NICEScorer()
    >>> score = scorer.calculate(data)
    >>> print(score.total_normalized)  # 0-100
    """
    
    def __init__(self):
        self.weights = {
            'technical': 1.0,
            'onchain': 1.0,
            'sentiment': 1.0,
            'macro': 1.0,
            'etf': 1.0
        }
    
    def calculate(self, data: Optional[Dict] = None) -> NICEScore:
        """
        5레이어 점수 계산
        
        Args:
            data: 외부 데이터 (없으면 기본값 사용)
            
        Returns:
            NICEScore: 전체 점수 객체
        """
        data = data or {}
        
        return NICEScore(
            layer1_technical=self._score_layer1_technical(data.get('technical', {})),
            layer2_onchain=self._score_layer2_onchain(data.get('onchain', {})),
            layer3_sentiment=self._score_layer3_sentiment(data.get('sentiment', {})),
            layer4_macro=self._score_layer4_macro(data.get('macro', {})),
            layer5_etf=self._score_layer5_etf(data.get('etf', {}))
        )
    
    def _score_layer1_technical(self, data: Dict) -> LayerScore:
        """
        Layer 1: 기술분석 점수 (0-30)
        
        - RSI(5m) 50-70 → +10점, <30 or >85 → +5점 (역추세)
        - MACD 상향 → +10점, 하향 → +2점
        - 거래량 +50% 이상 → +10점
        """
        score = 0.0
        details = {}
        
        # RSI 점수 (0-10)
        rsi = data.get('rsi', 50)
        if 50 <= rsi <= 70:
            rsi_score = 10
        elif rsi > 85:
            rsi_score = 8  # 과매수 역추세 기회
        elif rsi < 30:
            rsi_score = 8  # 과매도 역추세 기회
        else:
            rsi_score = 5
        score += rsi_score
        details['rsi'] = rsi_score
        
        # MACD 점수 (0-10)
        macd_signal = data.get('macd_signal', 'neutral')
        if macd_signal == 'bullish':
            macd_score = 10
        elif macd_signal == 'bearish':
            macd_score = 2
        else:
            macd_score = 5
        score += macd_score
        details['macd'] = macd_score
        
        # 거래량 점수 (0-10)
        volume_change = data.get('volume_change_pct', 0)
        if volume_change >= 50:
            vol_score = 10
        elif volume_change >= 20:
            vol_score = 7
        elif volume_change >= 0:
            vol_score = 5
        else:
            vol_score = 2
        score += vol_score
        details['volume'] = vol_score
        
        return LayerScore(name='technical', score=min(score, 30), details=details)
    
    def _score_layer2_onchain(self, data: Dict) -> LayerScore:
        """
        Layer 2: OnChain 점수 (0-30)
        
        - 고래 유입 양수 → +15점
        - MVRV 1.5-3.0 (건강) → +15점
        """
        score = 0.0
        details = {}
        
        # 고래 유입 (0-15)
        whale_inflow = data.get('whale_inflow_btc', 0)
        if whale_inflow > 10:
            whale_score = 15
        elif whale_inflow > 0:
            whale_score = 10
        elif whale_inflow > -5:
            whale_score = 5
        else:
            whale_score = 0
        score += whale_score
        details['whale_inflow'] = whale_score
        
        # MVRV (0-15)
        mvrv = data.get('mvrv', 2.0)
        if 1.5 <= mvrv <= 3.0:
            mvrv_score = 15
        elif 1.0 <= mvrv <= 4.0:
            mvrv_score = 10
        else:
            mvrv_score = 5
        score += mvrv_score
        details['mvrv'] = mvrv_score
        
        return LayerScore(name='onchain', score=min(score, 30), details=details)
    
    def _score_layer3_sentiment(self, data: Dict) -> LayerScore:
        """
        Layer 3: 시장심리 점수 (0-30)
        
        - 공포탐욕지수 40-60 (중립) → +15점
        - 유동성 지표 양호 → +15점
        """
        score = 0.0
        details = {}
        
        # 공포탐욕 (0-15)
        fear_greed = data.get('fear_greed', 50)
        if 40 <= fear_greed <= 60:
            fg_score = 15  # 중립이 최적
        elif 25 <= fear_greed <= 75:
            fg_score = 12
        elif fear_greed < 25:
            fg_score = 10  # 극심한 공포 = 매수 기회
        else:
            fg_score = 5   # 극심한 탐욕 = 위험
        score += fg_score
        details['fear_greed'] = fg_score
        
        # 유동성 (0-15)
        liquidity = data.get('liquidity', 'normal')
        if liquidity == 'high':
            liq_score = 15
        elif liquidity == 'normal':
            liq_score = 10
        else:
            liq_score = 5
        score += liq_score
        details['liquidity'] = liq_score
        
        return LayerScore(name='sentiment', score=min(score, 30), details=details)
    
    def _score_layer4_macro(self, data: Dict) -> LayerScore:
        """
        Layer 4: 매크로 점수 (0-30)
        
        - Fed 금리 안정/인하 → +10점
        - CPI 하락 추세 → +10점
        - USD Index 하락 → +10점 (리스크온)
        """
        score = 0.0
        details = {}
        
        # Fed 금리 (0-10)
        fed_stance = data.get('fed_stance', 'neutral')
        if fed_stance == 'dovish':
            fed_score = 10
        elif fed_stance == 'neutral':
            fed_score = 7
        else:
            fed_score = 3
        score += fed_score
        details['fed'] = fed_score
        
        # CPI (0-10)
        cpi_trend = data.get('cpi_trend', 'stable')
        if cpi_trend == 'falling':
            cpi_score = 10
        elif cpi_trend == 'stable':
            cpi_score = 7
        else:
            cpi_score = 3
        score += cpi_score
        details['cpi'] = cpi_score
        
        # USD Index (0-10)
        usd_trend = data.get('usd_trend', 'stable')
        if usd_trend == 'falling':
            usd_score = 10  # 달러 약세 = 리스크온
        elif usd_trend == 'stable':
            usd_score = 7
        else:
            usd_score = 3
        score += usd_score
        details['usd'] = usd_score
        
        return LayerScore(name='macro', score=min(score, 30), details=details)
    
    def _score_layer5_etf(self, data: Dict) -> LayerScore:
        """
        Layer 5: ETF/기관 점수 (0-30)
        
        - BTC ETF 유입 > 0 → +15점
        - ETH ETF 유입 > 0 → +15점
        """
        score = 0.0
        details = {}
        
        # BTC ETF (0-15)
        btc_etf_flow = data.get('btc_etf_flow_m', 0)
        if btc_etf_flow > 500:
            btc_score = 15
        elif btc_etf_flow > 0:
            btc_score = 12
        elif btc_etf_flow > -100:
            btc_score = 8
        else:
            btc_score = 3
        score += btc_score
        details['btc_etf'] = btc_score
        
        # ETH ETF (0-15)
        eth_etf_flow = data.get('eth_etf_flow_m', 0)
        if eth_etf_flow > 200:
            eth_score = 15
        elif eth_etf_flow > 0:
            eth_score = 12
        elif eth_etf_flow > -50:
            eth_score = 8
        else:
            eth_score = 3
        score += eth_score
        details['eth_etf'] = eth_score
        
        return LayerScore(name='etf', score=min(score, 30), details=details)


# 테스트용
if __name__ == '__main__':
    scorer = NICEScorer()
    
    # 샘플 데이터로 테스트
    test_data = {
        'technical': {'rsi': 62, 'macd_signal': 'bullish', 'volume_change_pct': 38},
        'onchain': {'whale_inflow_btc': 12, 'mvrv': 2.1},
        'sentiment': {'fear_greed': 45, 'liquidity': 'normal'},
        'macro': {'fed_stance': 'neutral', 'cpi_trend': 'falling', 'usd_trend': 'stable'},
        'etf': {'btc_etf_flow_m': 1200, 'eth_etf_flow_m': 380}
    }
    
    result = scorer.calculate(test_data)
    print(f"Total Score: {result.total_normalized:.1f}/100")
    print(f"Raw Score: {result.total_raw:.1f}/150")
    for layer_name, layer_data in result.to_dict()['layers'].items():
        print(f"  {layer_name}: {layer_data['score']}/{layer_data['max']}")
