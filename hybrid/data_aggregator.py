"""
Hybrid Data Aggregator
======================
기존 Crypto 분석 시스템 데이터를 NICE 5레이어 형식으로 변환

데이터 매핑:
- Market Gate → Layer 1 (기술분석)  
- Lead-Lag → Layer 2 (OnChain 상관관계)
- VCP → Layer 3 (패턴/심리)
- External APIs → Layer 4, 5 (매크로, ETF)
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class DataAggregator:
    """
    기존 시스템 데이터 → NICE 레이어 변환기
    
    사용법:
    >>> agg = DataAggregator()
    >>> data = agg.collect_all()
    >>> print(data['technical'])  # Layer 1 데이터
    """
    
    def __init__(self):
        self._cache = {}
        self._last_update = None
    
    def collect_all(self) -> Dict:
        """
        모든 소스에서 데이터 수집 후 NICE 형식으로 반환
        """
        return {
            'technical': self._collect_technical(),
            'onchain': self._collect_onchain(),
            'sentiment': self._collect_sentiment(),
            'macro': self._collect_macro(),
            'etf': self._collect_etf(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _collect_technical(self) -> Dict:
        """
        Layer 1: 기술분석 데이터
        Source: Market Gate, VCP
        """
        data = {
            'rsi': 50,
            'macd_signal': 'neutral',
            'volume_change_pct': 0
        }
        
        try:
            # Market Gate에서 기술 지표 가져오기
            from crypto_market.market_gate import run_market_gate_sync
            gate_result = run_market_gate_sync()
            
            if gate_result and gate_result.metrics:
                # EMA 기울기로 추세 판단
                ema_slope = gate_result.metrics.get('btc_ema200_slope_pct_20', 0)
                
                # RSI 대체값 (EMA 기울기 기반)
                if ema_slope > 1:
                    data['rsi'] = 65  # 상승 추세
                    data['macd_signal'] = 'bullish'
                elif ema_slope < -1:
                    data['rsi'] = 35  # 하락 추세
                    data['macd_signal'] = 'bearish'
                else:
                    data['rsi'] = 50
                    data['macd_signal'] = 'neutral'
                
                # Gate 점수로 거래량 대체
                if gate_result.score > 70:
                    data['volume_change_pct'] = 50
                elif gate_result.score > 50:
                    data['volume_change_pct'] = 25
                else:
                    data['volume_change_pct'] = 0
                    
        except ImportError:
            pass
        except Exception as e:
            print(f"Technical data error: {e}")
        
        return data
    
    def _collect_onchain(self) -> Dict:
        """
        Layer 2: OnChain 데이터
        Source: Lead-Lag 분석, 외부 API
        """
        data = {
            'whale_inflow_btc': 0,
            'mvrv': 2.0
        }
        
        try:
            # Lead-Lag에서 상관관계 데이터
            from crypto_market.market_gate import run_market_gate_sync
            gate_result = run_market_gate_sync()
            
            if gate_result and gate_result.metrics:
                # 펀딩레이트로 고래 활동 추정
                funding = gate_result.metrics.get('funding_rate', 0)
                if funding is not None:
                    if funding > 0.0005:
                        data['whale_inflow_btc'] = -5  # 숏 포지션 증가
                    elif funding < -0.0001:
                        data['whale_inflow_btc'] = 10  # 롱 포지션 증가
                    else:
                        data['whale_inflow_btc'] = 2
                        
        except ImportError:
            pass
        except Exception as e:
            print(f"OnChain data error: {e}")
        
        return data
    
    def _collect_sentiment(self) -> Dict:
        """
        Layer 3: 시장심리 데이터
        Source: Market Gate, 외부 API
        """
        data = {
            'fear_greed': 50,
            'liquidity': 'normal'
        }
        
        try:
            from crypto_market.market_gate import run_market_gate_sync
            gate_result = run_market_gate_sync()
            
            if gate_result and gate_result.metrics:
                fg = gate_result.metrics.get('fear_greed_index')
                if fg is not None:
                    data['fear_greed'] = fg
                
                # Alt breadth로 유동성 추정
                breadth = gate_result.metrics.get('alt_breadth_above_ema50', 0.5)
                if breadth is not None:
                    if breadth > 0.6:
                        data['liquidity'] = 'high'
                    elif breadth < 0.3:
                        data['liquidity'] = 'low'
                    else:
                        data['liquidity'] = 'normal'
                        
        except ImportError:
            pass
        except Exception as e:
            print(f"Sentiment data error: {e}")
        
        return data
    
    def _collect_macro(self) -> Dict:
        """
        Layer 4: 매크로 데이터
        Source: 외부 API (Fed, CPI)
        """
        # 기본값 (실시간 API 연동 시 업데이트)
        data = {
            'fed_stance': 'neutral',
            'cpi_trend': 'stable',
            'usd_trend': 'stable'
        }
        
        try:
            import yfinance as yf
            
            # DXY (USD Index) 추세 확인
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="5d")
            
            if not hist.empty and len(hist) >= 2:
                change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
                if change > 0.5:
                    data['usd_trend'] = 'rising'
                elif change < -0.5:
                    data['usd_trend'] = 'falling'
                    
        except Exception as e:
            print(f"Macro data error: {e}")
        
        return data
    
    def _collect_etf(self) -> Dict:
        """
        Layer 5: ETF/기관 데이터
        Source: 외부 API
        """
        # 기본값 (실시간 API 연동 시 업데이트)
        data = {
            'btc_etf_flow_m': 0,
            'eth_etf_flow_m': 0
        }
        
        try:
            import yfinance as yf
            
            # GBTC 볼륨으로 ETF 흐름 추정
            gbtc = yf.Ticker("GBTC")
            hist = gbtc.history(period="5d")
            
            if not hist.empty:
                avg_vol = hist['Volume'].mean()
                last_vol = hist['Volume'].iloc[-1]
                
                # 볼륨 증가를 유입으로 해석
                if last_vol > avg_vol * 1.5:
                    data['btc_etf_flow_m'] = 500
                elif last_vol > avg_vol:
                    data['btc_etf_flow_m'] = 200
                else:
                    data['btc_etf_flow_m'] = 50
                    
        except Exception as e:
            print(f"ETF data error: {e}")
        
        return data


# 테스트용
if __name__ == '__main__':
    agg = DataAggregator()
    data = agg.collect_all()
    
    print("=== Data Aggregator Test ===\n")
    for layer, values in data.items():
        if layer != 'timestamp':
            print(f"{layer}:")
            for k, v in values.items():
                print(f"  {k}: {v}")
            print()
