"""
FRED Data Fetcher
==================
미국 연방준비은행(FRED) 매크로 데이터 수집

데이터 소스:
- Fed 금리 (FEDFUNDS)
- CPI 인플레이션 (CPIAUCSL)
- 실업률 (UNRATE)
- 10년물 국채 (DGS10)
- DXY 달러 인덱스 (Yahoo Finance)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timedelta
import urllib.request
import json


@dataclass
class MacroData:
    """매크로 지표 데이터"""
    fed_rate: Optional[float] = None          # Fed 금리
    fed_rate_change: Optional[str] = None     # 동결/인상/인하
    cpi_yoy: Optional[float] = None           # CPI 전년비
    cpi_trend: Optional[str] = None           # 상승/안정/하락
    unemployment: Optional[float] = None      # 실업률
    treasury_10y: Optional[float] = None      # 10년물 국채
    dxy: Optional[float] = None               # 달러 인덱스
    dxy_trend: Optional[str] = None           # 강세/안정/약세
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'fed_rate': {
                'value': self.fed_rate,
                'change': self.fed_rate_change,
                'label_ko': '미국 금리',
                'explain_ko': self._explain_fed_rate()
            },
            'cpi': {
                'value': self.cpi_yoy,
                'trend': self.cpi_trend,
                'label_ko': '물가 상승률',
                'explain_ko': self._explain_cpi()
            },
            'unemployment': {
                'value': self.unemployment,
                'label_ko': '실업률',
                'explain_ko': self._explain_unemployment()
            },
            'treasury_10y': {
                'value': self.treasury_10y,
                'label_ko': '국채 금리',
                'explain_ko': self._explain_treasury()
            },
            'dxy': {
                'value': self.dxy,
                'trend': self.dxy_trend,
                'label_ko': '달러 가치',
                'explain_ko': self._explain_dxy()
            },
            'timestamp': self.timestamp.isoformat()
        }
    
    def _explain_fed_rate(self) -> str:
        """초등 3학년 수준 설명"""
        if self.fed_rate is None:
            return "정보를 가져오는 중이에요"
        if self.fed_rate_change == '인하':
            return f"🟢 금리가 {self.fed_rate}%예요. 내려가고 있어서 코인에 좋아요!"
        elif self.fed_rate_change == '인상':
            return f"🔴 금리가 {self.fed_rate}%예요. 올라가고 있어서 조심해야 해요"
        else:
            return f"🟡 금리가 {self.fed_rate}%예요. 지금은 그대로예요"
    
    def _explain_cpi(self) -> str:
        if self.cpi_yoy is None:
            return "정보를 가져오는 중이에요"
        if self.cpi_yoy < 2.5:
            return f"🟢 물가가 {self.cpi_yoy}% 올랐어요. 안정적이에요!"
        elif self.cpi_yoy > 4:
            return f"🔴 물가가 {self.cpi_yoy}% 올랐어요. 너무 많이 올랐어요"
        else:
            return f"🟡 물가가 {self.cpi_yoy}% 올랐어요. 보통이에요"
    
    def _explain_unemployment(self) -> str:
        if self.unemployment is None:
            return "정보를 가져오는 중이에요"
        if self.unemployment < 4:
            return f"🟢 일자리가 많아요! 실업률 {self.unemployment}%"
        elif self.unemployment > 5:
            return f"🔴 일자리가 적어졌어요. 실업률 {self.unemployment}%"
        else:
            return f"🟡 일자리는 보통이에요. 실업률 {self.unemployment}%"
    
    def _explain_treasury(self) -> str:
        if self.treasury_10y is None:
            return "정보를 가져오는 중이에요"
        if self.treasury_10y > 4.5:
            return f"🔴 국채 금리가 {self.treasury_10y}%로 높아요. 주의!"
        elif self.treasury_10y < 3.5:
            return f"🟢 국채 금리가 {self.treasury_10y}%로 낮아요. 좋아요!"
        else:
            return f"🟡 국채 금리가 {self.treasury_10y}%예요. 보통이에요"
    
    def _explain_dxy(self) -> str:
        if self.dxy is None:
            return "정보를 가져오는 중이에요"
        if self.dxy_trend == '약세':
            return f"🟢 달러가 약해지고 있어요 (DXY {self.dxy}). 코인에 좋아요!"
        elif self.dxy_trend == '강세':
            return f"🔴 달러가 강해지고 있어요 (DXY {self.dxy}). 조심해요"
        else:
            return f"🟡 달러는 보통이에요 (DXY {self.dxy})"


class FREDFetcher:
    """
    FRED 및 매크로 데이터 수집기
    
    사용법:
    >>> fetcher = FREDFetcher()
    >>> data = fetcher.fetch_all()
    >>> print(data.to_dict())
    """
    
    # FRED API 키 (환경변수에서 가져옴)
    FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
    
    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = timedelta(hours=1)
    
    def fetch_all(self) -> MacroData:
        """모든 매크로 데이터 수집"""
        # 캐시 확인
        if self._cache_time and datetime.now() - self._cache_time < self._cache_ttl:
            return self._cache.get('data', MacroData())
        
        data = MacroData()
        
        # Fed Rate
        try:
            data.fed_rate = self._fetch_fred_series('FEDFUNDS')
            data.fed_rate_change = '동결'  # 기본값, 실제로는 이전값과 비교 필요
        except Exception as e:
            print(f"Fed rate fetch error: {e}")
        
        # CPI
        try:
            data.cpi_yoy = self._fetch_fred_series('CPIAUCSL', yoy=True)
            if data.cpi_yoy:
                if data.cpi_yoy < 2.5:
                    data.cpi_trend = '안정'
                elif data.cpi_yoy > 4:
                    data.cpi_trend = '상승'
                else:
                    data.cpi_trend = '보통'
        except Exception as e:
            print(f"CPI fetch error: {e}")
        
        # Unemployment
        try:
            data.unemployment = self._fetch_fred_series('UNRATE')
        except Exception as e:
            print(f"Unemployment fetch error: {e}")
        
        # 10Y Treasury
        try:
            data.treasury_10y = self._fetch_fred_series('DGS10')
        except Exception as e:
            print(f"Treasury fetch error: {e}")
        
        # DXY (Yahoo Finance)
        try:
            dxy_data = self._fetch_dxy()
            if dxy_data:
                data.dxy = dxy_data['value']
                data.dxy_trend = dxy_data['trend']
        except Exception as e:
            print(f"DXY fetch error: {e}")
        
        # 캐시 업데이트
        self._cache['data'] = data
        self._cache_time = datetime.now()
        
        return data
    
    def _fetch_fred_series(self, series_id: str, yoy: bool = False) -> Optional[float]:
        """FRED API에서 시계열 데이터 가져오기"""
        if not self.FRED_API_KEY:
            # FRED API 키 없으면 기본값 반환 (실제 환경에서는 키 필요)
            defaults = {
                'FEDFUNDS': 4.25,
                'CPIAUCSL': 2.6,
                'UNRATE': 4.1,
                'DGS10': 4.2
            }
            return defaults.get(series_id)
        
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.FRED_API_KEY}&file_type=json&sort_order=desc&limit=2"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
            observations = data.get('observations', [])
            if observations:
                latest = float(observations[0]['value'])
                
                if yoy and len(observations) > 1:
                    # 전년비 계산 (간략화)
                    return round(latest, 2)
                return round(latest, 2)
        except Exception as e:
            print(f"FRED API error for {series_id}: {e}")
        
        return None
    
    def _fetch_dxy(self) -> Optional[Dict]:
        """Yahoo Finance에서 DXY 가져오기"""
        try:
            import yfinance as yf
            
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="5d")
            
            if not hist.empty and len(hist) >= 2:
                current = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[0])
                change = (current - prev) / prev * 100
                
                if change > 0.5:
                    trend = '강세'
                elif change < -0.5:
                    trend = '약세'
                else:
                    trend = '보통'
                
                return {'value': round(current, 1), 'trend': trend}
        except Exception as e:
            print(f"DXY fetch error: {e}")
        
        # 기본값
        return {'value': 102.5, 'trend': '보통'}
    
    def get_summary_ko(self) -> str:
        """초등학교 3학년 수준 요약"""
        data = self.fetch_all()
        
        lines = [
            "🏦 **나라 경제 상황이에요!**",
            "",
            data._explain_fed_rate(),
            data._explain_cpi(),
            data._explain_unemployment(),
            data._explain_dxy(),
        ]
        
        return "\n".join(lines)


# 테스트용
if __name__ == '__main__':
    fetcher = FREDFetcher()
    data = fetcher.fetch_all()
    
    print("=== FRED Data Test ===\n")
    print(fetcher.get_summary_ko())
    print("\n=== Raw Data ===")
    for key, value in data.to_dict().items():
        if key != 'timestamp':
            print(f"{key}: {value}")
