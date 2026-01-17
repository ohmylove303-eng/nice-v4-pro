"""
Palantir Mini - 경량 실시간 분석기
====================================
빠른 급등/급락 감지 및 세션별 분석

Features:
- quick_score(): 5초 내 빠른 점수 계산
- detect_surge(): 급등 코인 탐지
- get_current_session(): 현재 거래 세션 판단
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import hashlib


def _det_hash(symbol: str, seed: int = 0) -> int:
    """결정적 해시 생성"""
    h = hashlib.md5(f"{symbol.upper()}{seed}".encode()).hexdigest()
    return int(h[:8], 16) % 1000


@dataclass
class TradingSession:
    """거래 세션 정보"""
    name: str           # 세션 이름
    region: str         # 지역 (asia, europe, america, global)
    start_hour: int     # 시작 시간 (KST)
    start_minute: int   # 시작 분
    emoji: str          # 이모지
    liquidity: str      # 유동성 수준 (high, medium, low)
    volatility: str     # 변동성 수준


class PalantirMini:
    """
    경량 Palantir 분석기
    
    빠른 실시간 판단을 위한 경량화 버전
    - 5초 내 점수 계산
    - 급등/급락 감지
    - 세션별 분석
    """
    
    # 8개 거래 세션 정의 (KST 기준)
    SESSIONS = [
        TradingSession("아시아 프리마켓", "asia", 6, 30, "🌅", "low", "medium"),
        TradingSession("아시아 본격", "asia", 9, 0, "🌏", "high", "high"),
        TradingSession("아시아 재개", "asia", 12, 0, "☀️", "medium", "medium"),
        TradingSession("유럽 프리마켓", "europe", 14, 0, "🌍", "medium", "high"),
        TradingSession("유럽 본격", "europe", 16, 30, "🌆", "high", "high"),
        TradingSession("미국 프리마켓", "america", 19, 30, "🌇", "high", "high"),
        TradingSession("미국 본격", "america", 21, 30, "🌎", "high", "very_high"),
        TradingSession("글로벌 심야", "global", 0, 0, "🌐", "medium", "medium"),
    ]
    
    # 카테고리별 Perplexity 쿼리 템플릿
    PERPLEXITY_CATEGORIES = {
        'finance': "암호화폐 {symbol} 재무 분석 및 가격 전망",
        'prediction': "암호화폐 {symbol} 예측 시장 동향 및 선물 심리",
        'politics': "암호화폐 규제 및 정책 관련 최신 뉴스",
        'tech': "블록체인 기술 업데이트 및 {symbol} 네트워크 상태",
        'economy': "글로벌 경제 지표와 암호화폐 시장 영향",
        'geopolitics': "지정학적 이슈와 암호화폐 시장 영향"
    }
    
    def __init__(self):
        self.reliability = 0.85  # 기본 신뢰도
        
    def get_current_session(self, now: datetime = None) -> TradingSession:
        """현재 거래 세션 판단"""
        if now is None:
            now = datetime.now()
        
        current_minutes = now.hour * 60 + now.minute
        
        # 세션을 시간 순으로 정렬하여 현재 세션 찾기
        sessions_sorted = sorted(
            self.SESSIONS, 
            key=lambda s: s.start_hour * 60 + s.start_minute
        )
        
        current_session = sessions_sorted[-1]  # 기본값: 마지막 세션
        
        for session in sessions_sorted:
            session_minutes = session.start_hour * 60 + session.start_minute
            if current_minutes >= session_minutes:
                current_session = session
            else:
                break
        
        return current_session
    
    def get_next_session(self, now: datetime = None) -> tuple:
        """다음 세션 및 남은 시간 계산"""
        if now is None:
            now = datetime.now()
        
        current_minutes = now.hour * 60 + now.minute
        current_session = self.get_current_session(now)
        
        sessions_sorted = sorted(
            self.SESSIONS, 
            key=lambda s: s.start_hour * 60 + s.start_minute
        )
        
        # 다음 세션 찾기
        for i, session in enumerate(sessions_sorted):
            session_minutes = session.start_hour * 60 + session.start_minute
            if session_minutes > current_minutes:
                minutes_until = session_minutes - current_minutes
                return session, minutes_until
        
        # 다음 날 첫 세션
        first_session = sessions_sorted[0]
        minutes_until = (24 * 60 - current_minutes) + (first_session.start_hour * 60 + first_session.start_minute)
        return first_session, minutes_until
    
    def quick_score(
        self, 
        symbol: str, 
        price: float, 
        change_5m: float, 
        volume_ratio: float,
        session: TradingSession = None
    ) -> Dict:
        """
        빠른 점수 계산 (5초 내)
        
        Args:
            symbol: 코인 심볼
            price: 현재 가격
            change_5m: 5분 변동률 (%)
            volume_ratio: 거래량 비율 (평균 대비)
            session: 현재 세션
        
        Returns:
            quick_score, surge_signal, confidence
        """
        base_score = 50
        
        # 1. 모멘텀 점수 (최대 +25)
        if change_5m >= 5:
            momentum_score = 25
        elif change_5m >= 3:
            momentum_score = 20
        elif change_5m >= 1:
            momentum_score = 10
        elif change_5m >= 0:
            momentum_score = 5
        else:
            momentum_score = max(-10, change_5m * 2)  # 하락 시 감점
        
        # 2. 거래량 점수 (최대 +15)
        if volume_ratio >= 3:
            volume_score = 15
        elif volume_ratio >= 2:
            volume_score = 10
        elif volume_ratio >= 1.5:
            volume_score = 5
        else:
            volume_score = 0
        
        # 3. 세션 보너스 (최대 +10)
        session_bonus = 0
        if session:
            if session.liquidity == 'high':
                session_bonus += 5
            if session.volatility in ['high', 'very_high']:
                session_bonus += 5
        
        # 총점 계산
        total_score = min(100, max(0, base_score + momentum_score + volume_score + session_bonus))
        
        # 급등 신호 판단
        if total_score >= 80 and change_5m >= 3:
            surge_signal = "🚀 초급등"
        elif total_score >= 70 and change_5m >= 2:
            surge_signal = "📈 급등"
        elif total_score >= 60:
            surge_signal = "⬆️ 상승"
        elif total_score <= 30 and change_5m <= -3:
            surge_signal = "📉 급락"
        else:
            surge_signal = "➡️ 보합"
        
        # 신뢰도 계산
        confidence = min(1.0, 0.6 + (volume_ratio * 0.1) + (abs(change_5m) * 0.02))
        
        return {
            'symbol': symbol,
            'quick_score': total_score,
            'surge_signal': surge_signal,
            'confidence': round(confidence, 2),
            'breakdown': {
                'base': base_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'session': session_bonus
            }
        }
    
    def detect_surge(
        self, 
        coins_data: List[Dict],
        threshold_change: float = 3.0,
        threshold_volume: float = 1.5
    ) -> List[Dict]:
        """
        급등 코인 탐지
        
        Args:
            coins_data: 코인 데이터 리스트 (symbol, price, change_5m, volume_ratio)
            threshold_change: 변동률 임계값 (%)
            threshold_volume: 거래량 비율 임계값
        
        Returns:
            급등 후보 코인 리스트 (점수순 정렬)
        """
        session = self.get_current_session()
        surge_candidates = []
        
        for coin in coins_data:
            symbol = coin.get('symbol', '')
            change_5m = coin.get('change_5m', 0)
            volume_ratio = coin.get('volume_ratio', 1)
            
            # 급등 조건 체크
            if change_5m >= threshold_change or volume_ratio >= threshold_volume * 2:
                score_result = self.quick_score(
                    symbol=symbol,
                    price=coin.get('price', 0),
                    change_5m=change_5m,
                    volume_ratio=volume_ratio,
                    session=session
                )
                
                surge_candidates.append({
                    **coin,
                    **score_result,
                    'session': session.name
                })
        
        # 점수순 정렬
        return sorted(surge_candidates, key=lambda x: x['quick_score'], reverse=True)
    
    def get_perplexity_prompts(self, symbol: str) -> Dict[str, str]:
        """Perplexity AI용 카테고리별 프롬프트 생성"""
        return {
            category: template.format(symbol=symbol)
            for category, template in self.PERPLEXITY_CATEGORIES.items()
        }
    
    def calculate_palantir_reliability(
        self, 
        data_freshness: float = 0.9,
        source_count: int = 3,
        cross_validation: bool = True
    ) -> float:
        """
        Palantir 신뢰도 계산 (NICE 점수에 반영용)
        
        Args:
            data_freshness: 데이터 신선도 (0~1)
            source_count: 데이터 소스 수
            cross_validation: 교차 검증 여부
        
        Returns:
            reliability (0~1)
        """
        base = 0.6
        
        # 신선도 가중치
        freshness_weight = data_freshness * 0.2
        
        # 소스 수 가중치
        source_weight = min(0.15, source_count * 0.03)
        
        # 교차 검증 보너스
        validation_bonus = 0.05 if cross_validation else 0
        
        reliability = min(1.0, base + freshness_weight + source_weight + validation_bonus)
        self.reliability = reliability
        
        return round(reliability, 3)


# 테스트
if __name__ == '__main__':
    mini = PalantirMini()
    
    # 현재 세션 테스트
    session = mini.get_current_session()
    print(f"현재 세션: {session.emoji} {session.name}")
    
    next_session, minutes = mini.get_next_session()
    print(f"다음 세션: {next_session.name} ({minutes}분 후)")
    
    # 빠른 점수 테스트
    result = mini.quick_score('BTC', 98000, 4.5, 2.3, session)
    print(f"\nBTC 빠른 점수: {result['quick_score']} - {result['surge_signal']}")
    
    # 급등 탐지 테스트
    test_coins = [
        {'symbol': 'BTC', 'price': 98000, 'change_5m': 4.5, 'volume_ratio': 2.3},
        {'symbol': 'ETH', 'price': 3500, 'change_5m': 1.2, 'volume_ratio': 1.1},
        {'symbol': 'SOL', 'price': 195, 'change_5m': 8.5, 'volume_ratio': 3.5},
    ]
    surges = mini.detect_surge(test_coins)
    print(f"\n급등 후보: {len(surges)}개")
    for s in surges:
        print(f"  - {s['symbol']}: {s['quick_score']} ({s['surge_signal']})")
