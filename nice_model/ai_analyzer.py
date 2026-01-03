"""
NICE v4 AI Analyzer
====================
LLM(Gemini/GPT) 기반 NICE 분석 모듈

사용 규칙:
- Gemini: 필요시마다 자유롭게 사용
- GPT-4: 하루 2번 (오전 9시, 오후 9시) 중요 시간에만 사용

유동성 높은 시간:
- 09:00 KST (한국 장 오전)
- 21:00 KST (미국 장 시작)
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Literal
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AIUsageTracker:
    """AI 사용량 추적기"""
    gpt_daily_limit: int = 2
    gpt_used_today: int = 0
    gpt_last_used: Optional[datetime] = None
    gpt_allowed_hours: list = field(default_factory=lambda: [9, 21])  # 09:00, 21:00 KST
    gemini_used_today: int = 0
    last_reset_date: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'gpt_daily_limit': self.gpt_daily_limit,
            'gpt_used_today': self.gpt_used_today,
            'gpt_last_used': self.gpt_last_used.isoformat() if self.gpt_last_used else None,
            'gpt_allowed_hours': self.gpt_allowed_hours,
            'gemini_used_today': self.gemini_used_today,
            'last_reset_date': self.last_reset_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AIUsageTracker':
        tracker = cls()
        tracker.gpt_daily_limit = data.get('gpt_daily_limit', 2)
        tracker.gpt_used_today = data.get('gpt_used_today', 0)
        tracker.gpt_allowed_hours = data.get('gpt_allowed_hours', [9, 21])
        tracker.gemini_used_today = data.get('gemini_used_today', 0)
        tracker.last_reset_date = data.get('last_reset_date', '')
        
        if data.get('gpt_last_used'):
            tracker.gpt_last_used = datetime.fromisoformat(data['gpt_last_used'])
        
        return tracker


class NICEAIAnalyzer:
    """
    NICE 모델 AI 분석기
    
    Gemini: 필요시 자유롭게 사용
    GPT-4: 하루 2번 (09:00, 21:00 KST)만 사용
    """
    
    TRACKER_FILE = Path(__file__).parent / '.ai_usage.json'
    
    def __init__(self):
        self._load_env()
        self.tracker = self._load_tracker()
        self._check_daily_reset()
    
    def _load_env(self):
        """환경변수 로드"""
        # .env 파일 경로
        env_path = Path(__file__).parent / '.env'
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
        
        # API 키 설정
        self.gemini_key = os.environ.get('GEMINI_API_KEY', '')
        self.openai_key = os.environ.get('OPENAI_API_KEY', '')
        self.gpt_key = os.environ.get('GPT_API_KEY', self.openai_key)  # 별칭
        self.fred_key = os.environ.get('FRED_API_KEY', '')
    
    def _load_tracker(self) -> AIUsageTracker:
        """사용량 추적기 로드"""
        if self.TRACKER_FILE.exists():
            try:
                with open(self.TRACKER_FILE, 'r') as f:
                    data = json.load(f)
                return AIUsageTracker.from_dict(data)
            except:
                pass
        return AIUsageTracker()
    
    def _save_tracker(self):
        """사용량 추적기 저장"""
        with open(self.TRACKER_FILE, 'w') as f:
            json.dump(self.tracker.to_dict(), f, indent=2)
    
    def _check_daily_reset(self):
        """일일 리셋 확인"""
        today = datetime.now().strftime('%Y-%m-%d')
        if self.tracker.last_reset_date != today:
            self.tracker.gpt_used_today = 0
            self.tracker.gemini_used_today = 0
            self.tracker.last_reset_date = today
            self._save_tracker()
    
    def can_use_gpt(self) -> tuple[bool, str]:
        """
        GPT 사용 가능 여부 확인
        
        Returns:
            (can_use, reason)
        """
        now = datetime.now()
        current_hour = now.hour
        
        # 일일 한도 확인
        if self.tracker.gpt_used_today >= self.tracker.gpt_daily_limit:
            return False, f"일일 한도 초과 ({self.tracker.gpt_used_today}/{self.tracker.gpt_daily_limit})"
        
        # 허용 시간 확인 (±30분 여유)
        allowed = False
        for allowed_hour in self.tracker.gpt_allowed_hours:
            if abs(current_hour - allowed_hour) <= 1:  # ±1시간 허용
                allowed = True
                break
        
        if not allowed:
            next_allowed = min(
                [h for h in self.tracker.gpt_allowed_hours if h > current_hour],
                default=self.tracker.gpt_allowed_hours[0]
            )
            return False, f"GPT 사용 제한 시간. 다음 허용: {next_allowed:02d}:00 KST"
        
        return True, "사용 가능"
    
    def can_use_gemini(self) -> tuple[bool, str]:
        """Gemini 사용 가능 여부 (항상 허용)"""
        if not self.gemini_key:
            return False, "Gemini API 키 없음"
        return True, "사용 가능"
    
    def analyze_with_gemini(self, prompt: str, context: Dict = None) -> Dict:
        """
        Gemini로 NICE 분석 (자유 사용)
        
        Args:
            prompt: 분석 프롬프트
            context: NICE 데이터 컨텍스트
            
        Returns:
            분석 결과
        """
        can_use, reason = self.can_use_gemini()
        if not can_use:
            return {'error': reason, 'provider': 'gemini'}
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # 컨텍스트 포함 프롬프트 생성
            full_prompt = self._build_nice_prompt(prompt, context)
            
            response = model.generate_content(full_prompt)
            
            # 사용량 기록
            self.tracker.gemini_used_today += 1
            self._save_tracker()
            
            return {
                'success': True,
                'provider': 'gemini',
                'response': response.text,
                'usage_today': self.tracker.gemini_used_today,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e), 'provider': 'gemini'}
    
    def analyze_with_gpt(self, prompt: str, context: Dict = None, force: bool = False) -> Dict:
        """
        GPT로 NICE 분석 (제한적 사용)
        
        Args:
            prompt: 분석 프롬프트
            context: NICE 데이터 컨텍스트
            force: 강제 사용 (시간 제한 무시)
            
        Returns:
            분석 결과
        """
        can_use, reason = self.can_use_gpt()
        if not can_use and not force:
            return {'error': reason, 'provider': 'gpt', 'can_use_gemini': True}
        
        if not self.gpt_key and not self.openai_key:
            return {'error': 'GPT API 키 없음', 'provider': 'gpt'}
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.gpt_key or self.openai_key)
            
            full_prompt = self._build_nice_prompt(prompt, context)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency analyst specializing in the NICE 5-layer scoring model."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            # 사용량 기록
            self.tracker.gpt_used_today += 1
            self.tracker.gpt_last_used = datetime.now()
            self._save_tracker()
            
            return {
                'success': True,
                'provider': 'gpt',
                'response': response.choices[0].message.content,
                'usage_today': self.tracker.gpt_used_today,
                'remaining': self.tracker.gpt_daily_limit - self.tracker.gpt_used_today,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e), 'provider': 'gpt'}
    
    def _build_nice_prompt(self, prompt: str, context: Dict = None) -> str:
        """NICE 분석용 프롬프트 생성"""
        base_context = """
NICE v4 5-Layer Scoring System:
- Layer 1 (Technical): RSI, MACD, Volume (30점)
- Layer 2 (OnChain): Whale Flow, MVRV (30점)  
- Layer 3 (Sentiment): Fear & Greed, Funding Rate (30점)
- Layer 4 (Macro): Fed Rate, CPI, DXY, VIX (30점)
- Layer 5 (ETF): BTC/ETH ETF Flows (30점)

Score Interpretation:
- 75+ points: Type A (즉시 매수, Kelly 4%)
- 55-74 points: Type B (신중히, Kelly 2%)
- <55 points: Type C (거래 금지)

Current Time: {time}
""".format(time=datetime.now().strftime('%Y-%m-%d %H:%M KST'))
        
        if context:
            context_str = f"\n현재 데이터:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n"
        else:
            context_str = ""
        
        return f"{base_context}{context_str}\n분석 요청:\n{prompt}"
    
    def auto_analyze(self, context: Dict, prefer_gpt: bool = False) -> Dict:
        """
        자동 분석 (상황에 따라 Gemini 또는 GPT 선택)
        
        Args:
            context: NICE 데이터
            prefer_gpt: GPT 선호 (허용 시간일 때만)
            
        Returns:
            분석 결과
        """
        prompt = """
현재 암호화폐 시장을 NICE 5레이어 관점에서 분석해주세요:

1. 전체 신호 (Type A/B/C)
2. 각 레이어별 핵심 포인트
3. 추천 진입 전략
4. 주요 리스크 요인
5. 24시간 전망

간결하게 한국어로 응답해주세요.
"""
        
        if prefer_gpt:
            can_use_gpt, _ = self.can_use_gpt()
            if can_use_gpt:
                return self.analyze_with_gpt(prompt, context)
        
        # 기본은 Gemini
        return self.analyze_with_gemini(prompt, context)
    
    def get_usage_status(self) -> Dict:
        """현재 AI 사용량 상태"""
        can_gpt, gpt_reason = self.can_use_gpt()
        can_gemini, gemini_reason = self.can_use_gemini()
        
        return {
            'gpt': {
                'available': can_gpt,
                'reason': gpt_reason,
                'used_today': self.tracker.gpt_used_today,
                'limit': self.tracker.gpt_daily_limit,
                'allowed_hours': self.tracker.gpt_allowed_hours,
                'last_used': self.tracker.gpt_last_used.isoformat() if self.tracker.gpt_last_used else None
            },
            'gemini': {
                'available': can_gemini,
                'reason': gemini_reason,
                'used_today': self.tracker.gemini_used_today,
                'limit': 'unlimited'
            },
            'current_hour': datetime.now().hour,
            'date': datetime.now().strftime('%Y-%m-%d')
        }


# 테스트
if __name__ == '__main__':
    analyzer = NICEAIAnalyzer()
    
    print("=== AI Analyzer Status ===\n")
    
    status = analyzer.get_usage_status()
    print(f"Current Hour: {status['current_hour']:02d}:00")
    print(f"\nGPT-4:")
    print(f"  Available: {status['gpt']['available']}")
    print(f"  Reason: {status['gpt']['reason']}")
    print(f"  Used Today: {status['gpt']['used_today']}/{status['gpt']['limit']}")
    print(f"  Allowed Hours: {status['gpt']['allowed_hours']}")
    
    print(f"\nGemini:")
    print(f"  Available: {status['gemini']['available']}")
    print(f"  Used Today: {status['gemini']['used_today']}")
    
    # 테스트 분석 (Gemini)
    print("\n=== Test Analysis (Gemini) ===")
    
    sample_context = {
        'score': 78,
        'type': 'A',
        'layers': {
            'technical': {'score': 25, 'rsi': 62, 'macd': 'bullish'},
            'onchain': {'score': 22, 'whale_flow': 15},
            'sentiment': {'score': 18, 'fear_greed': 45},
            'macro': {'score': 8, 'fed_stance': 'neutral'},
            'etf': {'score': 5, 'btc_flow': -50}
        }
    }
    
    result = analyzer.auto_analyze(sample_context)
    if result.get('success'):
        print(f"Provider: {result['provider']}")
        print(f"Response:\n{result['response'][:500]}...")
    else:
        print(f"Error: {result.get('error')}")
