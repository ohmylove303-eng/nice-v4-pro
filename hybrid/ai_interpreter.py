"""
AI Interpreter (Gemini)
========================
NICE 분석 결과를 초등학교 3학년 수준으로 쉽게 설명

사용법:
>>> interpreter = AIInterpreter()
>>> result = interpreter.explain_score(score_data)
>>> print(result)  # "🟢 지금 사도 돼요! 점수가 80점이에요..."
"""

import os
import json
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class AIExplanation:
    """AI 설명 결과"""
    summary: str          # 한 줄 요약
    detail: str           # 상세 설명
    recommendation: str   # 추천 행동
    emoji: str            # 대표 이모지
    color: str            # green/yellow/red
    
    def to_dict(self) -> Dict:
        return {
            'summary': self.summary,
            'detail': self.detail,
            'recommendation': self.recommendation,
            'emoji': self.emoji,
            'color': self.color
        }


class AIInterpreter:
    """
    Gemini AI를 사용한 쉬운 한글 해석기
    
    사용법:
    >>> interpreter = AIInterpreter()
    >>> explanation = interpreter.explain_nice_result(nice_result)
    """
    
    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_API_KEY', '')
    
    def explain_nice_result(self, nice_result: Dict) -> AIExplanation:
        """
        NICE 분석 결과를 쉬운 한글로 설명
        
        Args:
            nice_result: HybridOrchestrator.run().to_dict() 결과
            
        Returns:
            AIExplanation: 쉬운 설명
        """
        score = nice_result.get('score', 0)
        signal_type = nice_result.get('signal', {}).get('type', 'C')
        layers = nice_result.get('layers', [])
        
        # Gemini API 호출 시도
        if self.api_key:
            try:
                return self._call_gemini(nice_result)
            except Exception as e:
                print(f"Gemini API error: {e}")
        
        # API 없으면 규칙 기반 설명 생성
        return self._generate_rule_based(score, signal_type, layers)
    
    def _call_gemini(self, nice_result: Dict) -> AIExplanation:
        """Gemini API 호출"""
        import urllib.request
        
        prompt = f"""
당신은 초등학교 3학년 어린이에게 투자를 설명하는 친절한 선생님입니다.
아래 분석 결과를 아주 쉽고 간단하게 설명해주세요.

분석 결과:
- 종합 점수: {nice_result.get('score', 0)}점 (100점 만점)
- 신호: Type {nice_result.get('signal', {}).get('type', 'C')}
- 권장 행동: {nice_result.get('signal', {}).get('action', '')}

레이어별 점수:
{json.dumps(nice_result.get('layers', []), ensure_ascii=False, indent=2)}

다음 형식으로 답변해주세요:
1. 한 줄 요약 (20자 이내, 이모지 포함)
2. 왜 그런지 설명 (50자 이내)
3. 지금 뭘 해야 하는지 (30자 이내)
"""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 300}
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode())
        
        text = result['candidates'][0]['content']['parts'][0]['text']
        
        # AI 응답 파싱 (간단히)
        lines = text.strip().split('\n')
        summary = lines[0] if lines else "분석 중..."
        detail = lines[1] if len(lines) > 1 else ""
        recommendation = lines[2] if len(lines) > 2 else ""
        
        signal_type = nice_result.get('signal', {}).get('type', 'C')
        emoji = '🟢' if signal_type == 'A' else ('🟡' if signal_type == 'B' else '🔴')
        color = 'green' if signal_type == 'A' else ('yellow' if signal_type == 'B' else 'red')
        
        return AIExplanation(
            summary=summary,
            detail=detail,
            recommendation=recommendation,
            emoji=emoji,
            color=color
        )
    
    def _generate_rule_based(self, score: float, signal_type: str, layers: list) -> AIExplanation:
        """규칙 기반 설명 생성 (API 없을 때)"""
        
        if signal_type == 'A':
            summary = f"🟢 지금 사도 돼요! ({score:.0f}점)"
            detail = "차트도 좋고, 큰손 아저씨들도 사고 있어요!"
            recommendation = "용돈의 4%만 조금 사보세요"
            emoji = '🟢'
            color = 'green'
            
        elif signal_type == 'B':
            summary = f"🟡 조금 더 기다려요 ({score:.0f}점)"
            detail = "아직 완벽하지 않아요. 더 지켜보는 게 좋아요."
            recommendation = "사고 싶으면 아주 조금만 (2%)"
            emoji = '🟡'
            color = 'yellow'
            
        else:  # Type C
            summary = f"🔴 지금은 안 돼요 ({score:.0f}점)"
            detail = "지금 사면 돈을 잃을 수 있어요. 위험해요!"
            recommendation = "그냥 구경만 하세요"
            emoji = '🔴'
            color = 'red'
        
        # 레이어 분석 추가 - Dict 또는 List 형식 지원
        if layers:
            # Dict 형식 처리 (key: name, value: {score, max})
            if isinstance(layers, dict):
                layer_list = []
                for name, data in layers.items():
                    if isinstance(data, dict):
                        score = data.get('score', 0)
                        max_score = data.get('max', 30)
                        pct = (score / max_score) * 100 if max_score > 0 else 0
                        layer_list.append({'name': name, 'percentage': pct})
                layers = layer_list
            
            if layers:  # list로 변환 후 확인
                best_layer = max(layers, key=lambda x: x.get('percentage', 0) if isinstance(x, dict) else 0)
                
                layer_names_ko = {
                    'technical': '차트',
                    'onchain': '블록체인',
                    'sentiment': '사람들 기분',
                    'macro': '나라 경제',
                    'etf': '큰손 아저씨들'
                }
                
                if isinstance(best_layer, dict):
                    best_name = layer_names_ko.get(best_layer.get('name', ''), '분석')
                    best_pct = best_layer.get('percentage', 0)
                    detail += f" {best_name}이 가장 좋아요! ({best_pct:.0f}점)"
        
        return AIExplanation(
            summary=summary,
            detail=detail,
            recommendation=recommendation,
            emoji=emoji,
            color=color
        )
    
    def explain_for_kids(self, score: float, signal_type: str) -> str:
        """아이들을 위한 초간단 설명"""
        if signal_type == 'A':
            return f"""
🟢 **지금 사도 돼요!**

점수가 {score:.0f}점이에요. 
75점이 넘으니까 사도 괜찮아요!

💰 용돈에서 조금만 쓰세요 (4%)
예) 용돈이 10,000원이면 400원만!

🛑 만약 떨어지면 바로 팔아요 (-2%)
🎉 많이 오르면 여기서 팔아요 (+4%)
"""
        elif signal_type == 'B':
            return f"""
🟡 **조금만 더 기다려요!**

점수가 {score:.0f}점이에요.
75점이 안 되니까 조심해야 해요.

⏰ 점수가 75점 넘을 때까지 기다려요
📊 매일 점수를 확인해보세요
"""
        else:
            return f"""
🔴 **지금은 안 돼요!**

점수가 {score:.0f}점이에요.
너무 낮아서 위험해요!

❌ 지금 사면 돈을 잃을 수 있어요
👀 그냥 구경만 하세요
"""


# 테스트용
if __name__ == '__main__':
    interpreter = AIInterpreter()
    
    # 테스트 데이터
    test_result = {
        'score': 63.3,
        'signal': {'type': 'B', 'action': '신중히 거래'},
        'layers': [
            {'name': 'technical', 'percentage': 40},
            {'name': 'onchain', 'percentage': 83.3},
            {'name': 'sentiment', 'percentage': 56.7},
            {'name': 'macro', 'percentage': 70},
            {'name': 'etf', 'percentage': 66.7}
        ]
    }
    
    explanation = interpreter.explain_nice_result(test_result)
    print("=== AI Explanation Test ===\n")
    print(f"Summary: {explanation.summary}")
    print(f"Detail: {explanation.detail}")
    print(f"Recommendation: {explanation.recommendation}")
    print()
    print("=== Kids Explanation ===")
    print(interpreter.explain_for_kids(63.3, 'B'))
