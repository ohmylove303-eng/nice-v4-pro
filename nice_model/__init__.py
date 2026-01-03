"""
NICE Model Package
==================
NICE v4 5-Layer Scoring System
- 원본 NICE 모델 로직 100% 보존
- 외부 시스템과 통합 시 데이터 입력만 받음

Components:
- NICEScorer: 5레이어 점수 계산
- NICEClassifier: Type A/B/C 분류
- KellyCalculator: Kelly % 계산
- NICEDataCollector: 데이터 수집
- CoinNICEAnalyzer: 코인별 분석
- NICEMarketAnalyzer: 시장 분석
- NICEAIAnalyzer: LLM 기반 AI 분석 (Gemini/GPT)
"""

from .scorer import NICEScorer, NICEScore, LayerScore
from .classifier import NICEClassifier, NICESignal, SignalType
from .kelly import KellyCalculator, KellyResult
from .data_collector import NICEDataCollector, NICEData
from .coin_analyzer import CoinNICEAnalyzer, CoinNICEResult, NICEMarketAnalyzer
from .ai_analyzer import NICEAIAnalyzer

__all__ = [
    # Core
    'NICEScorer', 'NICEScore', 'LayerScore',
    'NICEClassifier', 'NICESignal', 'SignalType',
    'KellyCalculator', 'KellyResult',
    # Data
    'NICEDataCollector', 'NICEData',
    # Analyzers
    'CoinNICEAnalyzer', 'CoinNICEResult', 'NICEMarketAnalyzer',
    # AI
    'NICEAIAnalyzer'
]
