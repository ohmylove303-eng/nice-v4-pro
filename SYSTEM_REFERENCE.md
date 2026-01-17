# NICE v4 PRO - 시스템 참조 문서
## 최종 업데이트: 2026-01-10 23:09

---

## 📁 프로젝트 구조

```
자동화/
├── flask_app.py              # 메인 Flask 백엔드 (1674줄)
├── templates/
│   └── dashboard.html        # 프론트엔드 대시보드 (1930줄)
├── hybrid/
│   ├── protocol_gates.py     # Protocol Gates v2.6.1
│   ├── palantir_tracker.py   # Palantir Tracker AIP
│   ├── whale_analyzer.py     # 고래 분석기 (결정적)
│   ├── orchestrator.py       # HybridOrchestrator
│   └── crypto_data.py        # CoinGecko/Binance 데이터
└── nice_model/
    ├── classifier.py         # NICE 분류기
    ├── scorer.py             # 점수 계산
    └── kelly.py              # Kelly Criterion
```

---

## 🔵 백엔드 API 엔드포인트

### Core APIs

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/health` | GET | 헬스 체크 |
| `/api/crypto/analysis/<symbol>` | GET | 코인 AI 분석 (CoinGecko 실시간) |
| `/api/crypto/rankings` | GET | 코인 순위 (빗썸 기준) |
| `/api/nice/protocol-gates` | GET | Protocol Gates 상태 |
| `/api/nice/oco-orders/<symbol>` | GET | OCO 주문 계산 |
| `/api/nice/experts` | GET | 전문가 분석 |

### /api/crypto/analysis/<symbol> 응답

```json
{
  "symbol": "BTC",
  "name": "Bitcoin",
  "price": 98000.0,
  "change_24h": 2.5,
  "market_cap": 1900000000000,
  "circulation": 92.8,
  "circulating": 19500000,
  "total_supply": 21000000,
  "whale": "축적 중",
  "whale_wallets": 185,
  "whale_holding_pct": 38,
  "fractal": "Higher High",
  "fractal_strength": 85,
  "entry_price": 97510.0,
  "stop_loss": 95060.0,
  "take_profit": 103880.0,
  "nice_score": 78,
  "nice_type": "A",
  "source": "CoinGecko API",
  "timestamp": "2026-01-10T23:09:00"
}
```

---

## 🟠 프론트엔드 주요 함수

### 코인 검색 (빗썸 기준)

```javascript
// 빗썸 API로 코인 데이터 로드
async function loadCoinDatabase() {
    const res = await fetch('https://api.bithumb.com/public/ticker/ALL_KRW');
    // ... 거래량순 정렬, 한글명 매핑
}

// 한글 코인명 매핑
function getBithumbCoinName(symbol) {
    return names[symbol] || symbol;
}
```

### 코인 선택 시 업데이트

```javascript
async function selectCoin(symbol) {
    initTradingView(symbol);        // 차트
    updateWaveAnalysis();           // Elliott/Fib/추세선
    updateMarketStats();            // 시장 통계
    // AI 분석 패널 자동 갱신
}
```

### AI 리포트 새로고침

```javascript
async function refreshAIReport() {
    await loadAIReport();           // 레이어 차트
    await loadExpertAnalysis();     // 전문가 분석
}
```

---

## 📊 오늘 추가된 기능 (2026-01-10)

### 1. Elliott Wave / Fibonacci / 추세선 패널
- 코인 선택 시 자동 업데이트
- 가격 기반 Fib 레벨 계산

### 2. 주요 통계 패널 (TradingView 스타일)
- 거래량, 평균 볼륨(30)
- 시가총액, 출처
- 성과: 1W/1M/3M/6M/YTD/1Y 타일

### 3. 코인 검색 (빗썸 거래소 기준)
- 빗썸 API 우선, CoinGecko 폴백
- 50+ 코인 한글명 지원
- 거래량순 정렬

### 4. AI 리포트 새로고침 버튼
- 클릭 시 즉시 갱신
- 로딩 상태 + 완료 시간 표시

### 5. CoinGecko 실시간 가격 통합
- /api/crypto/analysis 엔드포인트 개선
- 모든 코인 실시간 가격 지원

---

## 🔧 배포 정보

| 항목 | 값 |
|------|-----|
| **플랫폼** | Render |
| **URL** | https://nice-v4-pro.onrender.com |
| **GitHub** | github.com/ohmylove303-eng/nice-v4-pro |
| **브랜치** | main |
| **자동 배포** | Git push → 자동 배포 |

---

## 📋 Genius Questions 검증 결과

| 질문 | 결과 |
|------|------|
| Q1: 기존 코드 보존? | ✅ 통과 |
| Q2: API 작동? | ✅ 5/5 통과 |
| Q3: UI 데이터 표시? | ✅ 통과 |
| Q4: 누락 파일? | error_detector.py (선택적) |
| Q5: 배포 준비? | ✅ Render 배포됨 |

---

## 📝 Git 커밋 히스토리 (오늘)

```
742cc15 - Add refresh button for AI Report
7ac4576 - Change coin search to Bithumb exchange
b8e366a - ADD: Market Statistics Panel (TradingView style)
96d2021 - Fix: Real-time CoinGecko data for ALL coins
315d87e - Expand coin search to support ALL coins
a94c9fa - Add coin search with autocomplete
```

---

**✅ 이 문서는 어디서든 참조 가능합니다.**
