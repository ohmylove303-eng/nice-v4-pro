# NICE v4 PRO - CHANGELOG
## 변경 이력

---

## [2026-01-10] - 대규모 업데이트

### ✨ Added (추가)
- **Elliott Wave 분석 패널**: 파동 위치, 해석, 1-5 파동 표시
- **Fibonacci Retracement 패널**: 0%, 23.6%, 38.2%, 50%, 61.8%, 100% 레벨
- **추세선 분석 패널**: 저항/지지선, 추세 강도 %
- **주요 통계 패널**: 거래량, 평균 볼륨, 시가총액, 1W/1M/3M/6M/YTD/1Y 성과
- **코인 검색 기능**: 빗썸 거래소 기준 전 코인 검색
- **한글 코인명 지원**: 50+ 코인 (비트코인, 이더리움 등)
- **AI 리포트 새로고침 버튼**: 수동 갱신 기능

### 🔄 Changed (변경)
- `/api/crypto/analysis`: 하드코딩 → CoinGecko 실시간 API
- `loadCoinDatabase()`: CoinGecko → 빗썸 API 우선
- 거래 추천가: 실시간 가격 기반 계산

### 🐛 Fixed (수정)
- 기타 코인 AI 분석 $0 표시 문제
- `marketMapData` const → let 오류
- 시장 지도 데이터 로딩 실패

---

## [2026-01-09] - 초기 배포

### ✨ Added
- Protocol Gates v2.6.1
- Palantir Tracker AIP
- HybridOrchestrator 통합
- TradingView 차트 통합
- 5-Layer NICE 분석
- OCO 주문 계산

---

## 파일별 변경

### flask_app.py
- `/api/crypto/analysis/<symbol>`: CoinGecko 실시간 통합
- 50+ 코인 ID 매핑 추가

### templates/dashboard.html
- Elliott/Fib/추세선 패널 HTML
- `updateWaveAnalysis()` 함수
- `updateMarketStats()` 함수
- `loadCoinDatabase()` 빗썸 API
- `getBithumbCoinName()` 한글 매핑
- `refreshAIReport()` 새로고침 버튼
