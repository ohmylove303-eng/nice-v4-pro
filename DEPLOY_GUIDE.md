# NICE v4 PRO 서버 배포 가이드

## 📁 프로젝트 구조

```
자동화/
├── flask_app.py              # 메인 Flask 서버
├── requirements.txt          # Python 패키지
├── .env                      # 환경변수 (API 키)
├── nice_model/               # NICE 분석 모듈
│   ├── __init__.py
│   ├── scorer.py
│   ├── classifier.py
│   ├── kelly.py
│   ├── data_collector.py
│   ├── api_providers.py
│   ├── coin_analyzer.py
│   ├── ai_analyzer.py
│   └── .env                  # NICE 전용 API 키
├── hybrid/                   # 하이브리드 분석 모듈
├── templates/                # HTML 템플릿
│   └── dashboard.html
└── static/                   # 정적 파일
```

---

## 1️⃣ 서버 요구사항

```bash
# Ubuntu 22.04 LTS 권장
# Python 3.10+
# 최소 2GB RAM, 2 CPU
```

---

## 2️⃣ 설치

### 2.1 패키지 설치

```bash
# 프로젝트 디렉토리로 이동
cd /path/to/자동화

# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2.2 환경변수 설정

```bash
# .env 파일 생성
cat > .env << 'EOF'
# Flask 설정
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=your-secret-key-here

# API 키
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
FRED_API_KEY=your_fred_api_key
EOF
```

### 2.3 nice_model/.env 설정

```bash
cat > nice_model/.env << 'EOF'
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GPT_API_KEY=your_openai_api_key
FRED_API_KEY=your_fred_api_key
EOF
```

---

## 3️⃣ 서버 실행 방법

### 개발 모드 (테스트용)

```bash
python flask_app.py
# http://localhost:5003/app
```

### 프로덕션 모드 (Gunicorn)

```bash
# Gunicorn 설치
pip install gunicorn

# 실행 (4 워커, 포트 5003)
gunicorn -w 4 -b 0.0.0.0:5003 flask_app:app
```

---

## 4️⃣ Systemd 서비스 등록 (자동 시작)

```bash
sudo nano /etc/systemd/system/nice-pro.service
```

```ini
[Unit]
Description=NICE v4 PRO Trading System
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/자동화
Environment="PATH=/home/ubuntu/자동화/.venv/bin"
ExecStart=/home/ubuntu/자동화/.venv/bin/gunicorn -w 4 -b 0.0.0.0:5003 flask_app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable nice-pro
sudo systemctl start nice-pro

# 상태 확인
sudo systemctl status nice-pro
```

---

## 5️⃣ Nginx 리버스 프록시 (선택)

```bash
sudo nano /etc/nginx/sites-available/nice-pro
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5003;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/nice-pro /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 6️⃣ SSL 인증서 (HTTPS)

```bash
# Certbot 설치
sudo apt install certbot python3-certbot-nginx

# SSL 발급
sudo certbot --nginx -d your-domain.com
```

---

## 7️⃣ 방화벽 설정

```bash
# UFW 사용 시
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 5003/tcp  # 직접 접속 허용 시
```

---

## 8️⃣ 모니터링

### 로그 확인

```bash
# 서비스 로그
sudo journalctl -u nice-pro -f

# Flask 로그
tail -f /home/ubuntu/자동화/flask.log
```

### 헬스체크 API

```bash
curl http://your-server:5003/api/nice/ai/status
```

---

## 9️⃣ 빠른 배포 스크립트

```bash
#!/bin/bash
# deploy.sh

echo "📦 NICE v4 PRO 배포 시작..."

# 가상환경 활성화
source .venv/bin/activate

# 패키지 업데이트
pip install -r requirements.txt

# 서비스 재시작
sudo systemctl restart nice-pro

echo "✅ 배포 완료!"
echo "🌐 http://your-server:5003/app"
```

---

## 🔧 주요 API 엔드포인트

| 엔드포인트 | 설명 |
|------------|------|
| `/app` | 대시보드 메인 |
| `/api/crypto/rankings` | 코인 순위 |
| `/api/nice/coin/<symbol>` | 코인별 NICE 분석 |
| `/api/nice/market` | 시장 전체 분석 |
| `/api/nice/ai/analyze` | AI 분석 (Gemini/GPT) |
| `/api/nice/ai/status` | AI 사용량 상태 |
| `/api/nice/experts` | 전문가 관점 |

---

## ⚠️ 주의사항

1. **API 키 보안**: `.env` 파일을 Git에 커밋하지 마세요
2. **GPT 사용 제한**: 하루 2번 (09:00, 21:00 KST)만 사용
3. **Gemini**: 제한 없이 사용 가능
4. **FRED API**: 매크로 데이터용 (선택)
