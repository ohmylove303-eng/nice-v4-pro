#!/usr/bin/env python3
from flask import Flask, jsonify, render_template, request, redirect
from datetime import datetime
import json
import pandas as pd

app = Flask(__name__)


# ============================================================
# CRYPTO API ENDPOINTS
# ============================================================

@app.route('/api/crypto/market-gate')
def api_crypto_market_gate():
    """Market Gate 분석 API"""
    try:
        from crypto_market.market_gate import run_market_gate_sync
        
        result = run_market_gate_sync()
        
        # 지표별 시그널 분류
        indicators = []
        for name, val in result.metrics.items():
            signal = 'Neutral'
            if isinstance(val, (int, float)) and val is not None:
                if name == 'btc_ema200_slope_pct_20':
                    signal = 'Bullish' if val > 1 else ('Bearish' if val < -1 else 'Neutral')
                elif name == 'fear_greed_index':
                    signal = 'Bullish' if val > 50 else ('Bearish' if val < 30 else 'Neutral')
                elif name == 'funding_rate':
                    if val is not None:
                        signal = 'Bullish' if -0.0003 < val < 0.0005 else 'Bearish'
                elif name == 'alt_breadth_above_ema50':
                    if val is not None:
                        signal = 'Bullish' if val > 0.5 else ('Bearish' if val < 0.35 else 'Neutral')
            
            indicators.append({
                'name': name,
                'value': val,
                'signal': signal
            })
        
        return jsonify({
            'gate_color': result.gate,
            'score': result.score,
            'summary': f"BTC 시장 상태: {result.gate} (점수: {result.score}/100)",
            'indicators': indicators,
            'top_reasons': result.reasons,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/crypto/lead-lag')
def api_crypto_lead_lag():
    """Lead-Lag 분석 API"""
    try:
        from crypto_market.lead_lag.data_fetcher import fetch_all_data
        from crypto_market.lead_lag.granger import find_granger_causal_indicators
        
        # 데이터 수집
        df = fetch_all_data(start_date="2020-01-01", resample="monthly")
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 500
        
        # BTC MoM을 예측하는 선행 지표 찾기
        target = "BTC_MoM"
        if target not in df.columns:
            target = "BTC"
        
        results = find_granger_causal_indicators(df, target=target, max_lag=6)
        
        leading_indicators = []
        for r in results[:10]:
            # 상관관계 계산
            corr = df[r.cause].corr(df[target].shift(r.best_lag))
            
            leading_indicators.append({
                'variable': r.cause,
                'lag': r.best_lag,
                'p_value': r.best_p_value,
                'correlation': float(corr) if not pd.isna(corr) else 0,
                'interpretation': r.get_interpretation()
            })
        
        return jsonify({
            'target': target,
            'leading_indicators': leading_indicators,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/crypto/vcp-signals')
def api_crypto_vcp_signals():
    """VCP 시그널 목록 API"""
    try:
        from crypto_market.storage import make_engine, get_recent_signals
        
        engine = make_engine("crypto_market/signals.sqlite3")
        signals = get_recent_signals(engine, limit=50)
        
        return jsonify({
            'signals': signals,
            'count': len(signals),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/crypto/timeline')
def api_crypto_timeline():
    """타임라인 이벤트 API"""
    try:
        from pathlib import Path
        
        timeline_path = Path("crypto_market/timeline_events.json")
        if timeline_path.exists():
            with open(timeline_path) as f:
                events = json.load(f)
        else:
            events = []
        
        return jsonify({
            'events': events,
            'count': len(events),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# BACKTEST API
# ============================================================

@app.route('/api/crypto/backtest')
def api_crypto_backtest():
    """백테스트 실행 API"""
    try:
        import yfinance as yf
        import numpy as np
        from crypto_market.vcp_backtest import BacktestConfig, BacktestEngine
        from crypto_market.indicators import ema, atr
        
        # 1. Fetch BTC historical data
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period="2y")
        
        if hist.empty or len(hist) < 100:
            return jsonify({'error': 'Insufficient data'}), 500
        
        # 2. Create config
        config = BacktestConfig(
            initial_capital=100000.0,
            entry_trigger="BREAKOUT",
            stop_loss_type="FIXED_PCT",
            stop_loss_value=5.0,
            take_profit_pct=15.0,
            trailing_stop_pct=7.0,
            max_hold_bars=30,
            commission_pct=0.1,
            slippage_pct=0.05
        )
        
        # 3. Simulate simple breakout strategy
        engine = BacktestEngine(config)
        trades = []
        
        df = hist.reset_index()
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'splits']
        
        # Calculate indicators
        df['ema20'] = ema(df['close'], 20)
        df['ema50'] = ema(df['close'], 50)
        df['atr14'] = atr(df[['high', 'low', 'close']], 14)
        df['high_20'] = df['high'].rolling(20).max()
        
        # Simple backtest simulation
        position = None
        entry_price = 0
        entry_idx = 0
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            if position is None:
                # Entry: breakout above 20-day high with EMA alignment
                if row['close'] > df.iloc[i-1]['high_20'] and row['close'] > row['ema20'] > row['ema50']:
                    position = "LONG"
                    entry_price = float(row['close'])
                    entry_idx = i
            else:
                # Exit conditions
                bars_held = i - entry_idx
                pnl_pct = (row['close'] - entry_price) / entry_price * 100
                
                exit_reason = None
                if pnl_pct <= -config.stop_loss_value:
                    exit_reason = "STOP_LOSS"
                elif pnl_pct >= config.take_profit_pct:
                    exit_reason = "TAKE_PROFIT"
                elif bars_held >= config.max_hold_bars:
                    exit_reason = "MAX_HOLD"
                
                if exit_reason:
                    trades.append({
                        'entry_date': str(df.iloc[entry_idx]['date'])[:10],
                        'exit_date': str(row['date'])[:10],
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(float(row['close']), 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'bars_held': bars_held,
                        'exit_reason': exit_reason
                    })
                    position = None
        
        # 4. Calculate metrics
        if trades:
            winners = [t for t in trades if t['pnl_pct'] > 0]
            losers = [t for t in trades if t['pnl_pct'] <= 0]
            total_pnl = sum(t['pnl_pct'] for t in trades)
            
            metrics = {
                'total_trades': len(trades),
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': round(len(winners) / len(trades) * 100, 1),
                'avg_win': round(sum(t['pnl_pct'] for t in winners) / len(winners), 2) if winners else 0,
                'avg_loss': round(sum(t['pnl_pct'] for t in losers) / len(losers), 2) if losers else 0,
                'total_return': round(total_pnl, 2),
                'avg_bars_held': round(sum(t['bars_held'] for t in trades) / len(trades), 1)
            }
        else:
            metrics = {'total_trades': 0, 'win_rate': 0, 'total_return': 0}
        
        return jsonify({
            'config': {
                'initial_capital': config.initial_capital,
                'stop_loss': f"{config.stop_loss_value}%",
                'take_profit': f"{config.take_profit_pct}%",
                'trailing_stop': f"{config.trailing_stop_pct}%",
                'max_hold_bars': config.max_hold_bars
            },
            'metrics': metrics,
            'trades': trades[-20:],  # Last 20 trades
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================
# DASHBOARD ROUTE
# ============================================================

@app.route('/app')
def dashboard():
    """메인 대시보드"""
    return render_template('dashboard.html')


@app.route('/')
def index():
    """루트 → 대시보드로 리다이렉트"""
    return redirect('/app')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import os
    
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    port = int(os.environ.get('PORT', 5001))
    
    print(f"🚀 Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
