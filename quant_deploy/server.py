"""
A股量化系统 - 后端服务
依赖安装: pip install flask flask-cors akshare pandas numpy
启动方式: python server.py
访问地址: http://127.0.0.1:5000
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import requests

app = Flask(__name__, static_folder="static")
CORS(app)
# ── 工具函数 ────────────────────────────────────────────
def calc_ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean().round(3)

def calc_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif.round(3), dea.round(3), hist.round(3)

def calc_bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    mid = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = (mid + k * std).round(3)
    lower = (mid - k * std).round(3)
    return upper, mid.round(3), lower

def ma_signal(closes: pd.Series) -> dict:
    ma5  = calc_ma(closes, 5)
    ma20 = calc_ma(closes, 20)
    if ma5.iloc[-1] is None or ma20.iloc[-1] is None:
        return {"signal": "关注", "ma_status": "数据不足"}
    prev5, prev20 = ma5.iloc[-2], ma20.iloc[-2]
    last5, last20 = ma5.iloc[-1], ma20.iloc[-1]
    if pd.isna(prev5) or pd.isna(prev20):
        return {"signal": "关注", "ma_status": "计算中"}
    if prev5 < prev20 and last5 > last20:
        return {"signal": "买入", "ma_status": "金叉↑", "strength": 4}
    if prev5 > prev20 and last5 < last20:
        return {"signal": "卖出", "ma_status": "死叉↓", "strength": 4}
    if last5 > last20:
        return {"signal": "持有", "ma_status": "多头排列", "strength": 3}
    return {"signal": "关注", "ma_status": "空头排列", "strength": 2}

def _calc_backtest_stats(equity: list, trades: list) -> dict:
    sell_trades = [t for t in trades if t["type"] == "sell"]
    if not sell_trades:
        return {"total_return": 0, "win_rate": 0, "trade_count": 0, "sharpe": 0, "max_drawdown": 0}
    win_trades = [t for t in sell_trades if t.get("ret", 0) > 0]
    total_return = (equity[-1] - 1) * 100
    eq = pd.Series(equity)
    roll_max = eq.cummax()
    drawdown = ((eq - roll_max) / roll_max).min() * 100
    returns = eq.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
    return {
        "total_return": round(total_return, 2),
        "win_rate": round(len(win_trades) / len(sell_trades) * 100, 1),
        "trade_count": len(sell_trades),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(drawdown, 2),
    }

def backtest_ma_strategy(df: pd.DataFrame) -> dict:
    closes = df["收盘"].astype(float)
    ma5  = calc_ma(closes, 5)
    ma20 = calc_ma(closes, 20)
    position = 0
    buy_price = 0
    trades = []
    equity = [1.0]
    for i in range(1, len(closes)):
        if pd.isna(ma5.iloc[i]) or pd.isna(ma20.iloc[i]):
            equity.append(equity[-1])
            continue
        if ma5.iloc[i-1] < ma20.iloc[i-1] and ma5.iloc[i] > ma20.iloc[i] and position == 0:
            position = 1
            buy_price = closes.iloc[i]
            trades.append({"type": "buy", "price": buy_price, "date": df["日期"].iloc[i]})
        elif ma5.iloc[i-1] > ma20.iloc[i-1] and ma5.iloc[i] < ma20.iloc[i] and position == 1:
            sell_price = closes.iloc[i]
            ret = (sell_price - buy_price) / buy_price
            trades.append({"type": "sell", "price": sell_price, "date": df["日期"].iloc[i], "ret": round(ret, 4)})
            equity.append(equity[-1] * (1 + ret))
            position = 0
            continue
        equity.append(equity[-1] * (1 + (closes.iloc[i] - closes.iloc[i-1]) / closes.iloc[i-1]) if position == 1 else equity[-1])
    return _calc_backtest_stats(equity, trades)

def backtest_boll_strategy(df: pd.DataFrame) -> dict:
    closes = df["收盘"].astype(float)
    boll_u, boll_m, boll_l = calc_bollinger(closes)
    position = 0
    buy_price = 0
    trades = []
    equity = [1.0]
    for i in range(1, len(closes)):
        if pd.isna(boll_u.iloc[i]) or pd.isna(boll_l.iloc[i]):
            equity.append(equity[-1])
            continue
        if closes.iloc[i] <= boll_l.iloc[i] and position == 0:
            position = 1
            buy_price = closes.iloc[i]
            trades.append({"type": "buy", "price": buy_price, "date": df["日期"].iloc[i]})
        elif closes.iloc[i] >= boll_u.iloc[i] and position == 1:
            sell_price = closes.iloc[i]
            ret = (sell_price - buy_price) / buy_price
            trades.append({"type": "sell", "price": sell_price, "date": df["日期"].iloc[i], "ret": round(ret, 4)})
            equity.append(equity[-1] * (1 + ret))
            position = 0
            continue
        equity.append(equity[-1] * (1 + (closes.iloc[i] - closes.iloc[i-1]) / closes.iloc[i-1]) if position == 1 else equity[-1])
    return _calc_backtest_stats(equity, trades)

def _fetch_kdata(symbol: str, days: int) -> pd.DataFrame:
    prefix = "sh" if symbol.startswith("6") else "sz"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},day,,,{days},qfq"
    r = requests.get(url, timeout=8)
    raw = r.json()
    key = f"{prefix}{symbol}"
    kdata = raw["data"][key].get("qfqday", raw["data"][key].get("day", []))
    dates = [row[0] for row in kdata]
    closes = [float(row[2]) for row in kdata]
    return pd.DataFrame({"日期": dates, "收盘": closes})

# ── API 路由 ────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/kline")
def api_kline():
    symbol = request.args.get("symbol", "300418")
    period = request.args.get("period", "daily")
    days = int(request.args.get("days", 130))
    adjust = request.args.get("adjust", "qfq")
    try:
        prefix = "sh" if symbol.startswith("6") else "sz"
        period_map = {"daily": "day", "weekly": "week"}
        p = period_map.get(period, "day")
        adj = adjust if adjust in ("qfq", "hfq", "") else "qfq"
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix}{symbol},{p},,,{days},{adj}"
        r = requests.get(url, timeout=8)
        raw = r.json()
        key = f"{prefix}{symbol}"
        kdata = raw["data"][key].get(f"{adj}{p}", raw["data"][key].get(p, []))
        dates, opens, closes, highs, lows, volumes = [], [], [], [], [], []
        for row in kdata[-days:]:
            dates.append(row[0])
            opens.append(float(row[1]))
            closes.append(float(row[2]))
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            volumes.append(float(row[5]))
        closes_s = pd.Series(closes)
        ma5_s = calc_ma(closes_s, 5)
        ma20_s = calc_ma(closes_s, 20)
        ma60_s = calc_ma(closes_s, 60)
        rsi_s = calc_rsi(closes_s)
        dif, dea, hist = calc_macd(closes_s)
        boll_u, boll_m, boll_l = calc_bollinger(closes_s)
        sig = ma_signal(closes_s)
        def safe_list(s):
            return [None if pd.isna(v) else v for v in s.tolist()]
        chg_pct = [0.0] + [round((closes[i]-closes[i-1])/closes[i-1]*100, 2) for i in range(1, len(closes))]
        result = {
            "symbol": symbol,
            "name": STOCK_POOL.get(symbol, symbol),
            "dates": dates,
            "open": [round(v, 3) for v in opens],
            "close": [round(v, 3) for v in closes],
            "high": [round(v, 3) for v in highs],
            "low": [round(v, 3) for v in lows],
            "volume": volumes,
            "chg_pct": chg_pct,
            "ma5": safe_list(ma5_s),
            "ma20": safe_list(ma20_s),
            "ma60": safe_list(ma60_s),
            "rsi": safe_list(rsi_s),
            "macd_dif": safe_list(dif),
            "macd_dea": safe_list(dea),
            "macd_hist": safe_list(hist),
            "boll_upper": safe_list(boll_u),
            "boll_mid": safe_list(boll_m),
            "boll_lower": safe_list(boll_l),
            "signal": sig,
            "latest": {
                "close": closes[-1],
                "chg_pct": chg_pct[-1],
                "high": highs[-1],
                "low": lows[-1],
                "date": dates[-1],
            },
        }
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/backtest")
def api_backtest():
    symbol = request.args.get("symbol", "300418")
    days = int(request.args.get("days", 500))
    strategy = request.args.get("strategy", "ma")
    try:
        df = _fetch_kdata(symbol, days)
        if strategy == "boll":
            result = backtest_boll_strategy(df)
        else:
            result = backtest_ma_strategy(df)
        result["symbol"] = symbol
        result["name"] = STOCK_POOL.get(symbol, symbol)
        result["strategy"] = strategy
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/realtime")
def api_realtime():
    try:
        symbols = list(STOCK_POOL.keys())
        codes = ",".join([("sh" if s.startswith("6") else "sz") + s for s in symbols])
        url = f"https://qt.gtimg.cn/q={codes}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        r.encoding = "gbk"
        records = []
        for line in r.text.strip().split("\n"):
            if "=" not in line:
                continue
            data = line.split("~")
            if len(data) < 32:
                continue
            symbol = data[2]
            records.append({
                "symbol":  symbol,
                "name":    data[1],
                "price":   round(float(data[3]), 2),
                "chg_pct": round(float(data[32]), 2),
                "chg_amt": round(float(data[31]), 2),
                "volume":  float(data[6]),
                "amount":  float(data[37]) if len(data) > 37 else 0,
                "high":    round(float(data[33]), 2),
                "low":     round(float(data[34]), 2),
            })
        return jsonify({"ok": True, "data": records, "ts": datetime.now().strftime("%H:%M:%S")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/signals")
def api_signals():
    results = []
    for symbol, name in STOCK_POOL.items():
        try:
            prefix = "sh" if symbol.startswith("6") else "sz"
            url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?
