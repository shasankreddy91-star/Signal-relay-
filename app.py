# app.py
# pip install fastapi uvicorn[standard] requests pandas ta python-dotenv
import os, time, threading, requests, pandas as pd
from fastapi import FastAPI
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TWELVE_API_KEY")
SYMBOL = "AUD/USD"          # If this fails, try "AUDUSD"
INTERVAL = "1min"
OUTPUT_SIZE = 120           # last 120 minutes
POLL_SECONDS = 5            # refresh cadence

app = FastAPI()
latest = {"signal": "WAIT", "reason": "warming_up"}

def fetch_candles():
    url = (
        "https://api.twelvedata.com/time_series"
        f"?apikey={API_KEY}&symbol={SYMBOL}"
        f"&interval={INTERVAL}&outputsize={OUTPUT_SIZE}&format=JSON"
    )
    j = requests.get(url, timeout=10).json()
    if "values" not in j:
        raise RuntimeError(f"Bad response: {j}")
    df = pd.DataFrame(j["values"])
    # Ensure types and order
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("datetime").reset_index(drop=True)
    return df

def compute_signal(df):
    if len(df) < 25:
        return {"signal":"WAIT","reason":"need at least 25 bars"}
    close = df["close"]
    rsi3  = RSIIndicator(close, window=3).rsi()
    ema9  = EMAIndicator(close, window=9).ema_indicator()
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    bb    = BollingerBands(close, window=20, window_dev=2)
    up, low = bb.bollinger_hband(), bb.bollinger_lband()

    i = -1
    rsi_now, rsi_prev = float(rsi3.iloc[i]), float(rsi3.iloc[i-1])
    c_now = float(close.iloc[i])
    e9, e20 = float(ema9.iloc[i]), float(ema20.iloc[i])
    upb, lowb = float(up.iloc[i]), float(low.iloc[i])
    e20_slope = float(ema20.iloc[i] - ema20.iloc[i-1])

    call = (rsi_prev < 20 <= rsi_now) and (c_now <= lowb * 1.001) and (c_now > e9) and (e20_slope > 0)
    put  = (rsi_prev > 80 >= rsi_now) and (c_now >= upb * 0.999) and (c_now < e9) and (e20_slope < 0)

    if call:
        sig, reason = "CALL", "RSI upcross+lower band bounce+EMA9 reclaim+EMA20 up"
    elif put:
        sig, reason = "PUT",  "RSI downcross+upper band fade+EMA9 loss+EMA20 down"
    else:
        sig, reason = "WAIT", "no setup"

    return {
        "timestamp": df["datetime"].iloc[i].isoformat(),
        "close": c_now,
        "rsi3": rsi_now,
        "ema9": e9,
        "ema20": e20,
        "bb_upper": upb,
        "bb_lower": lowb,
        "ema20_slope": e20_slope,
        "signal": sig,
        "reason": reason
    }

def loop():
    global latest
    while True:
        try:
            df = fetch_candles()
            latest = compute_signal(df)
        except Exception as e:
            latest = {"signal":"WAIT","error":str(e)}
        time.sleep(POLL_SECONDS)

threading.Thread(target=loop, daemon=True).start()

@app.get(”/signal”)
def get_signal():
    # Minimal payload the bot needs
    return {"signal": latest.get("signal","WAIT"), "confidence": confidence(latest), "meta": latest}

@app.post("/refresh")
def force_refresh():
    # optional: trigger immediate fetch
    try:
        df = fetch_candles()
        res = compute_signal(df)
        global latest
        latest = res
        return {"ok": True, "meta": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def confidence(state):
    # Simple heuristic: higher when EMA trend and RSI cross align near bands
    if state.get("signal") in ("CALL","PUT"):
        slope = abs(state.get("ema20_slope",0))
        near_band = min(
            abs(state["close"]-state["bb_lower"])/max(1e-6,state["bb_lower"]),
            abs(state["close"]-state["bb_upper"])/max(1e-6,state["bb_upper"])
        )
        base = 60 + min(20, slope*10000) - min(20, near_band*10000)
        return round(max(50, min(90, base)), 1)
    return 0.0
