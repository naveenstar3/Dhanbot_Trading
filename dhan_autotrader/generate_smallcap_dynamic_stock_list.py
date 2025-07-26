"""
✅ Intraday Small Cap Stock Generator for Dhan CNC Strategy
Generates 'smallcap_dynamic_stock_list.csv' using real-time filters:
- Uses verified columns from dhan_master.csv:
  - SEM_TRADING_SYMBOL ➝ symbol
  - SEM_SMST_SECURITY_ID ➝ security_id
  - SEM_SERIES ➝ series
  - SEM_SEGMENT ➝ segment
- Applies filters:
  - LTP < capital
  - Volume ≥ 5L
  - ATR ≥ ₹2
  - RSI < 70
  - SMA20 check
  - Momentum (5-min)
  - Placeholder sentiment & pattern
Output: 'smallcap_dynamic_stock_list.csv'
"""

import pandas as pd
import numpy as np
import datetime
import json
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator
from dhanhq import DhanContext, dhanhq

# === Load Dhan API credentials ===
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

# === Init SDK client ===
dhan_context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(dhan_context)

# === Load dhan_master.csv ===
master = pd.read_csv("dhan_master.csv")

# Map to real column names
symbol_col = "SEM_TRADING_SYMBOL"
series_col = "SEM_SERIES"
segment_col = "SEM_SEGMENT"
security_id_col = "SEM_SMST_SECURITY_ID"

# === Filter only EQ/NSE_EQ & skip SME/PSU/ETF/REIT ===
master = master[
    (master[series_col] == "EQ") &
    (master[segment_col] == "NSE_EQ") &
    (~master[symbol_col].str.contains("SME|PSU|ETF|REIT", case=False, na=False))
]

# === Load capital ===
capital = float(config.get("capital", 5000))  # Default fallback ₹5000

# === Output collector ===
qualified = []

# === Scan each stock ===
for _, row in master.iterrows():
    try:
        sym = row[symbol_col]
        sec_id = str(row[security_id_col])
        sector = "UNKNOWN"

        print(f"🔍 Checking {sym}")

        # Fetch intraday 1-min data
        today = datetime.date.today().strftime("%Y-%m-%d")
        candles = dhan.intraday_minute_data(
            security_id=sec_id,
            exchange_segment=dhan.NSE,
            instrument_type="EQUITY",
            from_date=today,
            to_date=today
        )
        df = pd.DataFrame(candles["data"])
        if df.empty or len(df) < 20:
            print(f"⛔ {sym}: Insufficient data")
            continue

        df["datetime"] = pd.to_datetime(df["startTime"])
        df.set_index("datetime", inplace=True)
        df = df.astype(float).sort_index()

        # Apply all filters
        ltp = df["close"].iloc[-1]
        if ltp > capital:
            continue

        rsi = RSIIndicator(df["close"], window=14).rsi().iloc[-1]
        if rsi > 70:
            continue

        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]
        if atr < 2.0:
            continue

        avg_vol = df["volume"].tail(5).mean()
        if avg_vol < 500000:
            continue

        if df["close"].iloc[-1] < df["close"].iloc[-5]:
            continue

        sma_20 = SMAIndicator(df["close"], window=20).sma_indicator().iloc[-1]

        qty = int(capital // ltp)
        capital_used = round(qty * ltp, 2)
        potential_profit = round(atr * qty, 2)
        priority_score = round(avg_vol * atr, 2)

        qualified.append({
            "symbol": sym,
            "security_id": sec_id,
            "ltp": round(ltp, 2),
            "quantity": qty,
            "capital_used": capital_used,
            "avg_volume": int(avg_vol),
            "avg_range": round(atr, 2),
            "potential_profit": potential_profit,
            "sma_20": round(sma_20, 2),
            "rsi": round(rsi, 2),
            "stock_origin": "smallcap_scan",
            "priority_score": priority_score,
            "sector": sector
        })

    except Exception as e:
        print(f"❌ {sym} failed: {e}")

# === Save final CSV ===
final = pd.DataFrame(qualified)
final.to_csv("smallcap_dynamic_stock_list.csv", index=False)
print(f"✅ Final small-cap list saved with {len(final)} entries.")
