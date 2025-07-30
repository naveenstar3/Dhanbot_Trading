"""
✅ Intraday Small Cap Stock Generator for Dhan CNC Strategy
Super-debug version: Pinpoints why all stocks are excluded.
"""

import pandas as pd
import numpy as np
import datetime
import json
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator
from dhanhq import DhanContext, dhanhq
import time

# === Load Dhan API credentials ===
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

# === Init SDK client ===
dhan_context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(dhan_context)

# === Load dhan_master.csv and print diagnostics ===
df_raw = pd.read_csv("dhan_master.csv")
print(f"Loaded {len(df_raw)} total rows from dhan_master.csv")
print("Raw columns:", df_raw.columns.tolist())
print("SEM_SERIES unique:", df_raw['SEM_SERIES'].unique())
print("SEM_SEGMENT unique:", df_raw['SEM_SEGMENT'].unique())
print("Sample symbols:", df_raw['SEM_TRADING_SYMBOL'].head().tolist())

# Map to real column names
symbol_col = "SEM_TRADING_SYMBOL"
series_col = "SEM_SERIES"
segment_col = "SEM_SEGMENT"
security_id_col = "SEM_SMST_SECURITY_ID"

# === Check for common data issues (whitespace, case) ===
df_raw[series_col] = df_raw[series_col].astype(str).str.strip().str.upper()
df_raw[segment_col] = df_raw[segment_col].astype(str).str.strip().str.upper()

# === Print value counts for filter columns ===
print("Value counts for SEM_SERIES:\n", df_raw[series_col].value_counts())
print("Value counts for SEM_SEGMENT:\n", df_raw[segment_col].value_counts())

# === Apply initial smallcap filters (can adjust below as needed) ===
master = df_raw[
    (df_raw[series_col] == "EQ") &
    (df_raw[segment_col] == "E") &
    (~df_raw[symbol_col].str.contains("SME|PSU|ETF|REIT", case=False, na=False))
]

print(f"Rows after EQ/NSE_EQ/smallcap filter: {len(master)}")

# === Load capital ===
capital = float(config.get("capital", 5000))  # Default fallback ₹5000
print(f"🟢 Scan will use capital = ₹{capital:,.2f}")

# === Output collector ===
qualified = []

if len(master) == 0:
    print("⚠️ No stocks left after filter! Check filters or dhan_master.csv content.")
else:
    # === Scan each stock ===
    total_stocks = len(master)
    for i, (idx, row) in enumerate(master.iterrows(), start=1):
        try:
            sym = row[symbol_col]
            sec_id = str(row[security_id_col])
            sector = "UNKNOWN"

            print(f"\n🔍 Checking {sym} (Security ID: {sec_id}) [{i}/{total_stocks}]")
    
            # Respect Dhan API limit (1 request per second)
            # time.sleep(1.2)
    
            # Fetch intraday 1-min data
            today = datetime.date.today().strftime("%Y-%m-%d")
            try:
                candles = dhan.intraday_minute_data(
                    security_id=sec_id,
                    exchange_segment=dhan.NSE,
                    instrument_type="EQUITY",
                    from_date=today,
                    to_date=today
                )
                # Robust check: candles must be a dict with "data" and at least 1 row
                if not candles or "data" not in candles or not candles["data"]:
                    print(f"⛔ {sym}: No intraday data returned or API throttled (empty or missing 'data')")
                    continue
                df = pd.DataFrame(candles["data"])
                if df.empty:
                    print(f"⛔ {sym}: Empty DataFrame after fetch")
                    continue
            except Exception as e:
                print(f"⛔ {sym}: Failed to fetch intraday data: {e}")
                continue
    

            if df.empty or len(df) < 20:
                print(f"⛔ {sym}: Insufficient data ({len(df)} rows)")
                continue

            # Find the actual time column (Dhan often returns "timestamp", not "startTime")
            if "startTime" in df.columns:
                time_col = "startTime"
            elif "timestamp" in df.columns:
                time_col = "timestamp"
            else:
                print(f"⛔ {sym}: No valid time column ('startTime' or 'timestamp') in data: {df.columns.tolist()}")
                continue
            
            df["datetime"] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
            df.set_index("datetime", inplace=True)
            try:
                df = df.astype(float).sort_index()
            except Exception as e:
                print(f"⛔ {sym}: Data type conversion failed: {e}")
                continue
            

            ltp = df["close"].iloc[-1]
            if ltp > capital:
                print(f"⛔ {sym}: LTP ₹{ltp:.2f} > capital ₹{capital:.2f} — Skipped")
                continue
            else:
                print(f"✅ {sym}: LTP ₹{ltp:.2f} < capital ₹{capital:.2f}")

            rsi = RSIIndicator(df["close"], window=14).rsi().iloc[-1]
            if np.isnan(rsi):
                print(f"⛔ {sym}: RSI calculation failed (NaN)")
                continue
            if rsi > 70:
                print(f"⛔ {sym}: RSI {rsi:.2f} > 70 — Skipped")
                continue
            else:
                print(f"✅ {sym}: RSI {rsi:.2f} ≤ 70")

            atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]
            if np.isnan(atr):
                print(f"⛔ {sym}: ATR calculation failed (NaN)")
                continue
            if atr < 1.0:
                print(f"⛔ {sym}: ATR {atr:.2f} < 1.0 — Skipped")
                continue
            else:
                print(f"✅ {sym}: ATR {atr:.2f} ≥ 1.0")
            
            avg_vol = df["volume"].tail(5).mean()
            if np.isnan(avg_vol):
                print(f"⛔ {sym}: Volume calculation failed (NaN)")
                continue
            if avg_vol < 10000:
                print(f"⛔ {sym}: Avg Vol {avg_vol:,.0f} < 10,000 — Skipped")
                continue
            else:
                print(f"✅ {sym}: Avg Vol {avg_vol:,.0f} ≥ 10,000")
            
            # Momentum check: last close > close from 5 candles ago
            if df["close"].iloc[-1] < df["close"].iloc[-5]:
                print(f"⛔ {sym}: Negative momentum (Last close {df['close'].iloc[-1]:.2f} < {df['close'].iloc[-5]:.2f}) — Skipped")
                continue
            else:
                print(f"✅ {sym}: Momentum positive")

            sma_20 = SMAIndicator(df["close"], window=20).sma_indicator().iloc[-1]
            if np.isnan(sma_20):
                print(f"⛔ {sym}: SMA20 calculation failed (NaN)")
                continue
            else:
                print(f"✅ {sym}: SMA20 computed ({sma_20:.2f})")

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
            print(f"✅ {sym}: PASSED ALL FILTERS and added to final list")

        except Exception as e:
            print(f"❌ {sym} failed: {e}")

# === Save final CSV ===
final = pd.DataFrame(qualified)
final.to_csv("smallcap_dynamic_stock_list.csv", index=False)
print(f"\n✅ Final small-cap list saved with {len(final)} entries.")
