
from itertools import islice

def batched(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

# 📄 File: Dynamic_Gpt_Momentum.py

import pandas as pd
import openai
import yfinance as yf
import datetime
import pytz
import json
import os
import time as systime

# ✅ Load config.json (OpenAI Key inside)
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

OPENAI_API_KEY = config["openai_api_key"]

# ✅ Load Dynamic Stock List
def load_dynamic_stocks():
    try:
        with open('D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt', 'r') as f:
            return [line.strip().upper() for line in f if line.strip()]
    except Exception as e:
        print(f"⚠️ Error loading dynamic stock list: {e}")
        return []

STOCKS_TO_WATCH = load_dynamic_stocks()

# ✅ Fetch Recent 5-min and 15-min Candles
def fetch_candle_data(symbol):
    try:
        now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        start = now - datetime.timedelta(days=2)
        data_5min = yf.download(
            tickers=f"{symbol}.NS",
            start=start.strftime('%Y-%m-%d'),
            interval="5m",
            progress=False,
            auto_adjust=True
        )
        data_15min = yf.download(
            tickers=f"{symbol}.NS",
            start=start.strftime('%Y-%m-%d'),
            interval="15m",
            progress=False,
            auto_adjust=True
        )
        return data_5min, data_15min
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return None, None

# ✅ Prepare Live Intraday Data
def prepare_data():
    records = []
    for stock in STOCKS_TO_WATCH:
        data_5, data_15 = fetch_candle_data(stock)
        if data_5 is None or data_5.empty or data_15 is None or data_15.empty:
            continue
        try:
            if data_5['Open'].empty or data_5['Close'].empty or data_5['Volume'].empty:
                continue
            open_price = data_5['Open'].iloc[-1].item()
            close_price = data_5['Close'].iloc[-1].item()
            volume_value = data_5['Volume'].iloc[-1].item()
            change_pct_5m = round(((close_price - open_price) / open_price) * 100, 2)

            last_5_candles = data_15.tail(5)
            trend_strength = "Strong" if all(
                last_5_candles['Close'].iloc[i].item() > last_5_candles['Open'].iloc[i].item()
                    for i in range(len(last_5_candles))
            ) else "Weak"
            
            record = {
                "symbol": stock,
                "5min_change_pct": change_pct_5m,
                "volume_value": volume_value,
                "trend_strength": trend_strength
            }
            records.append(record)
            systime.sleep(1.5)
        except Exception as e:
            print(f"⚠️ Error processing {stock}: {e}")
            continue
    df = pd.DataFrame(records)
    return df

# ✅ Ask GPT to Pick Best Stock
def ask_gpt_to_pick_stock(df):
    openai.api_key = OPENAI_API_KEY
    try:
        prompt = f"""
You are an expert intraday stock trading advisor.
Analyze the following stock data carefully:

{df.to_string(index=False)}

Rules:
- Prefer 'Strong' trend_strength
- Prefer 5min_change_pct > 0.20%
- Volume must be meaningful (volume_value > 500000 ideally)
- Avoid 'Weak' stocks even if % is high
- If all stocks are risky, reply "SKIP".

Pick exactly one safest stock to BUY.
Reply strictly with stock symbol (example: INFY) or "SKIP".
"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        decision = response.choices[0].message.content.strip().upper()
        return decision
    except Exception as e:
        print(f"⚠️ Error with GPT selection: {e}")
        return "SKIP"

# ✅ Check if Market is Open
def is_market_open():
    now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    if now.weekday() >= 5:
        return False
    if now.hour < 9 or (now.hour == 9 and now.minute < 15):
        return False
    if now.hour > 15 or (now.hour == 15 and now.minute > 30):
        return False
    return True

# ✅ Main (for manual testing only)
if __name__ == "__main__":
    if not is_market_open():
        print("🚫 Market is closed. Skipping momentum analysis.")
        exit(0)

    df = prepare_data()
    if df.empty:
        print("⚠️ No valid data fetched. Exiting.")
    else:
        print("\n📊 Live Data:\n", df)
        print("\n🤖 Sending to GPT for analysis...\n")
        decision = ask_gpt_to_pick_stock(df)
        print(f"\n✅ GPT Decision: {decision}")
