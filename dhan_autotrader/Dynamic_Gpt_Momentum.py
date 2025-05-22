
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
import csv
import requests
from utils_logger import log_bot_action


# ✅ Load config.json (OpenAI Key inside)
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

# ✅ Patch missing client_id to avoid KeyError
if "client_id" not in config or not config["client_id"]:
    config["client_id"] = "101123227287"

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

def get_security_id(symbol):
    try:
        master_path = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
        with open(master_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key in ["SM_SYMBOL_NAME", "SEM_CUSTOM_SYMBOL", "SEM_TRADING_SYMBOL"]:
                    if row.get(key) and row[key].strip().upper() == symbol.strip().upper():
                        return row["SEM_SMST_SECURITY_ID"]
    except Exception as e:
        print(f"⚠️ Failed to fetch security ID for {symbol}: {e}")
    return None

# ✅ Fetch Recent 5-min and 15-min Candles
def fetch_candle_data(symbol):
    try:
        security_id = get_security_id(symbol)
        if not security_id:
            print(f"⚠️ No security ID found for {symbol}")
            return None, None

        headers = {
            "access-token": config["access_token"],
            "client-id": config["client_id"],
            "Content-Type": "application/json"
        }

        url = "https://api.dhan.co/v2/charts/intraday"
        now = datetime.datetime.now()
        from_date = (now - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
        to_date = now.strftime('%Y-%m-%d')

        def fetch(interval):
            india = pytz.timezone("Asia/Kolkata")
            now = datetime.datetime.now(india)
            from_dt = (now - datetime.timedelta(days=2)).replace(hour=9, minute=15, second=0, microsecond=0)
            to_dt = now
        
            payload = {
                "securityId": security_id,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "interval": interval,
                "oi": False,
                "fromDate": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": to_dt.strftime("%Y-%m-%d %H:%M:%S")
            }
        
            print(f"📤 Sending request for {symbol} [{interval}] with payload: {payload}")
            res = requests.post(url, headers=headers, json=payload)
        
            if res.status_code != 200:
                print(f"❌ API request failed for {symbol} [{interval}] - Status: {res.status_code}")
                print(f"🔎 Response text: {res.text}")
                return None
        
            try:
                response_json = res.json()
                if "open" not in response_json or not response_json["open"]:
                    print(f"⚠️ Empty or missing OHLC data in response for {symbol} [{interval}]")
                    return None
                df = pd.DataFrame({
                    "Open": response_json["open"],
                    "Close": response_json["close"],
                    "Volume": response_json["volume"],
                    "Timestamp": pd.to_datetime(response_json["timestamp"], unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata')
                })
                return df
            except Exception as e:
                print(f"⚠️ Failed parsing response for {symbol} [{interval}]: {e}")
                return None
        
        data_1 = fetch("1MIN")
        if data_1 is not None:
            used_data = data_1
            used_interval = "1MIN"
        else:
            used_data = fetch("5MIN")
            used_interval = "5MIN" if used_data is not None else "None"

        print(f"✅ Used {used_interval} data for {symbol}")
        return used_data, fetch("15MIN")

    except Exception as e:
        print(f"⚠️ Error fetching OHLC for {symbol}: {e}")
        return None, None
        
def get_delivery_percentage(symbol):
        return 35.0

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
        
# ✅ Prepare Live Intraday Data
def prepare_data():
    log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "START", "Preparing momentum + delivery + RSI")
    print(f"📦 Total candidates from dynamic_stock_list.txt: {len(STOCKS_TO_WATCH)}")
    
    records = []
    total_attempted = 0
    for stock in STOCKS_TO_WATCH:
        total_attempted += 1
        data_5, data_15 = fetch_candle_data(stock)

        if data_5 is None or data_5.empty or data_15 is None or data_15.empty:
            print(f"⚠️ Skipping {stock}: Empty candle data (5m or 15m)")
            continue

        try:
            open_price = data_5['Open'].iloc[-1]
            close_price = data_5['Close'].iloc[-1]
            volume_value = data_5['Volume'].iloc[-1]
            change_pct_5m = round(((close_price - open_price) / open_price) * 100, 2)

            last_5_candles = data_15.tail(5)
            trend_strength = "Strong" if all(
                last_5_candles['Close'].iloc[i] > last_5_candles['Open'].iloc[i]
                for i in range(len(last_5_candles))
            ) else "Weak"

            prev_close = data_5['Close'].iloc[-2]
            gap_pct = round(((open_price - prev_close) / prev_close) * 100, 2)

            # RSI Calculation
            rsi_series = calculate_rsi(data_5['Close'])
            rsi = round(rsi_series.iloc[-1], 2) if not rsi_series.empty else 0

            # Delivery %
            delivery = get_delivery_percentage(stock)

            # Momentum Score
            score = 0
            score += change_pct_5m * 0.5        # stronger weight for 5m momentum
            score += gap_pct * 0.15             # moderate gap weight
            score += delivery * 0.2             # stable delivery weight
            score += (volume_value / 1_00_000) * 0.05  # volume boost (scaled)
            
            if trend_strength == "Strong":
                score += 0.5  # small fixed bonus for strong trend
            
            momentum_score = round(score, 2)
            

            if delivery < 30 or rsi > 75 or gap_pct > 5:
                print(f"❌ Filtered {stock}: Delivery={delivery}%, RSI={rsi}, Gap={gap_pct}%")
                continue

            record = {
                "symbol": stock,
                "5min_change_pct": change_pct_5m,
                "gap_pct": gap_pct,
                "delivery_pct": delivery,
                "rsi": rsi,
                "trend_strength": trend_strength,
                "momentum_score": momentum_score,
                "volume_value": volume_value
            }
            records.append(record)
            print(f"✅ Added: {stock} | Score={momentum_score}")
            systime.sleep(1.2)

        except Exception as e:
            print(f"⚠️ Error processing {stock}: {e}")
            continue

    df = pd.DataFrame(records)

    print(f"📊 Completed: {len(records)}/{total_attempted} passed filters")

    if df.empty:
        log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "❌ EMPTY", "No stocks passed filters")
    else:
        log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "✅ COMPLETE", f"{len(df)} stocks processed")

    return df
    
# ✅ Ask GPT to Pick Best Stock
def ask_gpt_to_rank_stocks(df):
    openai.api_key = OPENAI_API_KEY
    try:
        prompt = f"""
You are a smart intraday momentum advisor.

Analyze the following stock data:

{df.to_string(index=False)}

Rules:
- Strong trend_strength is preferred
- delivery_pct must be ≥ 30
- RSI must be < 75
- Avoid gap_pct > 5%
- Prefer high momentum_score
- Avoid if 5min_change_pct < 0.2
- If all risky, reply "SKIP"

Reply with a comma-separated list of symbols (ex: RELIANCE,TCS) in rank order.
If no safe stocks, reply "SKIP"
"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        gpt_response = response.choices[0].message.content.strip().upper()

        if gpt_response == "SKIP" or not gpt_response:
            # ⛑️ Fallback: Pick top momentum score stock manually
            fallback = df.sort_values("momentum_score", ascending=False).head(1)["symbol"].tolist()
            log_bot_action("Dynamic_Gpt_Momentum.py", "ask_gpt_to_rank_stocks", "⚠️ GPT SKIP → FORCED PICK", f"Fallback: {fallback}")
            return fallback
        else:
            # ✅ Valid ranked response
            candidates = [s.strip() for s in gpt_response.split(",") if s.strip() in df["symbol"].values]
            if not candidates:
                fallback = df.sort_values("momentum_score", ascending=False).head(1)["symbol"].tolist()
                log_bot_action("Dynamic_Gpt_Momentum.py", "ask_gpt_to_rank_stocks", "⚠️ GPT BAD → FORCED PICK", f"Fallback: {fallback}")
                return fallback
            log_bot_action("Dynamic_Gpt_Momentum.py", "ask_gpt_to_rank_stocks", "✅ GPT SELECT", f"{candidates}")
            return candidates
    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        fallback = df.sort_values("momentum_score", ascending=False).head(1)["symbol"].tolist()
        log_bot_action("Dynamic_Gpt_Momentum.py", "ask_gpt_to_rank_stocks", "⚠️ GPT FAIL → FORCED PICK", f"Fallback: {fallback}")
        return fallback

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
        decision = ask_gpt_to_rank_stocks(df)
        print(f"\n✅ GPT Decision: {decision}")
