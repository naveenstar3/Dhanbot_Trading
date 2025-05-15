
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
            payload = {
                   "securityId": security_id,
                   "exchangeSegment": "NSE_EQ",
                   "instrument": "EQUITY",
                   "interval": interval,
                   "oi": False,
                   "fromDate": from_date,
                   "toDate": to_date
               }
            print(f"📤 Sending request for {symbol} [{interval}] with payload: {payload}")  
            res = requests.post(url, headers=headers, json=payload)
            
            if res.status_code != 200:
                print(f"❌ API request failed for {symbol} [{interval}] - Status: {res.status_code}")
                print(f"🔎 Response text: {res.text}")
                return None
       
            try:
                response_json = res.json()
                if "data" not in response_json or not response_json["data"]:
                    print(f"⚠️ Empty or missing data in response for {symbol} [{interval}]")
                    print(f"🔎 Full response: {response_json}")
                    return None
                data = pd.DataFrame(response_json["data"])
                if data.empty:
                    return None               
                data.rename(columns={"open": "Open", "close": "Close", "volume": "Volume"}, inplace=True)
                return data
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
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers)
        data = response.json()
        dp = data["securityWiseDP"]
        delivery_qty = int(dp["deliveredQuantity"])
        traded_qty = int(dp["tradedQuantity"])
        if traded_qty == 0:
            return 0
        return round((delivery_qty / traded_qty) * 100, 2)
    except Exception as e:
        print(f"⚠️ Delivery % fetch failed for {symbol}: {e}")
        return 0

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
    records = []
    for stock in STOCKS_TO_WATCH:
        data_5, data_15 = fetch_candle_data(stock)
        if data_5 is None or data_5.empty or data_15 is None or data_15.empty:
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

            # Gap-up check
            prev_close = data_5['Close'].iloc[-2]
            gap_pct = round(((open_price - prev_close) / prev_close) * 100, 2)

            # RSI Calculation
            rsi_series = calculate_rsi(data_5['Close'])
            rsi = round(rsi_series.iloc[-1], 2) if not rsi_series.empty else 0

            # Delivery %
            delivery = get_delivery_percentage(stock)

            # Momentum Score (custom ML style)
            momentum_score = round((change_pct_5m * 0.6 + gap_pct * 0.2 + delivery * 0.2), 2)

            if delivery < 30 or rsi > 70 or gap_pct > 5:
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
You are a smart intraday momentum advisor.

Analyze the following stock data:

{df.to_string(index=False)}

Rules:
- Strong trend_strength is preferred
- delivery_pct must be ≥ 30
- RSI must be < 70
- Avoid gap_pct > 5%
- Prefer high momentum_score
- Avoid if 5min_change_pct < 0.2
- If all risky, reply "SKIP"

Reply with ONE stock symbol to buy, or "SKIP"
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
