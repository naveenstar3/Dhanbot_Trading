# 📄 File: dynamic_stock_generator.py

import pandas as pd
import openai
import yfinance as yf
import datetime
import pytz
import json
import os
import requests
from bs4 import BeautifulSoup

# ✅ Load config.json (OpenAI Key inside)
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

OPENAI_API_KEY = config["openai_api_key"]

# ✅ Check if Market is Open

def is_market_open():
    now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False
    if now.hour < 9 or (now.hour == 9 and now.minute < 15):
        return False
    if now.hour > 15 or (now.hour == 15 and now.minute > 30):
        return False
    return True

# ✅ Fetch Symbols Dynamically from Yahoo Finance Most Active Stocks

def fetch_dynamic_symbols():
    try:
        url = "https://finance.yahoo.com/most-active?offset=0&count=100&market=IN"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"⚠️ Failed to fetch Yahoo page. Status Code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        symbols = []

        for item in soup.find_all('a', attrs={'class': 'Fw(600) C($linkColor)'}):
            symbol = item.text.strip().upper()
            if symbol.endswith(".NS"):  # Only take NSE stocks
                symbol = symbol.replace(".NS", "")
                symbols.append(symbol)

        print(f"✅ Successfully fetched {len(symbols)} live symbols from Yahoo Finance.")
        return symbols

    except Exception as e:
        print(f"⚠️ Error fetching dynamic symbols: {e}")
        return []

# ✅ Fetch 5-min Candle Data

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
        return data_5min
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return None

# ✅ Prepare Intraday Data

def prepare_data(symbols):
    records = []
    for stock in symbols:
        data_5 = fetch_candle_data(stock)
        if data_5 is None or data_5.empty:
            continue
        try:
            last_candle = data_5.iloc[-1]
            open_price = float(last_candle['Open'])
            close_price = float(last_candle['Close'])
            volume_value = float(last_candle['Volume'])
            change_pct_5m = round(((close_price - open_price) / open_price) * 100, 2)

            record = {
                "symbol": stock,
                "5min_change_pct": change_pct_5m,
                "volume_value": volume_value
            }
            records.append(record)
        except Exception as e:
            print(f"⚠️ Error processing {stock}: {e}")
            continue
    df = pd.DataFrame(records)
    return df

# ✅ Use OpenAI to pick best stocks

def ask_gpt_to_pick_stocks(df):
    openai.api_key = OPENAI_API_KEY
    try:
        prompt = f"""
You are an expert intraday stock trading advisor.
Analyze the following stock data carefully:

{df.to_string(index=False)}

Rules:
- Prefer stocks where 5min_change_pct > 0.20%
- Prefer volume_value > 500000
- Pick 5-10 best momentum stocks based on strength and liquidity.
- Return only a comma-separated list of stock symbols (example: RELIANCE,TCS,INFY)
If no good stocks, reply "SKIP".
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

# ✅ Save Dynamic List

def save_to_file(symbols, output_path):
    try:
        with open(output_path, "w") as f:
            for symbol in symbols:
                f.write(symbol + "\n")
        print(f"✅ Saved {len(symbols)} stocks to {output_path}")
    except Exception as e:
        print(f"⚠️ Error saving to file: {e}")

# ✅ Download Dhan Master CSV Once Daily

def download_dhan_master_csv():
    try:
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        response = requests.get(url)
        if response.status_code == 200:
            with open("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv", "w", encoding="utf-8") as f:
                f.write(response.text)
            print("✅ Dhan master CSV downloaded and saved.")
        else:
            print(f"⚠️ Failed to download Dhan master CSV. Status: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error downloading Dhan master CSV: {e}")

# ✅ Main Execution

if __name__ == "__main__":
    print("🚀 Generating Dynamic Stock List...")

    if not is_market_open():
        print("🚫 Market is closed. Skipping stock generation.")
        exit(0)

    symbols = fetch_dynamic_symbols()
    if not symbols:
        print("⚠️ No symbols fetched. Exiting.")
        exit(1)

    df = prepare_data(symbols)
    if df.empty:
        print("⚠️ No valid data fetched. Exiting.")
        exit(1)

    print("📊 Intraday Data Ready. Sending to GPT...")
    decision = ask_gpt_to_pick_stocks(df)

    if decision == "SKIP" or not decision.strip():
        print("⚠️ No good stocks today. Skipping.")
        exit(0)

    selected_symbols = [s.strip().upper() for s in decision.split(",") if s.strip()]
    output_file = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt"
    save_to_file(selected_symbols, output_file)

    # ✅ Download Dhan Master CSV after generating stock list
    download_dhan_master_csv()

    print("🎯 Dynamic Stock List Generation Completed!")
