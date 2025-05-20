# ✅ Step 1: Fetching Broad Stock Universe
import pandas as pd
import openai
import yfinance as yf
import pytz
import os
import requests
import json
import time as systime
import csv
from bs4 import BeautifulSoup
from dhan_api import get_historical_price, get_security_id
from dhan_api import get_live_price, get_current_capital 
from concurrent.futures import ThreadPoolExecutor, as_completed
from dhan_api import get_security_id_from_trading_symbol
from utils_logger import log_bot_action
from datetime import datetime
import zipfile
from io import BytesIO


# ✅ Quick debug: Check if security ID is working
from dhan_api import get_security_id
print("RELIANCE ID:", get_security_id("RELIANCE"))  # ✅ Add this line

# ✅ Load OpenAI Key
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)
OPENAI_API_KEY = config["openai_api_key"]

# ✅ Fallback Safe Stock List (For Offline Prefilter)
fallback_safe_list = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "LT", "SBIN", "AXISBANK", "ITC", "KOTAKBANK",
    "BHARTIARTL", "HCLTECH", "WIPRO", "BAJFINANCE", "ADANIENT",
    "POWERGRID", "HINDUNILVR", "ASIANPAINT", "ULTRACEMCO", "MARUTI",
    "DIVISLAB", "TITAN", "SUNPHARMA", "TECHM", "NESTLEIND",
    "JSWSTEEL", "COALINDIA", "BPCL", "BRITANNIA", "GRASIM",
    "ONGC", "INDUSINDBK", "EICHERMOT", "HDFCLIFE", "HEROMOTOCO"
]

def is_market_closed():
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    hour = now.hour + now.minute / 60.0
    return weekday >= 5 or hour < 9.25 or hour > 15.5

def get_current_capital():
    try:
        with open("D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv", "r") as f:
            return float(f.read().strip())
    except:
        return 3000  # default fallback capital
        
# ✅ Safe technical filter to check 5-min + 15-min momentum
def technical_filter(valid_ids):
    print("🚀 Starting Safe Technical Momentum Filter...")
    passed_symbols = []

    def evaluate_momentum(symbol, candles):
        try:
            if not candles or len(candles) < 5:
                return None
            df = pd.DataFrame(candles)
            latest = df['close'].iloc[-1]
            five_min_ago = df['close'].iloc[-2]
            fifteen_min_ago = df['close'].iloc[-4]

            momentum_5min = ((latest - five_min_ago) / five_min_ago) * 100
            momentum_15min = ((latest - fifteen_min_ago) / fifteen_min_ago) * 100

            if 0.2 <= momentum_5min <= 0.6 and momentum_15min > 0:
                return symbol
        except Exception as e:
            print(f"⚠️ Momentum calc error for {symbol}: {e}")
        return None

    for symbol, sec_id in valid_ids:
        try:
            systime.sleep(1.2)  # Rate limit guard
            candles = get_historical_price(sec_id, interval="5", limit=5)
            if candles:
                result = evaluate_momentum(symbol, candles)
                if result:
                    passed_symbols.append(result)
        except Exception as e:
            print(f"⚠️ Fetch error for {symbol}: {e}")
            continue

    print(f"✅ {len(passed_symbols)} symbols passed momentum filter.")
    return passed_symbols
    
# ✅ Optional: Volume Surge Filter using 15-min candles
def volume_surge_filter(valid_ids):
    print("📊 Starting Volume Surge Filter...")
    filtered = []

    for symbol, sec_id in valid_ids:
        try:
            systime.sleep(1.2)  # Prevent DH-904 rate-limit
            candles = get_historical_price(sec_id, interval="15", limit=6)
            if not candles or len(candles) < 5:
                continue

            df = pd.DataFrame(candles)
            avg_volume = df["volume"].iloc[:-1].mean()
            last_volume = df["volume"].iloc[-1]

            if last_volume >= 1.2 * avg_volume:
                filtered.append(symbol)
            else:
                print(f"⛔ {symbol} volume low: {last_volume} < 1.2×{round(avg_volume)}")

        except Exception as e:
            print(f"⚠️ Volume filter failed for {symbol}: {e}")
            continue

    print(f"✅ {len(filtered)} symbols passed volume surge filter.")
    return filtered
    
# ✅ Optional: News Sentiment Filter via Google RSS (no API key needed)
def sentiment_filter(symbols):
    print("📰 Running sentiment filter using Google News...")
    bad_words = [
        "fraud", "resign", "fire", "loss", "scam", "raid",
        "income tax", "default", "down", "cut", "criminal",
        "lawsuit", "penalty", "accident", "fine", "crash", "fall"
    ]
    passed = []

    for symbol in symbols:
        try:
            query = f"{symbol} stock"
            url = f"https://news.google.com/rss/search?q={query}+when:1d&hl=en-IN&gl=IN&ceid=IN:en"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")
            headlines = [item.title.text.lower() for item in items]

            bad_found = any(any(bad in h for bad in bad_words) for h in headlines)
            if not bad_found:
                passed.append(symbol)
            else:
                print(f"⛔ {symbol} blocked due to negative news")

        except Exception as e:
            print(f"⚠️ News check failed for {symbol}: {e}")
            continue

    print(f"✅ {len(passed)} symbols passed sentiment filter.")
    return passed
    
# ✅ Optional: RSI Filter using 15-min candles (Safe RSI < 70)
def rsi_filter(valid_ids):
    print("📐 Starting RSI Filter (RSI-14 < 70)...")
    passed = []

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    for symbol, sec_id in valid_ids:
        try:
            systime.sleep(1.2)
            candles = get_historical_price(sec_id, interval="15", limit=20)
            df = pd.DataFrame(candles)
            if df.empty or len(df) < 15:
                continue

            rsi_series = calculate_rsi(df["close"])
            latest_rsi = rsi_series.iloc[-1]

            if latest_rsi < 70:
                passed.append(symbol)
            else:
                print(f"⛔ {symbol} skipped — RSI {round(latest_rsi,1)} > 70")

        except Exception as e:
            print(f"⚠️ RSI check failed for {symbol}: {e}")
            continue

    print(f"✅ {len(passed)} symbols passed RSI filter.")
    return passed

# ✅ STEP 1: Fallback + Security ID validation
symbols = fallback_safe_list[:25]
valid_ids = []
for s in symbols:
    sid = get_security_id(s)
    if sid and str(sid).isdigit():
        valid_ids.append((s, sid))

def fetch_nse_bhavcopy_csv():
    from datetime import datetime as dt

    # Paths
    local_csv = "D:/Downloads/Dhanbot/nse_bhav/bhavcopy_latest.csv"
    local_zip = "D:/Downloads/Dhanbot/nse_bhav/bhavcopy_latest.zip"

    # Bhavcopy date
    today = dt.now().strftime("%d%b%Y").upper()
    zip_url = f"https://www1.nseindia.com/content/historical/EQUITIES/{dt.now().strftime('%Y')}/{dt.now().strftime('%b').upper()}/cm{today}bhav.csv.zip"
    live_url = "https://www1.nseindia.com/content/nsccl/bulk.csv"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    }

    # First try ZIP
    try:
        print(f"🌐 Trying historical ZIP: {zip_url}")
        zip_response = requests.get(zip_url, headers=headers, timeout=10)

        with open(local_zip, "wb") as f:
            f.write(zip_response.content)

        # Try unzip
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall("D:/Downloads/Dhanbot/nse_bhav")
            for fname in os.listdir("D:/Downloads/Dhanbot/nse_bhav"):
                if fname.startswith("cm") and fname.endswith("bhav.csv"):
                    os.rename(f"D:/Downloads/Dhanbot/nse_bhav/{fname}", local_csv)
                    print(f"✅ Bhavcopy ZIP extracted to: {local_csv}")
                    return local_csv

    except Exception as e:
        print(f"⚠️ ZIP failed: {e}")
        print("🌐 Trying live CSV fallback...")

    # Try live CSV fallback
    try:
        live_response = requests.get(live_url, headers=headers, timeout=10)
        with open(local_csv, "wb") as f:
            f.write(live_response.content)
        print(f"✅ Live Bhavcopy CSV saved to: {local_csv}")
        return local_csv
    except Exception as e:
        print(f"❌ Failed to fetch live CSV Bhavcopy: {e}")
        return None

# ⏳ Step: Preload affordable stock list from bhavcopy
bhav_path = fetch_nse_bhavcopy_csv()
bhav_prices = {}

if bhav_path and os.path.exists(bhav_path):
    try:
        df_bhav = pd.read_csv(bhav_path, on_bad_lines='skip')

        if "SERIES" in df_bhav.columns:
            # Historical bhavcopy ZIP format
            df_bhav = df_bhav[df_bhav["SERIES"] == "EQ"]
            bhav_prices = dict(zip(df_bhav["SYMBOL"].str.upper(), df_bhav["CLOSE"]))
        elif "Symbol" in df_bhav.columns and "Close Price" in df_bhav.columns:
            # Live bulk.csv format
            bhav_prices = dict(zip(df_bhav["Symbol"].str.upper(), df_bhav["Close Price"]))
        else:
            print("❌ Unknown Bhavcopy format. Columns:", df_bhav.columns.tolist())

    except Exception as e:
        print(f"⚠️ Failed to process Bhavcopy: {e}")

# ✅ STEP 2A: Filter by capital first
capital = get_current_capital()
affordable_ids = []

for symbol, sec_id in valid_ids:
    try:
        price = bhav_prices.get(symbol)
        if price is not None and price <= capital:
            affordable_ids.append((symbol, sec_id))
        else:
            print(f"⛔ {symbol} skipped — ₹{price} > ₹{capital}")
    except Exception as e:
        print(f"⚠️ Price check failed for {symbol}: {e}")
        continue

# ✅ STEP 2B: Apply safe momentum filter
tech_passed = technical_filter(affordable_ids)
# ✅ STEP 2C: Apply volume surge filter to tech_passed
vol_filtered = volume_surge_filter([
    (s, sec_id) for s, sec_id in affordable_ids if s in tech_passed
])
affordable = vol_filtered  # Final list to proceed
print(f"✅ {len(affordable)} symbols passed volume filter under ₹{capital}")
# ✅ STEP 2D: Apply sentiment filter to remove bad-news stocks
affordable = sentiment_filter(affordable)
print(f"✅ {len(affordable)} symbols passed sentiment news filter")
# ✅ STEP 2E: Apply RSI filter to avoid overbought stocks
affordable = rsi_filter([
    (s, sec_id) for s, sec_id in affordable_ids if s in affordable
])
print(f"✅ {len(affordable)} symbols passed RSI filter")

# ✅ STEP 3: Build DataFrame with 5min_change_pct and volume_value
df = pd.DataFrame(columns=["symbol", "5min_change_pct", "volume_value"])
symbol_to_id = {symbol: sec_id for symbol, sec_id in affordable_ids if symbol in affordable}

for symbol in affordable:
    try:
        security_id = symbol_to_id.get(symbol)
        if not security_id:
            print(f"⛔ No security_id for: {symbol}")
            continue

        # 🕒 API candle fetch for just this symbol
        systime.sleep(1.2)  # rate-limit guard
        candles_5m = get_historical_price(security_id, interval="5m", limit=5)
        candles_15m = get_historical_price(security_id, interval="15", limit=6)

        df_5m = pd.DataFrame(candles_5m)
        df_15m = pd.DataFrame(candles_15m)

        if df_5m.empty or df_15m.empty:
            print(f"❌ {symbol} — Empty candle data")
            continue

        change_pct = ((df_5m["close"].iloc[-1] - df_5m["close"].iloc[-2]) / df_5m["close"].iloc[-2]) * 100
        vol_value = df_15m["volume"].iloc[-1] * df_5m["close"].iloc[-1]

        df.loc[len(df)] = {
            "symbol": symbol,
            "5min_change_pct": round(change_pct, 2),
            "volume_value": round(vol_value)
        }

    except Exception as e:
        print(f"⚠️ Error building GPT data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        continue
# ✅ STEP 4: Ask GPT to pick from DataFrame
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
- Pick 5–10 best momentum stocks based on strength and liquidity.
- Return only a comma-separated list of stock symbols (example: RELIANCE,TCS,INFY)
If no good stocks, reply "SKIP".
"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"⚠️ GPT API error: {e}")
        return "SKIP"

# ✅ STEP 5: GPT Evaluation + Budget Safety Check
expected_cols = ["symbol", "5min_change_pct", "volume_value"]
if df.empty or not all(col in df.columns for col in expected_cols):
    print("⚠️ Skipping GPT call — DataFrame invalid.")
    final_stocks = fallback_safe_list[:3]
else:
    print("📊 Calling GPT for selection...")
    gpt_selected_raw = ask_gpt_to_pick_stocks(df)

    if gpt_selected_raw == "SKIP":
        print("⚠️ GPT said SKIP. Using fallback.")
        final_stocks = fallback_safe_list[:3]
    else:
        selected = [s.strip() for s in gpt_selected_raw.split(",") if s.strip()]
        final_stocks = []

        for symbol in selected:
            try:
                price = bhav_prices.get(symbol)
                if price and price <= capital:
                    final_stocks.append(symbol)
            except:
                continue

        if not final_stocks:
            print("⚠️ GPT picks not affordable. Using fallback.")
            final_stocks = fallback_safe_list[:3]

# ✅ STEP 6: Save to file
with open('D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt', 'w') as f:
    for stock in final_stocks:
        f.write(stock + "\n")

log_bot_action("dynamic_stock_generator.py", "Stock List Updated", "✅ COMPLETE", f"{len(final_stocks)} stocks saved")
print(f"✅ Final {len(final_stocks)} stocks saved to dynamic_stock_list.txt: {final_stocks}")
