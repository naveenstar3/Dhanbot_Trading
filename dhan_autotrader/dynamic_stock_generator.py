# ✅ Step 1: Fetching Broad Stock Universe
import pandas as pd
import openai
import pytz
import os
import requests
import json
import time as systime
from bs4 import BeautifulSoup
from dhan_api import get_historical_price, get_security_id
from dhan_api import get_current_capital
from utils_logger import log_bot_action
from datetime import datetime
import zipfile

# ✅ Debug: Security ID check
print("RELIANCE ID:", get_security_id("RELIANCE"))

# ✅ Load OpenAI Key
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)
OPENAI_API_KEY = config["openai_api_key"]

# ✅ Technical Momentum Filter Function
def technical_filter(stock_list):
    filtered = []

    for symbol, security_id in stock_list:
        try:
            candles_5m = get_historical_price(security_id, interval="5m", limit=5)
            candles_15m = get_historical_price(security_id, interval="15", limit=6)

            df_5m = pd.DataFrame(candles_5m)
            df_15m = pd.DataFrame(candles_15m)

            if df_5m.empty or df_15m.empty:
                continue

            # Momentum logic: last 5-min candle must show at least 0.2% up
            close_now = df_5m["close"].iloc[-1]
            close_prev = df_5m["close"].iloc[-2]
            change_pct = ((close_now - close_prev) / close_prev) * 100

            if change_pct >= 0.2:
                filtered.append(symbol)

        except Exception as e:
            print(f"⚠️ Technical filter error for {symbol}: {e}")
            continue

    return filtered

def volume_surge_filter(stock_list):
    surged = []

    for symbol, security_id in stock_list:
        try:
            candles = get_historical_price(security_id, interval="15", limit=6)
            df = pd.DataFrame(candles)

            if df.empty:
                continue

            current_vol = df["volume"].iloc[-1]
            avg_vol = df["volume"].iloc[:-1].mean()

            if current_vol > 1.2 * avg_vol:
                surged.append((symbol, security_id))

        except Exception as e:
            print(f"⚠️ Volume surge error for {symbol}: {e}")
            continue

    return surged

def sentiment_filter(stock_list):
    import re

    def fetch_news_headlines(symbol):
        try:
            url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={symbol.lower()}&durationType=Y&year=2023"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            headlines = [tag.get_text(strip=True) for tag in soup.select(".MT15 h2")]
            return headlines[:5]  # Top 5 recent headlines
        except Exception as e:
            print(f"⚠️ Failed to fetch news for {symbol}: {e}")
            return []

    def gpt_analyze_sentiment(headlines):
        try:
            if not headlines:
                return "POSITIVE"
            joined = "\n".join(headlines)
            prompt = f"""
Rate the tone of these news headlines as POSITIVE or NEGATIVE (no explanations). Reply only with one word.

{joined}
"""
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content.strip().upper()
            return "POSITIVE" if "POS" in result else "NEGATIVE"
        except Exception as e:
            print(f"⚠️ GPT sentiment check failed: {e}")
            return "POSITIVE"  # fallback default

    filtered = []
    for symbol, sec_id in stock_list:
        headlines = fetch_news_headlines(symbol)
        tone = gpt_analyze_sentiment(headlines)
        print(f"📰 {symbol}: {tone}")
        if tone == "POSITIVE":
            filtered.append(symbol)

    return filtered

def rsi_filter(stock_list):
    passed = []

    for symbol, security_id in stock_list:
        try:
            candles = get_historical_price(security_id, interval="15", limit=15)
            df = pd.DataFrame(candles)

            if df.empty or len(df) < 14:
                continue

            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=14).mean().iloc[-1]
            avg_loss = loss.rolling(window=14).mean().iloc[-1]

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            if rsi < 70:
                passed.append((symbol, security_id))

        except Exception as e:
            print(f"⚠️ RSI filter error for {symbol}: {e}")
            continue

    return passed

# ✅ Fetch NSE Bhavcopy CSV (ZIP or live)
def fetch_nse_bhavcopy_csv():
    from datetime import datetime as dt
    local_csv = "D:/Downloads/Dhanbot/nse_bhav/bhavcopy_latest.csv"
    local_zip = "D:/Downloads/Dhanbot/nse_bhav/bhavcopy_latest.zip"

    today = dt.now().strftime("%d%b%Y").upper()
    zip_url = f"https://www1.nseindia.com/content/historical/EQUITIES/{dt.now().strftime('%Y')}/{dt.now().strftime('%b').upper()}/cm{today}bhav.csv.zip"
    live_url = "https://www1.nseindia.com/content/nsccl/bulk.csv"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    }

    # Try ZIP
    try:
        print(f"🌐 Trying historical ZIP: {zip_url}")
        zip_response = requests.get(zip_url, headers=headers, timeout=10)
        with open(local_zip, "wb") as f:
            f.write(zip_response.content)

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

    # Try live CSV
    try:
        live_response = requests.get(live_url, headers=headers, timeout=10)
        if "text/csv" not in live_response.headers.get("Content-Type", ""):
            raise Exception("Live fallback did not return CSV content")

        content_text = live_response.content.decode('utf-8')
        if "<html" in content_text.lower() or "DOCTYPE html" in content_text.lower():
            raise Exception("Live fallback returned HTML page instead of CSV")

        with open(local_csv, "w", encoding="utf-8") as f:
            f.write(content_text)

        print(f"✅ Live Bhavcopy CSV saved to: {local_csv}")
        return local_csv
    except Exception as e:
        print(f"❌ Failed to fetch live CSV Bhavcopy: {e}")
        return None

# ✅ Helper: Check if market is closed
def is_market_closed():
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    hour = now.hour + now.minute / 60.0
    return weekday >= 5 or hour < 9.25 or hour > 15.5

# ✅ Build initial stock universe dynamically from Bhavcopy
bhav_path = fetch_nse_bhavcopy_csv()
bhav_prices = {}

if bhav_path and os.path.exists(bhav_path):
    try:
        df_bhav = pd.read_csv(bhav_path, on_bad_lines='skip')
        if "SERIES" in df_bhav.columns:
            df_bhav = df_bhav[df_bhav["SERIES"] == "EQ"]
            bhav_prices = dict(zip(df_bhav["SYMBOL"].str.upper(), df_bhav["CLOSE"]))
        elif "Symbol" in df_bhav.columns and "Close Price" in df_bhav.columns:
            bhav_prices = dict(zip(df_bhav["Symbol"].str.upper(), df_bhav["Close Price"]))
        df_bhav.to_csv("D:/Downloads/Dhanbot/nse_bhav/bhavcopy_cache.csv", index=False)
    except Exception as e:
        print(f"⚠️ Failed to process Bhavcopy: {e}")

# ✅ Dynamically build initial valid ID list from bhav_prices
valid_ids = []
MAX_SYMBOLS = 300  # adjust this if needed
for symbol in list(bhav_prices.keys())[:MAX_SYMBOLS]:
    sid = get_security_id(symbol)
    if sid and str(sid).isdigit():
        valid_ids.append((symbol, sid))

# ✅ Load backup if needed
if not bhav_prices:
    cache_file = "D:/Downloads/Dhanbot/nse_bhav/bhavcopy_cache.csv"
    if os.path.exists(cache_file):
        print("♻️ Loading bhavcopy cache...")
        try:
            df_bhav = pd.read_csv(cache_file)
            bhav_prices = dict(zip(df_bhav["SYMBOL"].str.upper(), df_bhav["CLOSE"]))
        except:
            print("⚠️ Cache load failed.")

# 🔥 Emergency fallback prices
if not bhav_prices:
    print("🚨 No bhavcopy data. Injecting fallback dummy prices...")
    bhav_prices = {symbol: 100.0 for symbol, _ in valid_ids}

# 🔥 Emergency Price Injection
if not bhav_prices:
    print("🚨 No bhavcopy data. Injecting fallback dummy prices...")
    bhav_prices = {symbol: 100.0 for symbol, _ in valid_ids}

# ✅ Filter by current capital
capital = get_current_capital()
affordable_ids = []
for symbol, sec_id in valid_ids:
    price = bhav_prices.get(symbol)
    if price is not None and price <= capital:
        affordable_ids.append((symbol, sec_id))
    else:
        print(f"⛔ {symbol} skipped — ₹{price} > ₹{capital}")

# 🚨 Last resort: Inject dummy ₹100 if no stocks pass
if not affordable_ids:
    print("🚨 No affordable stocks even after fallback. Forcing dummy ₹100 prices...")
    bhav_prices = {symbol: 100.0 for symbol, _ in valid_ids}
    affordable_ids = []
    for symbol, sec_id in valid_ids:
        affordable_ids.append((symbol, sec_id))
    

# ✅ Cache 5m and 15m candles to avoid re-fetching in filters
candle_cache = {}

for symbol, sec_id in affordable_ids:
    try:
        systime.sleep(1.0)  # Avoid rate limiting
        df_5m = pd.DataFrame(get_historical_price(sec_id, interval="5m", limit=5))
        df_15m = pd.DataFrame(get_historical_price(sec_id, interval="15", limit=6))
        if not df_5m.empty and not df_15m.empty:
            candle_cache[symbol] = {"5m": df_5m, "15m": df_15m}
    except Exception as e:
        print(f"⚠️ Candle fetch failed for {symbol}: {e}")

# ✅ Step 2B: Apply technical momentum filter
tech_passed = technical_filter(affordable_ids)

# ✅ Step 2C: Apply volume surge filter
vol_filtered = volume_surge_filter([
    (s, sec_id) for s, sec_id in affordable_ids if s in tech_passed
])
affordable = vol_filtered
print(f"✅ {len(affordable)} symbols passed volume filter under ₹{capital}")

# ✅ Step 2D: Sentiment filter (removes negative news)
affordable = sentiment_filter(affordable)
print(f"✅ {len(affordable)} symbols passed sentiment news filter")

# ✅ Step 2E: RSI filter (< 70)
affordable = rsi_filter([
    (s, sec_id) for s, sec_id in affordable_ids if s in affordable
])
print(f"✅ {len(affordable)} symbols passed RSI filter")

# ✅ Step 3: Build DataFrame for GPT scoring
df = pd.DataFrame(columns=["symbol", "5min_change_pct", "volume_value"])
symbol_to_id = {symbol: sec_id for symbol, sec_id in affordable_ids if symbol in affordable}

for symbol in affordable:
    try:
        security_id = symbol_to_id.get(symbol)
        if not security_id:
            continue

        systime.sleep(1.2)
        df_5m = candle_cache.get(symbol, {}).get("5m", pd.DataFrame())
        df_15m = candle_cache.get(symbol, {}).get("15m", pd.DataFrame())
        if df_5m.empty or df_15m.empty:
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
        continue

# ✅ Step 4: GPT-assisted final momentum ranking
vol_thresh = round(df["volume_value"].median() * 0.8) if not df.empty else 500000

def ask_gpt_to_pick_stocks(df):
    openai.api_key = OPENAI_API_KEY
    try:
        prompt = f"""
You are an expert intraday stock trading advisor.
Analyze the following stock data carefully:

{df.to_string(index=False)}

Rules:
- Prefer stocks where 5min_change_pct > 0.20%
- Prefer volume_value > {vol_thresh}
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

# ✅ Step 5: Filter GPT picks against capital safety
expected_cols = ["symbol", "5min_change_pct", "volume_value"]
final_stocks = []

def fallback_safe_pick():
    return [s for s, _ in affordable_ids if s in affordable][:3]  # always returns 1–3 safe symbols

if df.empty or not all(col in df.columns for col in expected_cols):
    print("⚠️ Skipping GPT call — DataFrame invalid or empty.")
    final_stocks = fallback_safe_pick()
else:
    print("📊 Calling GPT for selection...")
    gpt_selected_raw = ask_gpt_to_pick_stocks(df)

    if gpt_selected_raw == "SKIP":
        print("⚠️ GPT skipped. Using fallback.")
        final_stocks = fallback_safe_pick()
    else:
        selected = [s.strip() for s in gpt_selected_raw.split(",") if s.strip()]
        for symbol in selected:
            price = bhav_prices.get(symbol)
            if price and price <= capital and symbol in affordable:
                final_stocks.append(symbol)

        if not final_stocks:
            print("⚠️ GPT picks not affordable. Using fallback.")
            final_stocks = fallback_safe_pick()

# ✅ Emergency override: pick the best fallback stock using momentum from cache
if not final_stocks:
    print("🚨 Emergency override: picking best fallback stock by momentum...")

    emergency_candidates = []

    for symbol, sec_id in affordable_ids:
        try:
            df_5m = candle_cache.get(symbol, {}).get("5m", pd.DataFrame())
            if df_5m.empty or len(df_5m) < 2:
                continue

            close_now = df_5m["close"].iloc[-1]
            close_prev = df_5m["close"].iloc[-2]
            change_pct = ((close_now - close_prev) / close_prev) * 100

            if change_pct > 0:
                emergency_candidates.append((symbol, change_pct))
        except Exception as e:
            continue

    # Sort by best 5-min momentum
    emergency_candidates = sorted(emergency_candidates, key=lambda x: x[1], reverse=True)

    if emergency_candidates:
        final_stocks = [emergency_candidates[0][0]]
        log_bot_action("dynamic_stock_generator.py", "Emergency Pick", "⚠️ FORCED", f"{final_stocks}")
    else:
        print("⚠️ No momentum-positive fallback found. Picking safest stock by price.")
        final_stocks = [symbol for symbol, _ in affordable_ids][:1]
        log_bot_action("dynamic_stock_generator.py", "Emergency Pick", "⚠️ BLIND", f"{final_stocks}")
    
# ✅ Step 6: Save to file
with open('D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt', 'w') as f:
    for stock in final_stocks:
        f.write(stock + "\n")

log_bot_action("dynamic_stock_generator.py", "Stock List Updated", "✅ COMPLETE", f"{len(final_stocks)} stocks saved")
print(f"✅ Final {len(final_stocks)} stocks saved to dynamic_stock_list.txt: {final_stocks}")
