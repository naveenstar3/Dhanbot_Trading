# ✅ Step 1: Fetching Broad Stock Universe
import pandas as pd
import openai
import yfinance as yf
import datetime
import pytz
import os
import requests
import json  # ✅ <- Add this here
import time as systime
import csv
from bs4 import BeautifulSoup
from dhan_api import get_historical_price, get_security_id
from dhan_api import get_live_price, get_current_capital 
from concurrent.futures import ThreadPoolExecutor, as_completed
from dhan_api import get_security_id_from_trading_symbol

# ✅ Load config.json (OpenAI Key inside)
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

OPENAI_API_KEY = config["openai_api_key"]

# ✅ Fallback Safe Stock List (Expanded)

fallback_safe_list = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "LT", "SBIN", "AXISBANK", "ITC", "KOTAKBANK",
    "BHARTIARTL", "HCLTECH", "WIPRO", "BAJFINANCE", "ADANIENT",
    "POWERGRID", "HINDUNILVR", "ASIANPAINT", "ULTRACEMCO", "MARUTI",
    "DIVISLAB", "TITAN", "SUNPHARMA", "TECHM", "NESTLEIND",
    "JSWSTEEL", "COALINDIA", "BPCL", "BRITANNIA", "GRASIM",
    "ONGC", "INDUSINDBK", "EICHERMOT", "HDFCLIFE", "HEROMOTOCO"
]

def threaded_get_price(symbol, interval, limit):
    try:
        security_id = get_security_id(symbol)
        candles = get_historical_price(security_id, interval=interval, limit=limit)
        return symbol, candles
    except Exception as e:
        print(f"⚠️ Error fetching candles for {symbol}: {e}")
        return symbol, None

def get_current_capital():
    try:
        with open("D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv", "r") as f:
            return float(f.read().strip())
    except:
        return 3000  # default fallback capital

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_live_universe(limit=500):
    print("📂 Filtering NSE equity stocks from Dhan master file...")
    symbols = []

    try:
        with open("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                exchange = row.get("SEM_EXM_EXCH_ID", "").strip().upper()
                segment = row.get("SEM_SEGMENT", "").strip().upper()
                instrument = row.get("SEM_INSTRUMENT_NAME", "").strip().upper()
                symbol_name = row.get("SM_SYMBOL_NAME", "").strip().upper()

                if (
                    exchange == "NSE" and
                    segment == "E" and
                    instrument == "EQUITY" and
                    symbol_name and
                    all(x not in symbol_name for x in ["SDL", "NCD", "GOI", "AMC", "DEBENTURE", "BOND", "FMP"])
                ):
                    symbols.append(symbol_name)

    except Exception as e:
        print(f"⚠️ Failed to read dhan_master.csv: {e}")

    symbols = list(set(symbols))[:limit]  # limit to first 500
    print(f"✅ Loaded {len(symbols)} NSE equity stocks from dhan_master.csv.")
    return symbols

# Example usage:
if __name__ == "__main__":
    symbols = fetch_live_universe()
    print(symbols[:20])  # Show first 20 for debug


# ✅ Step 2: Technical Filter - Live Momentum (Medium Strictness)

def technical_filter(symbols):
    print("🚀 Starting Technical Momentum Filter using Dhan candles...")
    passed_symbols = []

    def evaluate_momentum(symbol, candles):
        try:
            if not candles or len(candles) < 5:
                return None

            df = pd.DataFrame(candles)
            if df.empty or len(df) < 5:
                return None

            latest = df['close'].iloc[-1]
            five_min_ago = df['close'].iloc[-2]
            fifteen_min_ago = df['close'].iloc[-4]

            momentum_5min = ((latest - five_min_ago) / five_min_ago) * 100
            momentum_15min = ((latest - fifteen_min_ago) / fifteen_min_ago) * 100

            if 0.2 <= momentum_5min <= 0.6 and momentum_15min > 0:
                return symbol
        except Exception as e:
            print(f"⚠️ Error in momentum evaluation for {symbol}: {e}")
        return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(threaded_get_price, symbol, "5m", 15): symbol for symbol in symbols
        }

        for future in as_completed(futures):
            symbol = futures[future]
            sym, candles = future.result()
            result = evaluate_momentum(sym, candles)
            if result:
                passed_symbols.append(result)

    print(f"✅ {len(passed_symbols)} passed technical momentum filter.")
    return passed_symbols

# Example usage (after fetch_live_universe step):
if __name__ == "__main__":
    live_symbols = fetch_live_universe()  # Step 1 output
    final_symbols = technical_filter(live_symbols)
    print(final_symbols[:20])  # Show first few for debug
    
    
# ✅ Step 3: Volume Surge Filter

def volume_surge_filter(symbols):
    print("🚀 Starting Volume Surge Filter using Dhan 1D candles...")
    filtered_symbols = []

    for symbol in symbols:
        try:
            security_id = get_security_id(symbol)
            if not security_id:
                continue

            candles = get_historical_price(security_id, interval="1d", limit=6)
            if not candles or len(candles) < 5:
                continue

            df = pd.DataFrame(candles)
            if df.empty or len(df) < 5:
                continue

            avg_volume_5d = df['volume'].iloc[:-1].mean()
            today_volume = df['volume'].iloc[-1]

            if today_volume >= 1.2 * avg_volume_5d:
                filtered_symbols.append(symbol)

            systime.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Volume check failed for {symbol}: {e}")
            continue

    print(f"✅ {len(filtered_symbols)} passed volume surge filter.")
    return filtered_symbols

# Example usage (after technical_filter step):
if __name__ == "__main__":
    live_symbols = fetch_live_universe()  # Step 1 output
    technical_passed = technical_filter(live_symbols)  # Step 2
    volume_passed = volume_surge_filter(technical_passed)
    print(volume_passed[:20])  # Preview


# ✅ Step 4: News Sentiment Check

def sentiment_filter(symbols):
    print("📰 Starting News Sentiment Screening...")
    clean_symbols = []

    negative_keywords = [
        "fraud", "scam", "raid", "penalty", "default",
        "problem", "loss", "downgrade", "resignation",
        "fire", "lawsuit", "debt", "bankruptcy", "fine"
    ]

    for symbol in symbols:
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=c545f9478aab45bd9886110793d08bdb&sortBy=publishedAt&pageSize=5"
            response = requests.get(url)

            if response.status_code == 200:
                articles = response.json().get("articles", [])

                negative_found = False
                for article in articles:
                    title = article.get("title", "").lower()
                    description = article.get("description", "").lower()
                    combined_text = title + " " + description

                    if any(word in combined_text for word in negative_keywords):
                        negative_found = True
                        break

                if not negative_found:
                    clean_symbols.append(symbol)

            time.sleep(0.5)  # Sleep to avoid API limit abuse

        except Exception as e:
            print(f"⚠️ NewsAPI error for {symbol}: {e}")
            continue

    print(f"✅ {len(clean_symbols)} symbols passed News Sentiment Check.")
    return clean_symbols

def rsi_filter(symbols):
    print("📉 Starting RSI filter (RSI ≤ 70)...")
    safe_symbols = []

    for symbol in symbols:
        try:
            security_id = get_security_id(symbol)
            if not security_id:
                continue

            candles = get_historical_price(security_id, interval="5m", limit=20)
            if not candles or len(candles) < 15:
                continue

            df = pd.DataFrame(candles)
            if df.empty or len(df) < 15:
                continue

            rsi_series = calculate_rsi(df['close'])
            latest_rsi = rsi_series.iloc[-1]

            if latest_rsi <= 70:
                safe_symbols.append(symbol)
            else:
                print(f"⛔ {symbol} RSI too high: {round(latest_rsi, 2)}")

            systime.sleep(0.5)

        except Exception as e:
            print(f"⚠️ RSI check failed for {symbol}: {e}")
            continue

    print(f"✅ {len(safe_symbols)} symbols passed RSI filter.")
    return safe_symbols

# Example usage (after volume_surge_filter step):
if __name__ == "__main__":
    live_symbols = fetch_live_universe()
    technical_passed = technical_filter(live_symbols)
    volume_passed = volume_surge_filter(technical_passed)
    sentiment_passed = sentiment_filter(volume_passed)
    rsi_passed = rsi_filter(sentiment_passed)
    print(sentiment_passed[:20])  # Preview

# ✅ GPT Final Stock Selection

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
        response = openai.chat.completions.create(  # ✅ NEW for openai>=1.0.0
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        decision = response.choices[0].message.content.strip().upper()
        return decision
    except Exception as e:
        print(f"⚠️ Error with GPT selection: {e}")
        return "SKIP"
# ✅ Full Assemble Now

if __name__ == "__main__":
    live_symbols = fetch_live_universe()
    technical_passed = technical_filter(live_symbols)
    volume_passed = volume_surge_filter(technical_passed)
    sentiment_passed = sentiment_filter(volume_passed)
    rsi_passed = rsi_filter(sentiment_passed)

    if not rsi_passed:
        print("⚠️ No stocks survived after RSI check. Using fallback.")
        rsi_passed = fallback_safe_list  # Use fallback if list is empty

    # ✅ Build GPT input DataFrame with 5min_change_pct and volume_value
    df = pd.DataFrame(columns=["symbol", "5min_change_pct", "volume_value"])

    for symbol in rsi_passed:
        try:
            security_id = get_security_id_from_trading_symbol(symbol)  # Use correct symbol match
            if not security_id:
                print(f"⛔ No security_id for: {symbol}")
                continue
    
            candles_5m = get_historical_price(security_id, interval="5m", limit=5)
            candles_1d = get_historical_price(security_id, interval="1d", limit=6)
    
            df_5m = pd.DataFrame(candles_5m)
            df_1d = pd.DataFrame(candles_1d)
    
            if df_5m.empty:
                print(f"❌ {symbol} — Empty 5m candles")
                continue
            if df_1d.empty:
                print(f"❌ {symbol} — Empty 1D candles")
                continue
    
            print(f"✅ {symbol} — 5m: {len(df_5m)}, 1D: {len(df_1d)}")
    
            change_pct = ((df_5m["close"].iloc[-1] - df_5m["close"].iloc[-2]) / df_5m["close"].iloc[-2]) * 100
            vol_value = df_1d["volume"].iloc[-1] * df_5m["close"].iloc[-1]
    
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
            
        # 🏦 Capital check
        capital = get_current_capital()
        affordable = []
        
        print(f"💰 Available Capital: ₹{capital}")
        
        for symbol in rsi_passed:
            try:
                price = get_live_price(symbol)
                if price and price <= capital:
                    affordable.append(symbol)
                else:
                    print(f"⛔ {symbol} skipped due to price ₹{price} > ₹{capital}")
            except Exception as e:
                print(f"⚠️ Failed price check for {symbol}: {e}")
                continue
        
        rsi_passed = affordable
        print(f"✅ {len(rsi_passed)} affordable stocks under ₹{capital}")
            
    # ✅ Final safety check before calling GPT
    expected_cols = ["symbol", "5min_change_pct", "volume_value"]
    if df.empty or not all(col in df.columns for col in expected_cols):
        print("⚠️ Skipping GPT call due to invalid DataFrame structure.")
        print(f"❌ Current DataFrame Columns: {list(df.columns)}")
        print(f"❌ Expected Columns: {expected_cols}")
        gpt_selected_raw = "SKIP"
    else:
        print("📊 GPT DataFrame structure valid. Proceeding with GPT selection...")
        print(df.head(5))  # Optional preview of GPT input
        gpt_selected_raw = ask_gpt_to_pick_stocks(df)
    
        
    if gpt_selected_raw == "SKIP":
        print("⚠️ GPT advised SKIP. Using fallback safe stocks.")
        final_stocks = fallback_safe_list
    else:
        selected = [s.strip() for s in gpt_selected_raw.split(",")]
        final_stocks = [s for s in selected if s]

        # ✅ Budget-friendly fallback
        affordable = []
        for symbol in final_stocks:
            try:
                price = get_live_price(symbol)
                capital = get_current_capital()
                if price * 1 <= capital:
                    affordable.append(symbol)
            except:
                continue

        if not affordable:
            print("⚠️ No affordable picks from GPT. Using fallback safe list.")
            affordable = fallback_safe_list[:3]

        final_stocks = affordable

    # ✅ Save final result
    with open('D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt', 'w') as f:
        for stock in final_stocks:
            f.write(stock + "\n")

    print(f"✅ Final {len(final_stocks)} stocks saved to dynamic_stock_list.txt.")

