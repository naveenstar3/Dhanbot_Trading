# ✅ Step 1: Fetching Broad Stock Universe
import pandas as pd
import openai
import yfinance as yf
import datetime
import pytz
import os
import requests
import json  # ✅ <- Add this here
from bs4 import BeautifulSoup


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

def fetch_live_universe():
    print("🚀 Fetching live active stocks from Yahoo...")

    all_symbols = set()

    endpoints = [
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives",
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_gainers",
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_volume_gainers"
    ]

    headers = {"User-Agent": "Mozilla/5.0"}

    for url in endpoints:
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                quotes = data.get('finance', {}).get('result', [])[0].get('quotes', [])
                for stock in quotes:
                    symbol = stock.get('symbol', '').strip().upper()
                    if symbol.endswith(".NS"):  # Only NSE stocks
                        all_symbols.add(symbol.replace(".NS", ""))
        except Exception as e:
            print(f"⚠️ Failed to fetch from {url}: {e}")

    # Convert set to sorted list
    live_symbols = sorted(list(all_symbols))
    print(f"✅ Fetched {len(live_symbols)} live NSE symbols from Yahoo.")

    return live_symbols

# Example usage:
if __name__ == "__main__":
    symbols = fetch_live_universe()
    print(symbols[:20])  # Show first 20 for debug


# ✅ Step 2: Technical Filter - Live Momentum (Medium Strictness)

def technical_filter(symbols):
    print("🚀 Starting Technical Momentum Filter...")
    passed_symbols = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol + ".NS")
            hist = ticker.history(period="1d", interval="5m")

            if hist.empty or len(hist) < 4:
                continue  # Not enough data points (need at least 3 candles)

            # Calculate 5-min and 15-min momentum
            latest_price = hist['Close'].iloc[-1]
            price_5min_ago = hist['Close'].iloc[-2]
            price_15min_ago = hist['Close'].iloc[-4]

            momentum_5min = ((latest_price - price_5min_ago) / price_5min_ago) * 100
            momentum_15min = ((latest_price - price_15min_ago) / price_15min_ago) * 100

            # Medium strictness check
            if 0.2 <= momentum_5min <= 0.6 and momentum_15min > 0:
                passed_symbols.append(symbol)

            # Sleep 1 sec between calls to avoid rate-limits
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ Error fetching data for {symbol}: {e}")
            continue

    print(f"✅ {len(passed_symbols)} symbols passed momentum filter.")
    return passed_symbols

# Example usage (after fetch_live_universe step):
if __name__ == "__main__":
    live_symbols = fetch_live_universe()  # Step 1 output
    final_symbols = technical_filter(live_symbols)
    print(final_symbols[:20])  # Show first few for debug
    
    
# ✅ Step 3: Volume Surge Filter

def volume_surge_filter(symbols):
    print("🚀 Starting Volume Surge Filter...")
    filtered_symbols = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol + ".NS")
            hist = ticker.history(period="5d", interval="1d")

            if hist.empty or len(hist) < 5:
                continue

            avg_volume_5d = hist['Volume'].iloc[:-1].mean()  # exclude today's candle
            today_volume = hist['Volume'].iloc[-1]

            if today_volume >= 1.2 * avg_volume_5d:
                filtered_symbols.append(symbol)

            # Sleep to avoid API abuse
            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Error checking volume for {symbol}: {e}")
            continue

    print(f"✅ {len(filtered_symbols)} symbols passed volume surge filter.")
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

# Example usage (after volume_surge_filter step):
if __name__ == "__main__":
    live_symbols = fetch_live_universe()
    technical_passed = technical_filter(live_symbols)
    volume_passed = volume_surge_filter(technical_passed)
    sentiment_passed = sentiment_filter(volume_passed)
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
        response = openai.chat.completions.create(
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

    if not sentiment_passed:
        print("⚠️ No stocks survived after sentiment check. Using fallback.")
        sentiment_passed = fallback_safe_list  # Hardcoded safe list

    # Prepare dataframe for GPT input
    df = pd.DataFrame(sentiment_passed, columns=["symbol"])

    # Send to GPT
    gpt_selected_raw = ask_gpt_to_pick_stocks(df)

    if gpt_selected_raw == "SKIP":
        print("⚠️ GPT advised SKIP. Using fallback safe stocks.")
        final_stocks = fallback_safe_list
    else:
        final_stocks = [s.strip() for s in gpt_selected_raw.split(",")]

    # ✅ Save final result to dynamic_stock_list.txt
    with open('D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt', 'w') as f:
        for stock in final_stocks:
            f.write(stock + "\n")

    print(f"✅ Final {len(final_stocks)} stocks saved to dynamic_stock_list.txt.")

