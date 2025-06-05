
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
import pytz
import json
import os
import time as systime
import csv
import requests
from utils_logger import log_bot_action
import datetime as dt
from textblob import TextBlob
import time
from datetime import datetime, timedelta

now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
# ✅ Load config.json (OpenAI Key inside)
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

OPENAI_API_KEY = config["openai_api_key"]

# ✅ Load Dynamic Stock List
def load_dynamic_stocks():
    try:
        stocks = []
        with open('D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row["symbol"].strip().upper()
                secid = row["security_id"].strip()
                stocks.append((symbol, secid))
        return stocks
    except Exception as e:
        print(f"⚠️ Error loading stock list: {e}")
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

def is_positive_sentiment(symbol, api_key):
    """
    Fetch recent news articles for the stock and evaluate average sentiment.
    Applies API call throttling and respects a 50-call rate limit.
    """
    # Global API call limiter
    global sentiment_call_count
    if 'sentiment_call_count' not in globals():
        sentiment_call_count = 0

    if sentiment_call_count >= 50:
        print(f"⛔ Sentiment skipped for {symbol} due to 50-request limit. Treated as neutral/pass.")
        return True

    to_date = datetime.now()
    from_date = to_date - timedelta(days=3)
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&"
        f"from={from_date_str}&"
        f"to={to_date_str}&"
        f"language=en&"
        f"sortBy=relevancy&"
        f"apiKey={api_key}"
    )

    try:
        response = requests.get(url)
        data = response.json()

        # ✅ Count only successful API hits
        sentiment_call_count += 1
        time.sleep(0.2)

        if data.get("status") != "ok":
            print(f"⚠️ News API error for {symbol}: {data}")
            return True  # Fail open

        articles = data.get("articles", [])
        if not articles:
            print(f"⚠️ No news articles found for {symbol}. Treating as neutral.")
            return True  # ✅ Allow it if no news

        sentiments = []
        for article in articles:
            content = f"{article.get('title', '')} {article.get('description', '')}"
            blob = TextBlob(content)
            sentiments.append(blob.sentiment.polarity)

        avg_sentiment = sum(sentiments) / len(sentiments)
        print(f"📰 {symbol} - Avg Sentiment Polarity: {avg_sentiment:.2f}")
        return avg_sentiment >= 0.05

    except Exception as e:
        print(f"❌ News fetch error for {symbol}: {e}")
        return False


# ✅ Fetch Recent 5-min and 15-min Candles
def fetch_candle_data(symbol, security_id=None):
    try:
        if not security_id:
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
        india = pytz.timezone("Asia/Kolkata")
        now = dt.datetime.now(india)
        from_dt = (now - dt.timedelta(days=2)).replace(hour=9, minute=15, second=0, microsecond=0)
        to_dt = now

        def fetch(interval):
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
                    "Timestamp": pd.to_datetime(response_json["timestamp"], unit='s')
                                    .tz_localize('UTC')
                                    .tz_convert('Asia/Kolkata')
                })
                return df
            except Exception as e:
                print(f"⚠️ Failed parsing response for {symbol} [{interval}]: {e}")
                return None

        # ✅ Only use 5MIN + 15MIN (no 1MIN fallback)
        data_5 = fetch("5MIN")
        data_15 = fetch("15MIN")

        if data_5 is not None:
            print(f"✅ Used 5MIN + 15MIN data for {symbol}")
        else:
            print(f"❌ No 5MIN data available for {symbol}")

        return data_5, data_15

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
    skipped_candidates = []  # 🪣 For fallback if all others fail
    log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "START", "Preparing momentum + delivery + RSI + 15min")
    print(f"📦 Total candidates from dynamic_stock_list.csv: {len(STOCKS_TO_WATCH)}")

    records = []
    total_attempted = 0

    for symbol, secid in STOCKS_TO_WATCH:
        print(f"⏳ Processing {total_attempted+1}/{len(STOCKS_TO_WATCH)} — {symbol}")
        # ✅ Apply sentiment filter before expensive API calls
        if not is_positive_sentiment(symbol, config.get("news_api_key", "")):
            print(f"❌ Skipping {symbol}: Negative news sentiment")
            continue

        total_attempted += 1

        try:
            data_5, data_15 = fetch_candle_data(symbol, secid)
            time.sleep(0.6)  # Throttle API requests to prevent rate limiting

            if data_5 is None or data_5.empty or data_15 is None or data_15.empty:
                print(f"⚠️ Skipping {symbol}: Empty candle data (5m or 15m)")
                continue

            # --- 5m Metrics
            open_price = data_5['Open'].iloc[-1]
            close_price = data_5['Close'].iloc[-1]
            volume_value = data_5['Volume'].iloc[-1]
            # ❌ Skip if recent volume is too low to be meaningful
            capital = float(open('D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv').read().strip())
            vol_threshold = max(1000, int(capital / 100))
            if volume_value < vol_threshold:
                print(f"⛔ Skipping {symbol}: Low 5-min volume = {volume_value} < {vol_threshold}")
                continue                   
            change_pct_5m = round(((close_price - open_price) / open_price) * 100, 2)

            # --- 15m Metrics
            prev_close_15 = data_15['Close'].iloc[-2] if len(data_15) > 1 else data_15['Close'].iloc[-1]
            close_15 = data_15['Close'].iloc[-1]
            change_pct_15m = round(((close_15 - prev_close_15) / prev_close_15) * 100, 2)

            last_5_candles = data_15.tail(5)
            trend_strength = "Strong" if all(
                last_5_candles['Close'].iloc[i] > last_5_candles['Open'].iloc[i]
                for i in range(len(last_5_candles))
            ) else "Weak"

            # --- Gap %
            prev_close_5 = data_5['Close'].iloc[-2]
            gap_pct = round(((open_price - prev_close_5) / prev_close_5) * 100, 2)

            # --- RSI
            rsi_series = calculate_rsi(data_15['Close'])
            rsi = round(rsi_series.iloc[-1], 2) if not rsi_series.empty else 0

            # --- Delivery %
            delivery = get_delivery_percentage(symbol)

            # --- Momentum Score (Hybrid)
            score = (
                change_pct_5m * 0.5 +
                change_pct_15m * 0.3 +
                gap_pct * 0.15 +
                delivery * 0.2 +
                (volume_value / 100000) * 0.05 +
                (1.0 if trend_strength == "Strong" else 0)
            )
            momentum_score = round(score, 2)

            # --- Filter Conditions
            reason = []
            if delivery < 30:
                reason.append(f"Delivery={delivery}%")
            if rsi > 70:
                reason.append(f"RSI={rsi:.2f}")
            if gap_pct > 5:
                reason.append(f"Gap={gap_pct:.2f}%")
            if change_pct_5m < 0.05:
                reason.append(f"5m={change_pct_5m:.2f}%")
            if change_pct_15m < 0.02:
                reason.append(f"15m={change_pct_15m:.2f}%")            
            
            print(f"{total_attempted}/{len(STOCKS_TO_WATCH)} ❌ Skipped {symbol}: " + ", ".join(reason))

            # ⚠️ Soft Fallback Logic: Relaxed to support low-volatility market days
            if change_pct_5m >= 0.01 and change_pct_15m >= 0.005 and rsi < 74 and delivery >= 20:
                print(f"⚠️ Soft pass {symbol} due to fallback conditions")
            else:
                continue
            

            # --- Final Record
            tick_size = 0.05
            buffer_price = round(close_price * 1.002, 2)
            tick_align_ok = round(buffer_price % tick_size, 2) == 0

            stock_snapshot = {
                "symbol": symbol,
                "close_price": close_price,
                "buffer_price": buffer_price,
                "rsi": rsi,
                "momentum_5m": change_pct_5m,
                "momentum_15m": change_pct_15m,
                "ml_score": momentum_score,
                "sentiment_score": 1.0,
                "tick_align_ok": tick_align_ok
            }
            skipped_candidates.append(stock_snapshot)

            if not tick_align_ok:
                print(f"⚠️ Tick misalign ignored: {symbol} @ {buffer_price}")           

            record = {
                "symbol": symbol,
                "5min_change_pct": change_pct_5m,
                "15min_change_pct": change_pct_15m,
                "gap_pct": gap_pct,
                "delivery_pct": delivery,
                "rsi": rsi,
                "trend_strength": trend_strength,
                "momentum_score": momentum_score,
                "volume_value": volume_value
            }
            records.append(record)
            print(f"{total_attempted}/{len(STOCKS_TO_WATCH)} ✅ Added: {symbol} | Score={momentum_score}")
            systime.sleep(1.2)

        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e.__class__.__name__} - {str(e)}")
            continue

    # ✅ Create final DataFrame and log summary
    df = pd.DataFrame(records)
    print(f"📊 Completed: {len(records)}/{total_attempted} passed filters")

    if df.empty:
        print("⚠️ No valid data fetched. Clearing stale live_stocks_trade_today.csv")
        with open("D:/Downloads/Dhanbot/dhan_autotrader/live_stocks_trade_today.csv", "w") as f:
            f.write("symbol,security_id\n")  # Empty header
        log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "❌ NO STOCKS", "Cleared stale trade list")
        exit(0)   

        fallback = sorted(
            [s for s in skipped_candidates if s["tick_align_ok"] and s["rsi"] < 75],
            key=lambda x: (x["ml_score"], x["momentum_5m"] + x["momentum_15m"]),
            reverse=True
        )

        if fallback:
            best = fallback[0]
            fallback_record = {
                "symbol": best["symbol"],
                "5min_change_pct": best["momentum_5m"],
                "15min_change_pct": best["momentum_15m"],
                "gap_pct": 0,
                "delivery_pct": 35,
                "rsi": best["rsi"],
                "trend_strength": "Weak",
                "momentum_score": best["ml_score"],
                "volume_value": 500000
            }
            print(f"🛟 Fallback candidate: {best['symbol']} | ML={best['ml_score']}")
            df = pd.DataFrame([fallback_record])
        else:
            log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "❌ EMPTY", "No stocks passed filters")
            return df
    else:
        log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "✅ COMPLETE", f"{len(df)} stocks processed")

    df["score"] = df["momentum_score"]
    df["sentiment"] = "neutral"
    map_df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv")
    df = pd.merge(df, map_df[["symbol", "security_id"]], on="symbol", how="left")
    df.to_csv("D:/Downloads/Dhanbot/dhan_autotrader/Today_Trade_Stocks.csv", index=False)    
    print(f"📁 Exported: Today_Trade_Stocks.csv with {len(df)} records.")

    return df

    
# ✅ Ask GPT to Pick Best Stock (Hybrid Momentum + Safety Filters)
def ask_gpt_to_rank_stocks(df):
    openai.api_key = OPENAI_API_KEY
    try:
        prompt = f"""
📅 Today is {now} IST.

You are an intraday stock advisor. Analyze the filtered candidates below using strict hybrid momentum criteria.

Stock Data:

{df.to_string(index=False)}

📌 Instructions:
- Rank stocks (up to 15) using:
    • 5min_change_pct > 0.5%
    • 15min_change_pct > 0.3%
    • RSI < 68
    • delivery_pct ≥ 30
    • trend_strength must be "Strong"
    • volume_value > ₹5L
- Prefer higher momentum_score
- Format: RELIANCE, TCS, INFY, ...
- If **none** qualify, respond only with: SKIP
"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        gpt_response = response.choices[0].message.content.strip().upper()

        if gpt_response == "SKIP" or not gpt_response:
            fallback = df.sort_values("momentum_score", ascending=False).head(5)["symbol"].tolist()
            log_bot_action("Dynamic_Gpt_Momentum.py", "ask_gpt_to_rank_stocks", "⚠️ GPT SKIP → FORCED PICK", f"Fallback: {fallback}")
            return fallback

        candidates = [s.strip() for s in gpt_response.split(",") if s.strip() in df["symbol"].values]

        if not candidates:
            fallback = df.sort_values("momentum_score", ascending=False).head(5)["symbol"].tolist()
            log_bot_action("Dynamic_Gpt_Momentum.py", "ask_gpt_to_rank_stocks", "⚠️ GPT BAD → FORCED PICK", f"Fallback: {fallback}")
            return fallback

        log_bot_action("Dynamic_Gpt_Momentum.py", "ask_gpt_to_rank_stocks", "✅ GPT RANKED", f"{candidates}")
        return candidates

    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        fallback = df.sort_values("momentum_score", ascending=False).head(5)["symbol"].tolist()
        log_bot_action("Dynamic_Gpt_Momentum.py", "ask_gpt_to_rank_stocks", "⚠️ GPT FAIL → FORCED PICK", f"Fallback: {fallback}")
        return fallback
        
# ✅ Check if Market is Open
def is_market_open():
    now = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
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

    if df.empty or df["symbol"].nunique() == 0:
        print("⚠️ No valid data to analyze. Skipping GPT call.")
        log_bot_action("Dynamic_Gpt_Momentum.py", "main", "❌ SKIPPED", "No valid records to rank")
        exit(0)

    print("\n📊 Live Data:\n", df)
    print("\n🤖 Sending to GPT for analysis...\n")
    decision = ask_gpt_to_rank_stocks(df)

    print(f"\n✅ GPT Decision: {decision}")

    # 🟢 Save GPT-final list to new CSV for autotrade
    live_df = df[df["symbol"].isin(decision)][["symbol"]].copy()

    # Reload original security_id map from dynamic_stock_list.csv
    map_df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv")
    live_df = pd.merge(live_df, map_df, on="symbol", how="left")

    # Ensure clean structure and drop duplicates
    live_df = live_df[["symbol", "security_id"]].dropna().drop_duplicates()

    # ✅ Save live stock list for autotrade.py
    live_df.to_csv("D:/Downloads/Dhanbot/dhan_autotrader/live_stocks_trade_today.csv", index=False)
    print(f"✅ Saved GPT-ranked stocks to live_stocks_trade_today.csv → {len(live_df)} stocks")
        
