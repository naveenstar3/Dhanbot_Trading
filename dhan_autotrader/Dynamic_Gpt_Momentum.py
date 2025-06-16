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
import io
import sys
import atexit

log_buffer = io.StringIO()

# Initialize global sentiment counter
sentiment_call_count = 0

class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = TeeLogger(sys.__stdout__, log_buffer)

# ✅ Save logs automatically at end of script
def save_logs_on_exit():
    try:
        log_file_path = "D:/Downloads/Dhanbot/dhan_autotrader/Logs/Dynamic_Gpt_Momentum.txt"
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(log_buffer.getvalue())
    except Exception as e:
        print(f"⚠️ Log write failed: {e}")

atexit.register(save_logs_on_exit)

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
    global sentiment_call_count

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
            if data.get("code") == "rateLimited":
                print(f"⛔ Sentiment skipped for {symbol} due to 50-request limit. Treated as neutral/pass.")
                return True
            print(f"⚠️ News API error for {symbol}: {data}")
            return True        

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
    
    # Use EMA instead of SMA for better accuracy
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Neutral 50 when no loss
        
# ✅ Prepare Live Intraday Data
def prepare_data():
    skipped_candidates = []  # 🪣 For fallback if all others fail
    log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "START", "Preparing momentum + delivery + RSI + 15min")
    print(f"📦 Total candidates from dynamic_stock_list.csv: {len(STOCKS_TO_WATCH)}")

    records = []
    total_attempted = 0

    for symbol, secid in STOCKS_TO_WATCH:
        print(f"⏳ Processing {total_attempted+1}/{len(STOCKS_TO_WATCH)} — {symbol}")
        total_attempted += 1  # Moved up before technical checks
        
        try:
            data_5, data_15 = fetch_candle_data(symbol, secid)
            systime.sleep(0.6)
            
            if data_5 is None or data_5.empty or data_15 is None or data_15.empty:
                print(f"⚠️ Skipping {symbol}: Empty candle data (5m or 15m)")
                continue
        
            # --- 5m Metrics
            open_price = data_5['Open'].iloc[-1]
            close_price = data_5['Close'].iloc[-1]
            volume_shares = data_5['Volume'].iloc[-1]
            price = close_price
            volume_value = volume_shares * price
            
            # ✅ Calculate volume threshold AFTER getting price
            try:
                with open('D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv') as f:
                    capital = float(f.read().strip())
            except Exception as e:
                capital = 500000  # Default fallback
                print(f"⚠️ Error reading capital file: {e}. Using fallback ₹{capital:,.2f}")
            
            india = pytz.timezone("Asia/Kolkata")
            now_india = dt.datetime.now(india)
            current_hour = now_india.hour
            
            # Time-based base threshold
            if current_hour < 11:  # Before 11 AM
                base_threshold = max(150_000, capital * 0.005)
            elif current_hour < 14:  # 11 AM - 2 PM
                base_threshold = max(100_000, capital * 0.003)
            else:  # After 2 PM
                base_threshold = max(50_000, capital * 0.002)
            
            # Price adjustment
            if price > 5000:  # Large-cap stocks
                vol_threshold = base_threshold * 0.7
            elif price > 1000:
                vol_threshold = base_threshold
            else:  # Small/mid caps
                vol_threshold = base_threshold * 1.3
            
            # ❌ Skip if ₹ turnover too low
            if volume_value < vol_threshold:
                if price > 1500 and volume_value >= 30000:
                    print(f"⚠️ Low turnover ₹{volume_value:.0f} but high price ₹{price:.2f} — Accepting")
                else:
                    print(f"⛔ Skipping {symbol}: Low ₹ volume = {volume_value:.0f} < ₹{vol_threshold:.0f}")
                    continue       
                                        
            change_pct_5m = round(((close_price - open_price) / open_price) * 100, 2)
            
            # --- Run sentiment ONLY AFTER technical calculations
            if volume_value > vol_threshold and change_pct_5m > 0.1:
                sentiment_check = is_positive_sentiment(symbol, config.get("news_api_key", ""))
                if sentiment_check == "error":
                    print(f"⚠️ API limit hit - skipping sentiment for {symbol}")
                elif not sentiment_check:
                    print(f"❌ Skipping {symbol}: Negative news sentiment")
                    continue

            # --- 15m Metrics
            prev_close_15 = data_15['Close'].iloc[-2] if len(data_15) > 1 else data_15['Close'].iloc[-1]
            close_15 = data_15['Close'].iloc[-1]
            change_pct_15m = round(((close_15 - prev_close_15) / prev_close_15) * 100, 2)

            # --- Trend Strength (safe calculation)
            if len(data_15) >= 5:
                last_5_candles = data_15.tail(5)
                trend_strength = "Strong" if all(
                    last_5_candles['Close'].iloc[i] > last_5_candles['Open'].iloc[i]
                    for i in range(len(last_5_candles))
                ) else "Weak"
            else:
                trend_strength = "Insufficient Data"
                print(f"⚠️ Only {len(data_15)} candles for trend check on {symbol}")

            # --- Gap % (safe calculation)
            if len(data_5) > 1:
                prev_close_5 = data_5['Close'].iloc[-2]
                gap_pct = round(((open_price - prev_close_5) / prev_close_5) * 100, 2)
            else:
                gap_pct = 0.0  # Default if not enough data
                print(f"⚠️ Insufficient data for gap calculation on {symbol}")

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

            # --- Filter Conditions (Hard Reject Rules)
            if rsi >= 74:
                if momentum_score >= 65:
                    print(f"⚠️ RSI High ({rsi:.2f}) but Accepted due to strong score: {momentum_score}")
                else:
                    print(f"❌ Hard Reject {symbol} — RSI too high: {rsi}")
                    continue           
            if rsi < 25:
                print(f"❌ Hard Reject {symbol} — RSI too low: {rsi}")
                continue
            if delivery < 30:
                print(f"❌ Hard Reject {symbol} — Delivery too low: {delivery}%")
                continue
                        
            # Get current market time
            current_hour = now_india.hour
            
            # Set dynamic thresholds
            if current_hour < 11:  # Morning
                min_5m = 0.25
                min_15m = 0.20
            elif current_hour < 14:  # Midday
                min_5m = 0.15
                min_15m = 0.10
            else:  # Afternoon
                min_5m = 0.10
                min_15m = 0.05

            if change_pct_5m < min_5m:
                if (momentum_score >= 60 and delivery >= 30):
                    print(f"⚠️ Fallback Accept {symbol} — 5m low ({change_pct_5m:.2f}%) but score strong ({momentum_score})")
                else:
                    print(f"❌ Hard Reject {symbol} — 5m momentum too low: {change_pct_5m:.2f}% < {min_5m}%")
                    continue
            
            if change_pct_15m < min_15m:
                if (momentum_score >= 65 and change_pct_5m >= min_5m):
                    print(f"⚠️ Fallback Accept {symbol} — 15m weak but strong score+5m")
                else:
                    print(f"❌ Hard Reject {symbol} — 15m momentum too low: {change_pct_15m:.2f}% < {min_15m}%")
                    continue
            
            if trend_strength != "Strong":
                if momentum_score >= 65 and change_pct_15m >= 0.2:
                    print(f"⚠️ Trend '{trend_strength}' but Accepted — momentum strong")
                else:
                    print(f"❌ Hard Reject {symbol} — Weak trend: {trend_strength}")
                    continue                    
            
            print(f"{total_attempted}/{len(STOCKS_TO_WATCH)} ✅ Passed filters: {symbol} | Score={momentum_score}")

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
                "volume_value": volume_value,
                "tick_align_ok": tick_align_ok
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
        print("⚠️ No valid data fetched. Attempting fallback to skipped candidates...")
        
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
                "volume_value": 500000,
                "tick_align_ok": True
            }
            print(f"🛟 Fallback candidate: {best['symbol']} | ML={best['ml_score']}")
            df = pd.DataFrame([fallback_record])
        else:
            print("⚠️ No fallback candidates available")
            with open("D:/Downloads/Dhanbot/dhan_autotrader/Today_Trade_Stocks.csv", "w") as f:
                f.write("symbol,security_id\n")
            log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "❌ EMPTY", "No stocks passed filters")
            return pd.DataFrame()  # Return empty DF instead of exiting
    else:
        log_bot_action("Dynamic_Gpt_Momentum.py", "prepare_data", "✅ COMPLETE", f"{len(df)} stocks processed")

    df["score"] = df["momentum_score"]
    df["sentiment"] = "neutral"
    
    # Create security_id map from original list
    security_id_map = {symbol: secid for symbol, secid in STOCKS_TO_WATCH}
    df["security_id"] = df["symbol"].map(security_id_map)
    
    # Handle missing security IDs
    missing_ids = df[df["security_id"].isna()]
    if not missing_ids.empty:
        print(f"⚠️ Missing security IDs for: {', '.join(missing_ids['symbol'].tolist())}")
        df = df.dropna(subset=["security_id"])
    
    df.to_csv("D:/Downloads/Dhanbot/dhan_autotrader/Today_Trade_Stocks.csv", index=False)    
    print(f"📁 Exported: Today_Trade_Stocks.csv with {len(df)} records.")

    return df

    
# ✅ Ask GPT to Pick Best Stock (Hybrid Momentum + Safety Filters)
def ask_gpt_to_rank_stocks(df):
    openai.api_key = OPENAI_API_KEY
    try:
        prompt = f"""
📅 Today is {now} IST.

You are an intraday stock advisor. Analyze the filtered candidates below using hybrid momentum + fallback safety rules.

Stock Data:

{df.to_string(index=False)}

📌 Instructions:
- Select up to 25 best stocks using:
    • Prefer 5min_change_pct ≥ 0.3% (fallback OK ≥ 0.2% with strong score)
    • Prefer 15min_change_pct ≥ 0.2% (fallback OK ≥ 0.15% with score ≥ 65)
    • RSI should be < 68
    • delivery_pct ≥ 30
    • trend_strength = Strong preferred (fallback OK if score ≥ 65 or change_pct_5m ≥ 0.3)
    • volume_value > ₹5L

- Rank using: momentum_score, recent % change, safety
- Format output: RELIANCE, TCS, INFY, ...
- If none strictly match, fallback to top 5 by momentum_score where:
    • RSI < 72
    • delivery_pct ≥ 20
    • tick_align_ok = true

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
    
    print(f"✅ RSI-passed count (<70): {df[df['rsi'] < 70].shape[0]}")
    print(f"✅ Sentiment passed count: {df.shape[0]}")
    print(f"✅ GPT will now evaluate these {df.shape[0]} stocks")
    print("\n📊 Live Data:\n", df)
    print("\n🤖 Sending to GPT for analysis...\n")
    
    decision = ask_gpt_to_rank_stocks(df)
    
    # 🔁 Fallback if GPT returns SKIP
    if not decision or decision == "SKIP":
        print("⚠️ GPT returned SKIP. Attempting fallback from live filtered list...")
        fallback_df = df[
            (df["rsi"] < 74) &
            (df["delivery_pct"] >= 30) &
            (df["tick_align_ok"] == True)
        ]
        fallback_df = fallback_df.sort_values(by=["momentum_score", "delivery_pct"], ascending=False)
        if fallback_df.empty:
            print("⛔ No fallback stock found in filtered list")
            decision = []
        else:
            fallback_pick = fallback_df.iloc[0]["symbol"]
            print(f"✅ Fallback selected from live filtered data: {fallback_pick}")
            decision = [fallback_pick]
    
    print(f"\n✅ Final Selection: {decision}")
    
    # 🟢 Save GPT-final list to new CSV for autotrade
    live_df = df[df["symbol"].isin(decision)][["symbol"]].copy()
    
    # Create security_id map from original list
    security_id_map = {symbol: secid for symbol, secid in STOCKS_TO_WATCH}
    live_df["security_id"] = live_df["symbol"].map(security_id_map)
    
    # Ensure clean structure and drop duplicates
    live_df = live_df[["symbol", "security_id"]].dropna().drop_duplicates()
    
    # ✅ Save live stock list for autotrade.py
    live_df.to_csv("D:/Downloads/Dhanbot/dhan_autotrader/Today_Trade_Stocks.csv", index=False)
    print(f"✅ Saved GPT-ranked stocks to Today_Trade_Stocks.csv → {len(live_df)} stocks")
    
    try:
        summary_path = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"
        prev_summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    
        latest = {
            "total_scanned": 0,
            "affordable": 0,
            "technical_passed": 0,
            "volume_passed": 0
        }
    
        if not prev_summary.empty:
            latest_row = prev_summary.iloc[-1].to_dict()
            latest.update({k: latest_row[k] for k in latest.keys() if k in latest_row})
    
        summary_row = {
            "date": datetime.now().strftime("%m/%d/%Y %H:%M"),
            "Script_Name": "Dynamic_Gpt_Momentum.py",
            "total_scanned": latest["total_scanned"],
            "affordable": latest["affordable"],
            "technical_passed": latest["technical_passed"],
            "volume_passed": latest["volume_passed"],
            "sentiment_passed": len(df) if 'df' in locals() else 0,
            "rsi_passed": df[df["rsi"] < 70].shape[0] if 'df' in locals() and not df.empty else 0,
            "final_selected": len(live_df) if 'live_df' in locals() else 0
        }
    
        pd.DataFrame([summary_row]).to_csv(summary_path, mode='a', header=not os.path.exists(summary_path), index=False)
    
    except Exception as e:
        print(f"⚠️ Could not update filter_summary_log.csv from GPT script: {e}")

    # ✅ Always save print logs — even on early exits
    try:
        log_file_path = "D:/Downloads/Dhanbot/dhan_autotrader/Logs/Dynamic_Gpt_Momentum.txt"
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(log_buffer.getvalue())
    except Exception as e:
        print(f"⚠️ Log write failed: {e}")