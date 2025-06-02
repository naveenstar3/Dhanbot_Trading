# ✅ PART 1: Imports and Config Setup
import os
import sys
import json
import time as systime
import pandas as pd
import requests
from datetime import datetime, timedelta
from dhan_api import get_historical_price, get_current_capital
from utils_logger import log_bot_action
import openai
from textblob import TextBlob
from datetime import datetime, timedelta
import time
import pytz

start_time = datetime.now()

def is_trading_day():
    # Skip weekends
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    if today.weekday() >= 5:
        return False

    # Load or download holiday list
    year = today.year
    holiday_file = f"nse_holidays_{year}.csv"

    if not os.path.exists(holiday_file):
        print(f"📥 Downloading NSE holiday calendar for {year}...")
        url = f"https://www.nseindia.com/api/holiday-master?type=trading"
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        try:
            r = session.get(url, timeout=10)
            holidays = r.json()["Trading"]
            dates = [datetime.strptime(day["date"], "%d-%b-%Y").date() for day in holidays if str(year) in day["date"]]
            pd.DataFrame({"date": dates}).to_csv(holiday_file, index=False)
        except Exception as e:
            print("⚠️ Could not fetch NSE holiday calendar:", e)
            return True  # fallback to allow run

    try:
        holiday_df = pd.read_csv(holiday_file)
        holiday_dates = pd.to_datetime(holiday_df["date"]).dt.date.tolist()
        return today not in holiday_dates
    except:
        return True  # fallback: assume open

if not is_trading_day():
    print("⛔ Today is a non-trading day. Skipping stock generation.")
    log_bot_action("dynamic_stock_generator.py", "market_status", "INFO", "Skipped: NSE holiday/weekend.")
    sys.exit(0)


with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
openai.api_key = config.get("openai_api_key")
NEWS_API_KEY = config.get("news_api_key", "")
capital_path = "D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv"
CAPITAL = float(pd.read_csv(capital_path, header=None).iloc[0, 0])


HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

FINAL_STOCK_LIMIT = 100
PREMARKET_MODE = True

# ✅ PART 2: Load filtered weekly universe CSV (already volume + affordability passed)
universe_path = "D:/Downloads/Dhanbot/dhan_autotrader/weekly_affordable_volume_filtered.csv"

try:
    filtered_df = pd.read_csv(universe_path)
except Exception as e:
    print(f"❌ Failed to read universe CSV: {e}")
    sys.exit(1)

if "symbol" not in filtered_df.columns or "security_id" not in filtered_df.columns:
    print("❌ Required columns 'symbol' and 'security_id' not found in weekly CSV.")
    sys.exit(1)

# 💡 If test mode is active
test_security_id = None
if len(sys.argv) > 1:
    test_security_id = str(sys.argv[1]).strip()
    print(f"🔬 Test Mode: Single Security ID = {test_security_id}")
    filtered_df = filtered_df[filtered_df["security_id"].astype(str) == test_security_id]
    if filtered_df.empty:
        print(f"❌ Security ID {test_security_id} not found in filtered universe.")
        sys.exit(1)

    symbol = filtered_df.iloc[0]["symbol"]

    # Run sentiment filter even in test mode
    sentiment_passed = 1 if is_positive_sentiment(symbol, NEWS_API_KEY) else 0
    print(f"📰 Sentiment Pass: {sentiment_passed}")

    # Write single stock to CSV only if sentiment passes
    if sentiment_passed:
        output_file = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"
        with open(output_file, "w") as f:
            f.write("symbol,security_id\n")
            f.write(f"{symbol},{test_security_id}\n")
        print(f"✅ Saved test-mode stock to {output_file}")
    else:
        print(f"⛔ Stock {symbol} skipped due to negative sentiment.")
        print(f"❌ {symbol} Skipped due to sentiment [1/1]")  # Manual index for test mode

    # Log to filter_summary_log.csv
    summary_log_path = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"
    now = datetime.now().strftime("%m/%d/%Y %H:%M")
    summary_data = {
        "date": now,
        "total_scanned": 1,
        "affordable": 1,
        "technical_passed": 1,
        "volume_passed": CAPITAL,
        "sentiment_passed": sentiment_passed,
        "rsi_passed": "",  # Not checked here
        "final_selected": 1 if sentiment_passed else 0
    }
    try:
        if os.path.exists(summary_log_path):
            existing_df = pd.read_csv(summary_log_path)
            existing_df = pd.concat([existing_df, pd.DataFrame([summary_data])], ignore_index=True)
            existing_df.to_csv(summary_log_path, index=False)
        else:
            pd.DataFrame([summary_data]).to_csv(summary_log_path, index=False)
        print("📝 Logged test summary to filter_summary_log.csv")
    except Exception as e:
        print(f"⚠️ Could not log summary: {e}")
    sys.exit(0)


# 🔄 Convert to list of tuples for loop
final_filtered_list = list(zip(filtered_df["symbol"].astype(str), filtered_df["security_id"].astype(str)))

# ✅ PART 3: Scoring and Ranking Using GPT (Unchanged Logic)
final_stocks = []
sentiment_passed = 0
rsi_passed = 0  # Not implemented, keep 0 for now

print("🔍 Scanning filtered universe for scoring...")
# ✅ PART 3: Scoring and Ranking Using GPT (Unchanged Logic)
final_stocks = []
sentiment_passed = 0
rsi_passed = 0  # Not implemented, keep 0 for now

print("🔍 Scanning filtered universe for scoring...")

for idx, (symbol, secid) in enumerate(final_filtered_list, start=1):
    try:
        avg_volume = float(filtered_df.loc[filtered_df["symbol"] == symbol, "avg_volume"].values[0])
        atr = float(filtered_df.loc[filtered_df["symbol"] == symbol, "atr"].values[0])
        ltp = float(filtered_df.loc[filtered_df["symbol"] == symbol, "ltp"].values[0])

        score = (
            (avg_volume / 1e6) * 0.5 +
            (ltp / 1000) * 0.25 +
            (atr / 5) * 0.25
        )

        print(f"\n🔎 [{idx}/{len(final_filtered_list)}] {symbol}")
        print(f"    • ✅ Volume OK: {avg_volume/1e6:.2f}M")
        print(f"    • ✅ Affordable: ₹{ltp:.2f} vs ₹{CAPITAL/FINAL_STOCK_LIMIT:.2f}")
        print(f"    • ✅ ATR: {atr}")
        print(f"    • ✅ Score: {score:.2f} → Selected")

        final_stocks.append((symbol, secid, round(score, 3)))
        time.sleep(0.1)

    except Exception as e:
        print(f"\n❌ [{idx}/{len(final_filtered_list)}] {symbol}")
        print(f"    • ⚠️ Skipped due to: {e}")
        continue


final_count = len(final_stocks)
if len(final_stocks) == 0:
    print("❌ No stocks passed scoring.")
    sys.exit(1)

# ✅ Sort and take top 150 to pass to GPT
top_stocks = sorted(final_stocks, key=lambda x: x[2], reverse=True)[:150]

# ✅ Format as GPT prompt
gpt_prompt = f"""
🧠 You are an expert intraday stock selector and market analyst. Today is {datetime.now().strftime('%Y-%m-%d')} IST.

You have been given a list of pre-screened stocks, each with a momentum score based on:
- 5-day average volume
- ATR (intraday movement potential)
- Stock price
- Recent EOD uptrend pattern
- Strong close signals

Your mission is to select **exactly {FINAL_STOCK_LIMIT} stocks** that are the **safest and most reliable intraday candidates** for today’s session.

📌 Important Instructions:
- These stocks must have the **highest probability** of passing **live filters** like RSI < 70, 5-min/15-min uptrend, and positive sentiment during market hours.
- Since intraday data is not available yet, base your ranking on **historical behavior that predicts future strength**, such as:
  - Recent consistent bullishness or strong close
  - High delivery % and volume consistency
  - Clean technical structure (e.g., breakout readiness)
  - Sector strength and market alignment
  - No red-flag news or corporate issues
  - Avoid SME/PSU/penny/junk stocks

⚠️ You are strictly accountable for the quality of your picks. At least **10% of these selected stocks MUST pass** real-time intraday momentum filters in the next step.
- Do not include weak, random, or speculative stocks just to fill the count.
- Prioritize quality over diversity.

🎯 Output format: Just the stock symbols, comma-separated, no explanation.

Candidate Stocks with Scores:
{", ".join([f"{s[0]}:{s[2]}" for s in top_stocks])}
"""

print("🤖 Asking GPT to rank top 50...")
client = openai.OpenAI(api_key=config.get("openai_api_key"))
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a trading assistant."},
        {"role": "user", "content": gpt_prompt}
    ],
    temperature=0.4
)

gpt_result = response.choices[0].message.content.strip()
if not gpt_result:
    print("❌ GPT response was empty.")
    sys.exit(1)

ranked_symbols = [s.strip().upper() for s in gpt_result.split(",") if s.strip()]

# ✅ PART 4: Final Output — Save Top 50 Picks
print("📦 Filtering GPT-ranked symbols from scored universe...")

final_selected = []
seen_symbols = set()

for symbol in ranked_symbols:
    match = next((s for s in final_stocks if s[0] == symbol), None)
    if match and symbol not in seen_symbols:
        final_selected.append(match)
        seen_symbols.add(symbol)

if len(final_selected) == 0:
    print("❌ No final stocks matched GPT output.")
    sys.exit(1)

# Save top 50 to dynamic_stock_list.csv
output_file = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"
with open(output_file, "w") as f:
    f.write("symbol,security_id\n")
    for s in final_selected[:FINAL_STOCK_LIMIT]:
        f.write(f"{s[0]},{s[1]}\n")

# === ✅ FINAL SUMMARY LOG UPDATE ===
try:
    summary_log_path = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"
    now = datetime.now().strftime("%m/%d/%Y %H:%M")  # e.g., 5/27/2025 14:45

    total_count = len(final_filtered_list)  # this is same as scanned, affordable, technical_passed, volume_passed

    summary_data = {
        "date": now,
        "total_scanned": total_count,
        "affordable": total_count,
        "technical_passed": total_count,
        "volume_passed": total_count,
        "sentiment_passed": sentiment_passed,
        "rsi_passed": rsi_passed,
        "final_selected": len(final_selected)
    }

    # Append or update summary row
    if os.path.exists(summary_log_path):
        existing_df = pd.read_csv(summary_log_path)
        mask = (existing_df["date"] == now) & (existing_df["total_scanned"] == total_count)
        if mask.any():
            existing_df.loc[mask, summary_data.keys()] = summary_data.values()
        else:
            existing_df = pd.concat([existing_df, pd.DataFrame([summary_data])], ignore_index=True)
        existing_df.to_csv(summary_log_path, index=False)
    else:
        pd.DataFrame([summary_data]).to_csv(summary_log_path, index=False)

    print("📝 Logged summary to filter_summary_log.csv")

except Exception as e:
    print(f"⚠️ Could not log summary: {e}")

end_time = datetime.now()
elapsed = end_time - start_time
print(f"• Total Time: {str(elapsed).split('.')[0]}")