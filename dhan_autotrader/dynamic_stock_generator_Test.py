# ✅ PART 1: Imports and Configuration
import pandas as pd
import pytz
import os
import requests
import json
import time as systime
from dhan_api import get_security_id, get_current_capital, get_historical_price
from utils_logger import log_bot_action
from datetime import datetime, timedelta
import datetime as dt
import sys
import openai

start_time = systime.time()
# === Credentials and Headers ===
with open("D:/Downloads/Dhanbot/dhan_autotrader/config.json", "r") as f:
    config = json.load(f)

FINAL_STOCK_LIMIT = 50
PREMARKET_MODE = True
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
openai.api_key = config.get("openai_api_key")

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

session = requests.Session()
session.headers.update(HEADERS)

def is_market_closed():
    if PREMARKET_MODE:
        return False
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    weekday = now.weekday()
    hour = now.hour + now.minute / 60.0
    return weekday >= 5 or hour < 9.25 or hour > 15.5
    
# === Profitability-Based Filters ===

def is_strong_close(eod_candles):
    try:
        prev_close = eod_candles[-1]["close"]
        prev_high = eod_candles[-1]["high"]
        return prev_close / prev_high >= 0.98
    except:
        return False

def is_uptrend_last_3_closes(eod_candles):
    try:
        closes = [c["close"] for c in eod_candles[-3:]]
        return closes[0] < closes[1] < closes[2]
    except:
        return False

def atr_above_threshold(eod_candles, threshold=5):
    try:
        ranges = [c["high"] - c["low"] for c in eod_candles[-5:]]
        atr = sum(ranges) / len(ranges)
        return atr >= threshold
    except:
        return False

def fetch_latest_price(symbol, security_id):
    now = datetime.now()
    if PREMARKET_MODE or now.hour < 9 or now.hour >= 16:
        india = pytz.timezone("Asia/Kolkata")
        prev_day = dt.datetime.now(india) - dt.timedelta(days=1)
        while prev_day.weekday() >= 5:  # Skip Sat/Sun
            prev_day -= dt.timedelta(days=1)
    
        from_time = prev_day.replace(hour=15, minute=20, second=0, microsecond=0)
        to_time = from_time + dt.timedelta(minutes=5)    
    else:
        from_time = now - timedelta(minutes=5)
        to_time = now

    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "1",
        "oi": False,
        "fromDate": from_time.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": to_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    print(f"📌 DEBUG: In fetch_latest_price() for {symbol}")
    try:
        resp = session.post("https://api.dhan.co/v2/charts/intraday", json=payload)
        if resp.status_code == 429:
            print(f"⏳ Rate limit hit for {symbol}")
            log_bot_action("dynamic_stock_generator.py", "PriceFetch", "❌ 429 Rate Limit", f"{symbol} - Rate limit hit")
            return 429
        elif resp.status_code != 200:
            print(f"❌ API failed for {symbol}: {resp.status_code} | {resp.text}")
            return None
        elif resp.status_code == 200:
            data = resp.json()
            closes = data.get("close", [])
            return float(closes[-1]) if closes else None
    except Exception as e:
        print(f"⚠️ {symbol} LTP fetch error: {e}")
    return None

def load_dhan_master(path):
    master_list = []
    try:
        reader = pd.read_csv(path)
        for _, row in reader.iterrows():
            symbol = str(row['SEM_TRADING_SYMBOL']).strip().upper()
            secid = str(row['SEM_SMST_SECURITY_ID']).strip()
            exch_type = str(row.get("SEM_EXCH_INSTRUMENT_TYPE", "")).strip().upper()
            series = str(row.get("SEM_SERIES", "")).strip().upper()

            skip_keywords = ['-RE', '-PP', 'SGB', 'TS', 'RJ', 'WB', 'AP', 'PN', 'HP',
                            'SFMP', 'M6DD', 'EMKAYTOOLS', 'ICICM', 'TRUST', 'REIT', 'INVIT', 'ETF', 'FUND']
            skip_exch_types = ['DBT', 'DEB', 'MF', 'GS', 'TB']
            skip_series = ['SG', 'GS', 'YL', 'MF', 'NC', 'TB']

            if (not secid.isdigit() or len(secid) < 4 or len(symbol) < 3 or 
                symbol.startswith(tuple('0123456789')) or
                any(kw in symbol for kw in skip_keywords) or
                exch_type in skip_exch_types or series in skip_series):
                continue

            master_list.append((symbol, secid))
    except Exception as e:
        print(f"❌ Error loading dhan_master.csv: {e}")
    return master_list

def get_affordable_symbols(master_list):
    capital = get_current_capital()
    affordable = []
    unavailable = []

    for idx, (symbol, secid) in enumerate(master_list, start=1):
        if not secid.isdigit() or len(secid) < 3:
            continue
    
        price = fetch_latest_price(symbol, secid)
        systime.sleep(0.5)  # ✅ Add delay after each fetch
    
        if price is None or price == 429:
            unavailable.append(symbol)
            continue
        elif price > capital:
            print(f"⛔ Skipped {symbol} ({idx}/{len(master_list)}) — ₹{price} > ₹{capital}")
            continue
    
        affordable.append((symbol, secid, 0.0))
        print(f"✅ Added {symbol} ({idx}/{len(master_list)})")
    
        systime.sleep(0.5)

    print(f"📊 Final affordable: {len(affordable)} | Unavailable: {len(unavailable)}")
    return affordable
    
def save_final_stock_list(stocks, filepath):
    df = pd.DataFrame(stocks, columns=["symbol", "security_id", "momentum"])
    df.to_csv(filepath, index=False)
    print(f"✅ Saved {len(stocks)} stocks to {filepath}")

def save_filter_summary(stats):
    file = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"
    header = ",".join(stats.keys())
    row = ",".join(str(x) for x in stats.values())
    date = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("date," + header + "\n")
    with open(file, "a") as f:
        f.write(f"{date},{row}\n")

def run_dynamic_stock_selection():
    test_security_id = None
    if len(sys.argv) > 1:
        test_security_id = str(sys.argv[1]).strip()
    print("🚀 Starting dynamic stock selection..." + (" (PRE-MARKET MODE)" if PREMARKET_MODE else ""))
    
    if is_market_closed():
        print("⏸️ Market is closed. Exiting.")
        return

    weekly_path = "D:/Downloads/Dhanbot/dhan_autotrader/weekly_affordable_volume_filtered.csv"
    output_file = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"
    
    try:
        df = pd.read_csv(weekly_path)
        master_list = list(zip(df["symbol"].astype(str).str.upper(), df["security_id"].astype(str)))
    except Exception as e:
        print(f"❌ Failed to load weekly core universe: {e}")
        return
    
    if test_security_id:
        master_list = [entry for entry in master_list if entry[1] == test_security_id]
        if not master_list:
            print(f"❌ Security ID {test_security_id} not found in weekly core universe")
            return
    
    print("📊 Checking affordability and volume together...")
    capital = get_current_capital()
    final_stocks = []
    affordable_count = 0
    volume_passed_count = 0
    unavailable = []
    
    now = datetime.now()
    from_time = (now - timedelta(days=8)).replace(hour=9, minute=15, second=0, microsecond=0)
    to_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    from_date = from_time.strftime('%Y-%m-%d %H:%M:%S')
    to_date = to_time.strftime('%Y-%m-%d %H:%M:%S')

    for idx, (symbol, secid) in enumerate(master_list, start=1):
        payload = {
            "securityId": str(secid),
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": "1",
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date
        }

        try:
            resp = session.post("https://api.dhan.co/v2/charts/intraday", json=payload)
        
            # Only print the error if it will actually block the execution
            if resp.status_code == 401:
                print(f"[{symbol}] ❌ Skipped — Access token is expired or invalid (401).")
                continue
            
            elif resp.status_code != 200:
                print(f"[{symbol}] ❌ Skipped — Volume fetch failed: {resp.status_code} | {resp.text}")
                continue
            
            try:
                data = resp.json()
            except Exception as e:
                print(f"[{symbol}] ❌ Failed to parse response JSON: {e}")
                continue
            
            # ✅ Check first if it’s a real error response
            if not isinstance(data, dict):
                print(f"[{symbol}] ❌ Invalid response structure (not a dict)")
                continue
            
            if not all(k in data for k in ["volume", "timestamp", "close"]):
                print(f"[{symbol}] ❌ Skipped — Incomplete data structure received.")
                continue
                       
            # ✅ Then proceed to validate structure
            if not all(k in data for k in ["volume", "timestamp", "close"]):
                print(f"[{symbol}] ❌ Skipped — Incomplete data structure received.")
                continue
                
        
            # ✅ Proceed with valid data
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
                "volume": data["volume"],
                "close": [c / 100 for c in data["close"]]
            })
            df["date"] = df["timestamp"].dt.date
            volume_by_day = df.groupby("date")["volume"].sum()
            avg_volume = int(volume_by_day.tail(5).mean())
            ltp = df["close"].iloc[-1]
        
            if not ltp or ltp > capital:
                print(f"[{symbol}] ⛔ Not affordable: ₹{ltp} > ₹{capital}")
                unavailable.append(symbol)
                continue
        
            affordable_count += 1
        

            if avg_volume >= 200000:
                volume_passed_count += 1

                # EOD scoring logic continues unchanged...
                atr = 0
                uptrend = False
                strong_close = False
                try:
                    eod_data = get_historical_price(secid, interval="EOD")
                    if eod_data and "close" in eod_data:
                        candles = [
                            {"open": o, "high": h, "low": l, "close": c}
                            for o, h, l, c in zip(eod_data["open"], eod_data["high"], eod_data["low"], eod_data["close"])
                        ]
                        if len(candles) >= 5:
                            atr = sum([c["high"] - c["low"] for c in candles[-5:]]) / 5
                            strong_close = is_strong_close(candles)
                            uptrend = is_uptrend_last_3_closes(candles)
                except Exception as e:
                    print(f"[{symbol}] ❌ EOD fetch error: {e}")

                score = (
                    (avg_volume / 1e6) * 0.4 +
                    (ltp / 1000) * 0.2 +
                    (atr / 5) * 0.2 +
                    (1 if uptrend else 0) * 0.1 +
                    (1 if strong_close else 0) * 0.1
                )

                final_stocks.append((symbol, secid, round(score, 3)))
                print(f"[{symbol}] ✅ Affordable and volume passed ({affordable_count}/{len(master_list)}) — Avg Vol: {avg_volume:,} | Score={round(score,2)}")

            systime.sleep(0.5)

        except Exception as e:
            print(f"[{symbol}] ❌ Volume fetch exception: {e}")
            log_bot_action("dynamic_stock_generator.py", "VolumeFetch", "❌ Exception", f"{symbol} - {e}")
            continue  
    
    final_stocks.sort(key=lambda x: x[2], reverse=True)
    
    # === GPT FILTER (skip if only 1 stock)
    if len(final_stocks) <= 1:
        print("🧠 Skipping GPT — not enough candidates for ranking.")
        final_stocks = final_stocks[:FINAL_STOCK_LIMIT]
    else:
        top_for_gpt = final_stocks[:100]  # GPT will pick from top 100 based on score
    
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
        gpt_prompt = f"""
        🧠 You are an expert intraday stock selector and market analyst. Today is {now} IST.
    
        You have been given a list of pre-screened stocks, each with a momentum score based on:
        - 5-day average volume
        - ATR (intraday movement potential)
        - Stock price
        - Recent EOD uptrend pattern
        - Strong close signals
    
        Your goal is to select **exactly 50 stocks** that are the **safest and most profitable intraday picks for today**.
    
        🔍 In addition to the scores, also consider:
        - **Macro sentiment** (global cues, Nifty/BSE trends)
        - **Fundamental strength** (avoid junk/SME/PSU-only companies)
        - **Recent news sentiment** (avoid negative headlines or red-flag events)
        - **Technical breakout readiness**
        - Prefer stocks with:
        - High delivery %
        - Sector strength
        - Consistent earnings or leadership
    
        ⚠️ Avoid:
        - Large gap-up stocks unless momentum is strong
        - Stocks prone to manipulation or illiquidity
    
        🎯 Output format: Just the stock symbols, comma-separated, no explanation.
    
        Stock List:
        {"\n".join([f"{s[0]} | Score: {s[2]}" for s in top_for_gpt])}
        """
    
        try:
            client = openai.OpenAI(api_key=config["openai_api_key"])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": gpt_prompt}]
            )
            selected = [s.strip() for s in response.choices[0].message.content.upper().split(",") if s.strip()]
            print(f"🤖 GPT picked: {selected}")
            final_stocks = [x for x in final_stocks if x[0] in selected][:FINAL_STOCK_LIMIT]
        except Exception as e:
            print(f"⚠️ GPT fallback used due to error: {e}")
            final_stocks = final_stocks[:FINAL_STOCK_LIMIT]
    
    
    save_final_stock_list(final_stocks, output_file)

    filter_stats = {
        "total_scanned": len(master_list),
        "affordable": affordable_count,
        "technical_passed": affordable_count,
        "volume_passed": volume_passed_count,   
        "sentiment_passed": "SKIPPED",
        "rsi_passed": "SKIPPED",
        "dynamic_list_selected": len(final_stocks)
    }
   
    save_filter_summary(filter_stats)
    log_bot_action("dynamic_stock_generator.py", "run_dynamic_stock_selection", "✅ FINISHED", 
              f"Scanned={len(master_list)}, Final={len(final_stocks)}")
    print(f"🕒 Total Run Time: {round((systime.time() - start_time)/60, 2)} minutes")
    
if __name__ == "__main__":
    run_dynamic_stock_selection()