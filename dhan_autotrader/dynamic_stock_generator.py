# ✅ PART 1: Imports and Configuration
import pandas as pd
import pytz
import os
import requests
import json
import time as systime
from dhan_api import get_security_id, get_current_capital
from utils_logger import log_bot_action
from datetime import datetime, timedelta

# === Credentials and Headers ===
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

FINAL_STOCK_LIMIT = 50
PREMARKET_MODE = True
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# ✅ Helper: Check if market is closed

def is_market_closed():
    if PREMARKET_MODE:
        return False
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    weekday = now.weekday()
    hour = now.hour + now.minute / 60.0
    return weekday >= 5 or hour < 9.25 or hour > 15.5

# ✅ Fetch live LTP using DHAN Candle API
def fetch_latest_price(symbol, security_id):
    now = datetime.now()

    if PREMARKET_MODE or now.hour < 9 or now.hour >= 16:
        from_time = (now - timedelta(days=1)).replace(hour=15, minute=20, second=0, microsecond=0)
        to_time = from_time + timedelta(minutes=5)
    else:
        from_time = now - timedelta(minutes=5)
        to_time = now

    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "1",
        "oi": "false",
        "fromDate": from_time.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": to_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        resp = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)
        if resp.status_code == 429:
            log_bot_action("dynamic_stock_generator.py", "PriceFetch", "❌ 429 Rate Limit", f"{symbol} - Rate limit hit")
            return 429
        elif resp.status_code == 400:
            print(f"⚠️ {symbol} rejected (400): Likely unsupported for candles or invalid securityId.")
            log_bot_action("dynamic_stock_generator.py", "PriceFetch", "❌ 400 Rejected", f"{symbol} - Invalid Security")
            return None
        elif resp.status_code == 200:
            data = resp.json()
            closes = data.get("close", [])
            if closes:
                return float(closes[-1])
            else:
                log_bot_action("dynamic_stock_generator.py", "PriceFetch", "⚠️ Empty OHLC", f"{symbol} - No close price found")
        else:
            print(f"❌ {symbol} response {resp.status_code}")
            log_bot_action("dynamic_stock_generator.py", "PriceFetch", f"❌ HTTP {resp.status_code}", f"{symbol} - Unexpected response")
    except Exception as e:
        print(f"⚠️ {symbol} LTP fetch error: {e}")
        log_bot_action("dynamic_stock_generator.py", "PriceFetch", "⚠️ Exception", f"{symbol} - {str(e)}")
    return None

# ✅ PART 2: Load and Filter Affordable Stocks from dhan_master.csv
def load_dhan_master(path):
    master_list = []
    try:
        reader = pd.read_csv(path)
        for _, row in reader.iterrows():
            symbol = str(row['SEM_TRADING_SYMBOL']).strip().upper()
            secid = str(row['SEM_SMST_SECURITY_ID']).strip()
            exch_type = str(row.get("SEM_EXCH_INSTRUMENT_TYPE", "")).strip().upper()
            series = str(row.get("SEM_SERIES", "")).strip().upper()

            # ✅ Skip invalid and unsupported symbols
            skip_keywords = [
                '-RE', '-PP', 'SGB', 'TS', 'RJ', 'WB', 'AP', 'PN', 'HP',
                'SFMP', 'M6DD', 'EMKAYTOOLS', 'ICICM', 'TRUST', 'REIT', 'INVIT', 'ETF', 'FUND'
            ]
            skip_exch_types = ['DBT', 'DEB', 'MF', 'GS', 'TB']
            skip_series = ['SG', 'GS', 'YL', 'MF', 'NC', 'TB']

            if (
                not secid.isdigit() or
                len(secid) < 4 or
                len(symbol) < 3 or
                symbol.startswith(tuple('0123456789')) or
                any(kw in symbol for kw in skip_keywords) or
                exch_type in skip_exch_types or
                series in skip_series
            ):
                continue

            master_list.append((symbol, secid))
    except Exception as e:
        print(f"❌ Error loading dhan_master.csv: {e}")
    return master_list

# ✅ Build affordable stock list
def get_affordable_symbols(master_list):
    capital = get_current_capital()
    affordable = []
    unavailable = []

    now = datetime.now()
    for idx, (symbol, secid) in enumerate(master_list, start=1):
        if not secid.isdigit() or len(secid) < 3:
            continue

        # ✅ Step 1: Fetch current/latest price
        price = fetch_latest_price(symbol, secid)
        if price is None:
            unavailable.append(symbol)
            log_bot_action("dynamic_stock_generator.py", "PriceFetch", "⚠️ NULL LTP", f"{symbol} - Price=None or fetch failed")
            continue
        elif price == 429:
            unavailable.append(symbol)
            log_bot_action("dynamic_stock_generator.py", "PriceFetch", "❌ 429 Rate Limit", f"{symbol} - Rate limited by Dhan")
            continue
        elif price > capital:
            print(f"⛔ Skipped {symbol} ({idx}/{len(master_list)}) — ₹{price} > ₹{capital}")
            continue

        # ✅ Step 2: Fetch 15:10–15:25 closing candles
        try:
            check_day = now - timedelta(days=1)
            from_time = check_day.replace(hour=15, minute=10, second=0, microsecond=0)
            to_time = from_time + timedelta(minutes=15)
            payload = {
                "securityId": secid,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "interval": "1",
                "oi": "false",
                "fromDate": from_time.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": to_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            resp = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)
            prev_closes = resp.json().get("close", [])
            if prev_closes:
                avg_close = round(sum(prev_closes) / len(prev_closes), 2)
                return_change = round(((price - avg_close) / avg_close) * 100, 2)
                print(f"🔍 {symbol} → Avg Close: ₹{avg_close} | LTP: ₹{price}")
            else:
                return_change = 0.0
        except Exception as e:
            return_change = 0.0
            log_bot_action("dynamic_stock_generator.py", "PrevClose", "⚠️ ERROR", f"{symbol} - Candle fetch failed: {str(e)}")

        # ✅ Step 3: Append with ranking value
        if return_change > 0.0:
            affordable.append((symbol, secid, return_change))
            print(f"✅ Affordable {idx}/{len(master_list)}: {symbol} → Momentum: {return_change}%")
        else:
            print(f"⚠️ Skipped {symbol} ({idx}/{len(master_list)}) → Momentum: {return_change}% (Too Low)")

        systime.sleep(0.5)

    skipped_count = idx - len(affordable) - len(unavailable)
    print(f"✅ Total affordable: {len(affordable)} | Skipped (Too Low/High): {skipped_count} | Unavailable: {len(unavailable)}")
    return affordable
    
# ✅ Finalize and save stock list
def save_final_stock_list(stocks, filepath):
    df = pd.DataFrame(stocks, columns=["symbol", "security_id", "momentum"])
    df.to_csv(filepath, index=False)
    log_bot_action("dynamic_stock_generator.py", "Stock List Updated", "✅ COMPLETE", f"{len(stocks)} stocks saved")
    print(f"✅ Final {len(stocks)} stocks saved to {filepath}")

# ✅ Save filter summary
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

# ✅ PART 3: Run dynamic stock selection
def run_dynamic_stock_selection():
    print("🚀 Starting dynamic stock selection..." + (" (PRE-MARKET MODE)" if PREMARKET_MODE else ""))

    if is_market_closed():
        print("⏸️ Market is closed. Exiting.")
        return

    master_path = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
    output_file = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt"

    master_list = load_dhan_master(master_path)
    affordable_ids = get_affordable_symbols(master_list)

    if not affordable_ids:
        print("🚨 No affordable stocks found. Exiting.")
        return

    affordable_ids = sorted(affordable_ids, key=lambda x: x[2], reverse=True)
    final_stocks = [symbol for symbol, _, _ in affordable_ids][:FINAL_STOCK_LIMIT]

    filter_stats = {
        "total_scanned": len(master_list),
        "affordable": len(affordable_ids),
        "technical_passed": "SKIPPED",
        "volume_passed": "SKIPPED",
        "sentiment_passed": "SKIPPED",
        "rsi_passed": "SKIPPED",
        "dynamic_list_selected": len(open(output_file).read().strip().splitlines())
    }
    save_filter_summary(filter_stats)
    log_bot_action("dynamic_stock_generator.py", "run_dynamic_stock_selection", "✅ FINISHED", f"Affordable={len(affordable_ids)}, Final={len(final_stocks)}")


# ✅ Main Entry
if __name__ == "__main__":
    run_dynamic_stock_selection()
