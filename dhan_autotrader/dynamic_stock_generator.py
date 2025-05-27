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
import datetime as dt
import sys


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

def is_market_closed():
    if PREMARKET_MODE:
        return False
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    weekday = now.weekday()
    hour = now.hour + now.minute / 60.0
    return weekday >= 5 or hour < 9.25 or hour > 15.5

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
        "oi": "false",
        "fromDate": from_time.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": to_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        resp = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)
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
        print(f"🔬 Test Mode: Single Security ID = {test_security_id}")

    print("🚀 Starting dynamic stock selection..." + (" (PRE-MARKET MODE)" if PREMARKET_MODE else ""))
    
    if is_market_closed():
        print("⏸️ Market is closed. Exiting.")
        return

    master_path = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
    output_file = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"

    master_list = load_dhan_master(master_path)

    if test_security_id:
        master_list = [entry for entry in master_list if entry[1] == test_security_id]
        if not master_list:
            print(f"❌ Security ID {test_security_id} not found in dhan_master.csv")
            return

    affordable_ids = get_affordable_symbols(master_list)

    if not affordable_ids:
        print("🚨 No affordable stocks found. Exiting.")
        return

    # ✅ Load volume filter data
    print("📊 Fetching 1-min intraday data (last 5d)...")

    filtered = []
    end = datetime.now()
    start = end - timedelta(days=5)
    from_date = start.strftime('%Y-%m-%d') + " 09:30:00"
    to_date = end.strftime('%Y-%m-%d') + " 15:30:00"
    
    for symbol, secid, _ in affordable_ids:
        payload = {
            "securityId": secid,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": "1",
            "oi": "false",
            "fromDate": from_date,
            "toDate": to_date
        }
    
        try:
            resp = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)
            data = resp.json()
            if not all(k in data for k in ["volume", "timestamp"]):
                print(f"[{symbol}] ❌ No EOD data.")
                continue
    
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
                "volume": data["volume"]
            })
            df["date"] = df["timestamp"].dt.date
            volume_by_day = df.groupby("date")["volume"].sum()
            avg_volume = int(volume_by_day.tail(5).mean())
    
            if avg_volume >= 200000:
                filtered.append((symbol, secid, 0.0))
                print(f"[{symbol}] ✅ Avg Vol = {avg_volume}")
            else:
                print(f"[{symbol}] ⛔ Low Vol = {avg_volume}")
    
            systime.sleep(0.5)
    
        except Exception as e:
            print(f"[{symbol}] ❌ Error: {e}")
            continue
    
    
    final_stocks = filtered[:FINAL_STOCK_LIMIT]
    
    save_final_stock_list(final_stocks, output_file)

    filter_stats = {
        "total_scanned": len(master_list),
        "affordable": len(affordable_ids),
        "technical_passed": len(affordable_ids),  # Now same as affordable count
        "volume_passed": "SKIPPED",
        "sentiment_passed": "SKIPPED",
        "rsi_passed": "SKIPPED",
        "dynamic_list_selected": len(final_stocks)
    }
    save_filter_summary(filter_stats)
    log_bot_action("dynamic_stock_generator.py", "run_dynamic_stock_selection", "✅ FINISHED", 
                  f"Affordable={len(affordable_ids)}, Final={len(final_stocks)}")

if __name__ == "__main__":
    run_dynamic_stock_selection()