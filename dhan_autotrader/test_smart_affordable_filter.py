import csv
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

# === Dhan Auth ===
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ4MDcyMDEzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNjg1NzM1OSJ9.ISl7D5ixliWbjnpWQwSXOXJToLpJ8FEGCIIwZTCKPCk6pOGnrO74jQa1SvZpsHhAm7tC1vjwnK1tH8vXaqoQaQ"
CLIENT_ID = "1106857359"
HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# === File Paths ===
MASTER_CSV = Path("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv")
CAPITAL_CSV = Path("D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv")
OUTPUT_TXT = Path("D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt")
UNSUPPORTED_TXT = Path("D:/Downloads/Dhanbot/dhan_autotrader/unavailable_symbols.txt")

# === Get Current Capital ===
def get_current_capital():
    try:
        with open(CAPITAL_CSV) as f:
            return float(f.read().strip())
    except Exception as e:
        print(f"⚠️ Failed to read capital, defaulting to ₹1000. Error: {e}")
        return 1000

# === Load dhan_master.csv ===
def load_dhan_symbols():
    symbols = []
    try:
        with open(MASTER_CSV, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row['SEM_TRADING_SYMBOL'].strip().upper()
                security_id = row['SEM_SMST_SECURITY_ID'].strip()
                if symbol.isalpha() and security_id.isdigit():
                    symbols.append((symbol, security_id))
    except Exception as e:
        print(f"❌ Error loading dhan_master.csv: {e}")
    return symbols

# === Reliable LTP fetch using Candle API ===
def fetch_latest_price(symbol, security_id):
    now = datetime.now()
    start_time = (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    end_time = now.strftime("%Y-%m-%d %H:%M:%S")

    url = "https://api.dhan.co/v2/charts/intraday"
    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "1",
        "oi": "false",
        "fromDate": start_time,
        "toDate": end_time
    }

    try:
        resp = requests.post(url, headers=HEADERS, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            closes = data.get("close", [])
            if closes:
                return float(closes[-1])
        else:
            print(f"❌ {symbol} response {resp.status_code}")
    except Exception as e:
        print(f"⚠️ {symbol} exception: {e}")
    return None

# === Main filtering logic ===
def fetch_affordable_stocks(symbols):
    capital = get_current_capital()
    affordable = []
    unsupported = []

    for symbol, secid in symbols:
        price = fetch_latest_price(symbol, secid)
        if price is None:
            unsupported.append(symbol)
        elif price <= capital:
            affordable.append((symbol, price, secid))
            print(f"✅ {symbol}: ₹{price}")
        time.sleep(0.6)

    # Save unsupported
    with open(UNSUPPORTED_TXT, "w") as f:
        for s in unsupported:
            f.write(f"{s}\n")
    print(f"🧾 {len(unsupported)} unsupported symbols logged.")
    return affordable

# === Save output ===
def save_affordable_to_file(stocks):
    with open(OUTPUT_TXT, "w") as f:
        for symbol, price, secid in stocks:
            f.write(f"{symbol},{price},{secid}\n")
    print(f"✅ Saved {len(stocks)} affordable stocks to {OUTPUT_TXT}")

# === Entry Point ===
if __name__ == "__main__":
    all_symbols = load_dhan_symbols()
    print(f"📦 Loaded {len(all_symbols)} total symbols")
    affordable = fetch_affordable_stocks(all_symbols)
    save_affordable_to_file(affordable)
    if not affordable:
        print("🚨 No affordable stocks found.")
