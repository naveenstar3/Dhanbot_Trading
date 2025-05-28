import pandas as pd
import requests
import json
import time as systime
import sys
from datetime import datetime, timedelta
from dhan_api import get_live_price
import datetime as dt
from utils_logger import log_bot_action 

PREMARKET_MODE = True 
start_time = datetime.now()

# === Load config for credentials ===
with open("D:/Downloads/Dhanbot/dhan_autotrader/config.json", "r") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# === Load Capital ===
try:
    capital_df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv")
    if capital_df.columns[0] in ["capital", "current_capital"]:
        CAPITAL = float(capital_df.iloc[-1, 0])
    else:
        # No header present — use raw value
        capital_df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv", header=None)
        CAPITAL = float(capital_df.iloc[-1, 0])
    print(f"💰 Capital Loaded: ₹{CAPITAL}")
except Exception as e:
    raise Exception(f"❌ Failed to load capital: {e}")

MASTER_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
OUTPUT_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/weekly_affordable_volume_filtered.csv"
SUMMARY_LOG = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"

# === Safer 10-day window to guarantee 5 trading days ===
to_time = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
from_time = to_time - timedelta(days=10)
from_date = from_time.strftime('%Y-%m-%d %H:%M:%S')
to_date = to_time.strftime('%Y-%m-%d %H:%M:%S')

print("🚀 Building weekly affordable + volume-based universe...")

master_df = pd.read_csv(MASTER_CSV)
# === Optional Test Mode: Single Security ID ===
test_security_id = None
if len(sys.argv) > 1:
    test_security_id = str(sys.argv[1]).strip()
    print(f"🔬 Test Mode: Single Security ID = {test_security_id}")
    master_df = master_df[master_df["SEM_SMST_SECURITY_ID"].astype(str) == test_security_id]
    if master_df.empty:
        print(f"❌ Security ID {test_security_id} not found in dhan_master.csv")
        sys.exit(1)

required_cols = ["SM_SYMBOL_NAME", "SEM_SMST_SECURITY_ID", "SEM_SEGMENT", "SEM_SERIES"]
for col in required_cols:
    if col not in master_df.columns:
        raise Exception(f"❌ Missing column: {col}")

# === Initial Filtering ===
filtered = []
for idx, row in master_df.iterrows():
    try:
        symbol = str(row["SEM_TRADING_SYMBOL"]).strip().upper()
        secid = str(row["SEM_SMST_SECURITY_ID"]).strip()
        exch_type = str(row.get("SEM_EXCH_INSTRUMENT_TYPE", "")).strip().upper()
        series = str(row.get("SEM_SERIES", "")).strip().upper()

        skip_keywords = ['-RE', '-PP', 'SGB', 'TS', 'RJ', 'WB', 'AP', 'PN', 'HP',
                         'SFMP', 'M6DD', 'EMKAYTOOLS', 'ICICM', 'TRUST', 'REIT', 'INVIT', 'ETF', 'FUND']
        skip_exch_types = ['DBT', 'DEB', 'MF', 'GS', 'TB']
        skip_series = ['SG', 'GS', 'YL', 'MF', 'NC', 'TB']

        if (
            not secid.isdigit() or len(secid) < 4 or len(symbol) < 3 or
            symbol.startswith(tuple("0123456789")) or
            any(kw in symbol for kw in skip_keywords) or
            exch_type in skip_exch_types or
            series in skip_series
        ):
            continue

        filtered.append((symbol, secid))

    except Exception as e:
        print(f"⚠️ Row skipped due to error: {e}")
        continue

# ✅ Apply de-duplication once after the loop
seen = set()
unique_filtered = []
for row in filtered:
    key = (row[1], row[0])  # (secid, symbol)
    if key not in seen:
        seen.add(key)
        unique_filtered.append(row)
filtered = unique_filtered

results = []
total_checked = len(filtered)
volume_passed_count = 0
affordable_count = 0

for idx, (symbol, secid) in enumerate(filtered, 1):
    print(f"\n🔄 [{idx}/{total_checked}] {symbol}")

    try:
        # Step 1: Check Volume first (avoid LTP if volume fails)
        payload = {
            "securityId": secid,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": "1",
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date
        }

        resp = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)
        if resp.status_code != 200:
            print(f"❌ Volume fetch error {resp.status_code}")
            continue

        data = resp.json()
        if not all(k in data for k in ["volume", "timestamp"]):
            print("❌ Missing volume/timestamp keys.")
            continue

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
            "volume": data["volume"]
        })
        df["date"] = df["timestamp"].dt.date
        volume_by_day = df.groupby("date")["volume"].sum()
        avg_volume = int(volume_by_day.tail(5).mean())

        if avg_volume < 200000:
            print(f"⛔ Low Volume: {avg_volume}")
            continue

        volume_passed_count += 1
        print(f"✅ Passed Volume: {avg_volume}")

        # Step 2: Only check affordability if volume passed
        ltp = get_live_price(symbol, secid, premarket=True)
        if not ltp or ltp > CAPITAL:
            print(f"⛔ Not Affordable: ₹{ltp} > ₹{CAPITAL}")
            continue

        affordable_count += 1
        print(f"✅ Affordable — ₹{ltp}")

        # Step 3: Calculate 5-day ATR
        ohlc_payload = {
            "securityId": secid,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": "1DAY",
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date
        }
    
        atr = 0.0
        try:
            ohlc_resp = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=ohlc_payload)
            if ohlc_resp.status_code == 200:
                ohlc_data = ohlc_resp.json()
                if all(k in ohlc_data for k in ["high", "low", "close"]):
                    highs = pd.Series(ohlc_data["high"])
                    lows = pd.Series(ohlc_data["low"])
                    closes = pd.Series(ohlc_data["close"])
                    tr = pd.concat([
                        highs - lows,
                        (highs - closes.shift(1)).abs(),
                        (lows - closes.shift(1)).abs()
                    ], axis=1).max(axis=1)
                    atr = round(tr.tail(5).mean(), 2)
        except Exception as e:
            print(f"⚠️ Failed ATR for {symbol}: {e}")
    
        # All filters passed
        results.append({
            "symbol": symbol,
            "security_id": secid,
            "ltp": ltp,
            "avg_volume": avg_volume,
            "atr": atr
        })
    
        print(f"🟢 Final Selected: {symbol} (LTP: ₹{ltp}, Volume: {avg_volume})")

        systime.sleep(0.5)

    except Exception as e:
        print(f"⚠️ Error for {symbol}: {e}")
        continue

# === Save to CSV ===
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved {len(results)} stocks to {OUTPUT_CSV}")

# === Log filter summary ===
summary_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_checked": total_checked,
    "volume_passed": volume_passed_count,
    "affordable_passed": affordable_count,
    "capital_used": CAPITAL
}

try:
    if not pd.io.common.file_exists(SUMMARY_LOG):
        pd.DataFrame([summary_row]).to_csv(SUMMARY_LOG, index=False)
    else:
        pd.DataFrame([summary_row]).to_csv(SUMMARY_LOG, mode='a', header=False, index=False)
    print(f"📝 Logged summary to filter_summary_log.csv")
except Exception as e:
    print(f"⚠️ Failed to write filter summary: {e}")

# === Final Summary ===
print(f"\n📊 Final Summary:")
end_time = datetime.now()
elapsed = end_time - start_time
print(f"• Total Checked: {total_checked}")
print(f"• Passed Volume: {volume_passed_count}")
print(f"• Passed Affordability: {affordable_count}")
print(f"• Final Saved: {len(results)}")
print(f"• Total Time: {str(elapsed).split('.')[0]}")

