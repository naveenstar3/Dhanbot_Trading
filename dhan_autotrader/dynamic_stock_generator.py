# premarket_nifty100_scanner.py
import os
import sys
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from dhan_api import get_live_price
import pytz
import time
from nsepython import nsefetch  # Using your working library
from db_logger import log_dynamic_stock_list, log_to_postgres
import io
import sys

log_buffer = io.StringIO()
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

log_buffer = io.StringIO()
sys.stdout = TeeLogger(sys.__stdout__, log_buffer)


# ======== CONFIGURATION ========
start_time = datetime.now()
CAPITAL_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv"
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
OUTPUT_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"
NIFTY100_CACHE = "D:/Downloads/Dhanbot/dhan_autotrader/nifty100_constituents.csv"
MASTER_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"

# ======== TRADING DAY CHECK ========
def is_trading_day():
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    if today.weekday() >= 5:  # Weekend check
        return False
    return True

# ======== TEST MODE CONTROL =========
FORCE_TEST = len(sys.argv) > 1 and sys.argv[1].strip().upper() == "NO"

if not is_trading_day() and not FORCE_TEST:
    print("⛔ Non-trading day. Exiting.")
    sys.exit(0)
elif not is_trading_day() and FORCE_TEST:
    print("🧪 Force test mode enabled via argument: market is OFF but proceeding.")

# ======== LOAD CREDENTIALS & CAPITAL ========
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

try:
    CAPITAL = float(pd.read_csv(CAPITAL_PATH, header=None).iloc[0, 0])
    print(f"💰 Capital Loaded: ₹{CAPITAL:,.2f}")
    log_to_postgres(datetime.now(), "Test_dynamic_stock_generator.py", "INFO", f"Capital loaded: ₹{CAPITAL:,.2f}")    
except Exception as e:
    print(f"❌ Capital loading failed: {e}")
    log_to_postgres(datetime.now(), "Test_dynamic_stock_generator.py", "ERROR", f"Capital loading failed: {e}")   
    sys.exit(1)

# ======== RELIABLE NIFTY 100 FETCH ========
def get_nifty100_constituents():
    """Fetch Nifty 100 using nsepython with robust error handling"""
    try:
        # Using the same method as your Top_Losers_Strategy
        url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100'
        data = nsefetch(url)
        symbols = [item["symbol"].strip().upper() for item in data["data"]]
        print(f"✅ Fetched {len(symbols)} current Nifty 100 constituents")
        return symbols
    except Exception as e:
        print(f"⚠️ NSE fetch failed: {e}. Using cached list")
        if os.path.exists(NIFTY100_CACHE):
            return pd.read_csv(NIFTY100_CACHE)["symbol"].tolist()
        return []  # Fallback

nifty100_symbols = get_nifty100_constituents()
if not nifty100_symbols:
    print("❌ No Nifty 100 symbols available")
    sys.exit(1)

# Save/update cache
pd.DataFrame(nifty100_symbols, columns=["symbol"]).to_csv(NIFTY100_CACHE, index=False)

def build_sector_map(nifty100_symbols):
    """Fetch sector-wise members from NSE and map symbols to sectors (dynamic)"""
    sector_index_map = {
        "NIFTY BANK": "NIFTY%20BANK",
        "NIFTY IT": "NIFTY%20IT",
        "NIFTY FMCG": "NIFTY%20FMCG",
        "NIFTY FIN SERVICE": "NIFTY%20FIN%20SERVICE",
        "NIFTY AUTO": "NIFTY%20AUTO",
        "NIFTY PHARMA": "NIFTY%20PHARMA",
        "NIFTY REALTY": "NIFTY%20REALTY",
        "NIFTY METAL": "NIFTY%20METAL",
        "NIFTY ENERGY": "NIFTY%20ENERGY"
    }

    symbol_sector_map = {}
    for sector_name, index_code in sector_index_map.items():
        try:
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_code}"
            data = nsefetch(url)
            for item in data["data"]:
                symbol = item["symbol"].strip().upper()
                if symbol in nifty100_symbols:
                    symbol_sector_map[symbol] = sector_name
        except Exception as e:
            print(f"⚠️ Failed sector map: {sector_name} – {str(e)[:60]}")
            continue
        time.sleep(0.8)
    
    return symbol_sector_map

def get_sector_strength():
    sector_indices = [
        "NIFTY BANK", "NIFTY IT", "NIFTY FMCG", "NIFTY FIN SERVICE", "NIFTY AUTO",
        "NIFTY PHARMA", "NIFTY REALTY", "NIFTY METAL", "NIFTY ENERGY"
    ]
    sector_gains = {}
    for sector in sector_indices:
        try:
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={sector.replace(' ', '%20')}"
            data = nsefetch(url)
            change = float(data["data"][0]["change"])
            sector_gains[sector] = change
        except Exception as e:
            print(f"⚠️ Sector fetch failed: {sector} → {str(e)[:50]}")
    # Return top 3 sectors by % gain
    top_sectors = sorted(sector_gains.items(), key=lambda x: x[1], reverse=True)[:7]
    print("📈 Top sectors today:", top_sectors)
    return [s[0] for s in top_sectors]

# ======== LOAD MASTER SECURITY LIST ========
try:
    master_df = pd.read_csv(MASTER_CSV)
    symbol_sector_map = build_sector_map(nifty100_symbols)
    master_df["base_symbol"] = master_df["SEM_TRADING_SYMBOL"].str.replace("-EQ", "").str.strip().str.upper()
    master_df["sector"] = master_df["base_symbol"].map(symbol_sector_map)
    top_sectors = get_sector_strength()
    nifty100_df = master_df[(master_df["base_symbol"].isin(nifty100_symbols))]
    print(f"📊 Master list filtered to {len(nifty100_df)} Nifty 100 stocks")
    
    if len(nifty100_df) == 0:
        print("❌ No matching securities found in master list")
        sys.exit(1)
except Exception as e:
    print(f"❌ Master CSV load failed: {e}")
    sys.exit(1)

# ======== PRE-MARKET SCAN ========
results = []
total_scanned = 0
affordable = 0
technical_passed = 0
volume_passed = 0
sentiment_passed = 0  # Placeholder for future
rsi_passed = 0        # Placeholder for future
final_selected = 0

MIN_VOLUME = 300000  # 5 lakh shares
MIN_ATR = 1.2  # ₹2 minimum daily range

print("\n🚀 Starting pre-market scan...")
for idx, row in nifty100_df.iterrows():
    total_scanned += 1
    symbol = row["base_symbol"]
    secid = str(row["SEM_SMST_SECURITY_ID"])  
    print(f"\n🔍 [{len(results)+1}/{len(nifty100_df)}] Scanning {symbol}")

    try:
        # Step 1: Get pre-market LTP
        ltp = get_live_price(symbol, secid, premarket=True)       
        # NEW: Fetch yesterday's close for bullish confirmation
        try:
            quote_url = f"https://api.dhan.co/quotes/isin?security_id={secid}&exchange=NSE"
            quote_resp = requests.get(quote_url, headers=HEADERS, timeout=5)
            quote_data = quote_resp.json()
            prev_close = float(quote_data.get("previousClose", 0))
        
            if ltp <= prev_close * 0.995:
                print(f"⛔ Not bullish: LTP ₹{ltp:.2f} ≤ Prev Close ₹{prev_close:.2f}")
                continue
        except Exception as e:
            print(f"⚠️ Close fetch failed: {str(e)[:60]}")
            continue
        
        # Step 2: Volume check (last 5 trading days)
        payload = {
            "securityId": secid,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",  # ADD THIS MISSING FIELD
            "interval": "1",
            "oi": False,  # ADD THIS MISSING FIELD
            "fromDate": (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S'),  # ADD TIME
            "toDate": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')  # USE YESTERDAY
        }
        vol_response = requests.post(
            "https://api.dhan.co/v2/charts/intraday", 
            headers=HEADERS, 
            json=payload,
            timeout=10
        )
        
        if vol_response.status_code != 200:
            print(f"❌ Volume API error: {vol_response.status_code}")
            continue
            
        vol_data = vol_response.json()
        df_vol = pd.DataFrame({
            "timestamp": pd.to_datetime(vol_data["timestamp"], unit="s"),
            "volume": vol_data["volume"]
        })
        df_vol["date"] = df_vol["timestamp"].dt.date
        daily_vol = df_vol.groupby("date")["volume"].sum()
        avg_volume = daily_vol.tail(5).mean()
        
        if pd.isna(avg_volume) or avg_volume < MIN_VOLUME:
            print(f"⛔ Low volume: {avg_volume:,.0f} < {MIN_VOLUME:,.0f}")
            continue
        volume_passed += 1
        
        # Step 3: Volatility check (ATR proxy)
        if "high" in vol_data and "low" in vol_data:
            df_vol["high"] = vol_data["high"]
            df_vol["low"] = vol_data["low"]
            df_vol["range"] = df_vol["high"] - df_vol["low"]
            daily_range = df_vol.groupby("date")["range"].max()
            atr = daily_range.tail(5).mean()
        else:
            print(f"⛔ Missing high/low data for volatility check")
            continue
        
        if pd.isna(atr) or atr < MIN_ATR:
            print(f"⛔ Low volatility: ₹{atr:.2f} < ₹{MIN_ATR:.2f}")
            continue
        technical_passed += 1
        
        # Step 4: Calculate position size
        quantity = int(CAPITAL // ltp)
        if quantity <= 0:
            print(f"⛔ Unaffordable: ₹{ltp:,.2f} > ₹{CAPITAL:,.2f}")
            continue
        affordable += 1        
        capital_used = quantity * ltp
        
        final_selected += 1
        results.append({
            "symbol": symbol,
            "security_id": secid,
            "ltp": ltp,
            "quantity": quantity,
            "capital_used": capital_used,
            "avg_volume": int(avg_volume),
            "avg_range": round(atr, 2),
            "potential_profit": round(quantity * atr, 2)
        })
        print(f"✅ SELECTED: ₹{ltp:,.2f} | Vol: {avg_volume:,.0f} | Range: ₹{atr:.2f}")

    except Exception as e:
        print(f"⚠️ Processing error: {str(e)[:70]}")
    finally:
        time.sleep(0.3)  # Rate limit protection

# ======== SAVE RESULTS ========
if results:
    results_df = pd.DataFrame(results)
    # Prioritize high volatility and volume
    results_df["priority_score"] = results_df["avg_range"] * results_df["avg_volume"]
    results_df = results_df.sort_values("priority_score", ascending=False)
    # 🔗 Merge sector info from master_df before saving
    sector_map = master_df.set_index("base_symbol")["sector"].to_dict()
    results_df["sector"] = results_df["symbol"].map(sector_map)   
    results_df.to_csv(OUTPUT_CSV, index=False)   
    log_to_postgres(datetime.now(), "Test_dynamic_stock_generator.py", "SUCCESS", f"{len(results_df)} stocks saved to dynamic_stock_list and DB.")
    log_dynamic_stock_list(results_df)
    print(f"\n✅ Saved {len(results_df)} stocks to {OUTPUT_CSV}")
    # 📄 Save summary to filter_summary_log.csv
    summary_path = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"
    summary_row = {
        "date": datetime.now().strftime("%m/%d/%Y %H:%M"),
        "Script_Name": "dynamic_stock_generator.py",
        "total_scanned": total_scanned,
        "affordable": affordable,
        "technical_passed": technical_passed,
        "volume_passed": volume_passed,
        "sentiment_passed": sentiment_passed,
        "rsi_passed": rsi_passed,
        "final_selected": final_selected
    }
    
    try:
        if os.path.exists(summary_path):
            pd.DataFrame([summary_row]).to_csv(summary_path, mode='a', header=False, index=False)
        else:
            pd.DataFrame([summary_row]).to_csv(summary_path, mode='w', header=True, index=False)
    except Exception as e:
        print(f"⚠️ Could not write to filter_summary_log.csv: {e}")
    
    print(f"📊 Top 5 opportunities:")
    print(results_df[["symbol", "ltp", "quantity", "potential_profit"]].head().to_string(index=False))
else:
    print("\n❌ No stocks passed all filters")
    log_to_postgres(datetime.now(), "Test_dynamic_stock_generator.py", "WARNING", "No stocks selected today.")

# ======== PERFORMANCE METRICS ========
elapsed = datetime.now() - start_time
print(f"\n⏱️ Total scan time: {elapsed.total_seconds():.1f} seconds")
print(f"💵 Capital available: ₹{CAPITAL:,.2f}")
print(f"📈 Potential positions: {len(results)}")
# 📝 Save all captured print outputs to a .txt log file
with open("D:/Downloads/Dhanbot/dhan_autotrader/Logs/dynamic_stock_generator.txt", "w", encoding="utf-8") as f:
    f.write(log_buffer.getvalue())