# premarket_nifty100_scanner.py
import os
import sys
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import time
from nsepython import nsefetch  # Using your working library
from db_logger import log_dynamic_stock_list, log_to_postgres
import io
import sys
from dhan_api import get_live_price, get_historical_price
import random 

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
        
        # Skip index representation and extract symbols
        symbols = []
        for item in data["data"]:
            symbol = item["symbol"].strip().upper()
            # Skip index representation like "NIFTY 100"
            if "NIFTY" not in symbol:
                symbols.append(symbol)
                
        print(f"✅ Fetched {len(symbols)} valid Nifty 100 constituents")
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
sentiment_passed = 0
sma_passed = 0
rsi_passed = 0
final_selected = 0

MIN_VOLUME = 300000  # 5 lakh shares
MIN_ATR = 1.2  # ₹2 minimum daily range

print("\n🚀 Starting pre-market scan...")
for count, (_, row) in enumerate(nifty100_df.iterrows(), start=1):
    total_scanned += 1
    symbol = row["base_symbol"]
    secid = str(row["SEM_SMST_SECURITY_ID"])
    print(f"\n🔍 [{count}/{len(nifty100_df)}] Scanning {symbol}")

    try:
        # Step 1: Get pre-market LTP
        ltp = get_live_price(symbol, secid, premarket=True)       
        
        # NEW: Fetch yesterday's close for bullish confirmation
        try:
            quote_url = f"https://api.dhan.co/quotes/isin?security_id={secid}&exchange=NSE"
            quote_resp = requests.get(quote_url, headers=HEADERS, timeout=5)
            
            # Handle rate limiting for quote API
            if quote_resp.status_code == 429:
                wait_time = 5 + random.uniform(0, 2)
                print(f"⏳ Rate limited on quote API. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                # Retry once
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
        try:
            url = "https://api.dhan.co/v2/charts/intraday"
            # Use naive datetime without timezone (API requirement)
            now = datetime.now()
            
            # Format dates without timezone
            from_date = (now - timedelta(days=5)).replace(hour=9, minute=30, second=0, microsecond=0)
            to_date = (now - timedelta(days=1)).replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Convert to API-compatible string format
            from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
            to_date_str = to_date.strftime("%Y-%m-%d %H:%M:%S")
            
            payload = {
                "securityId": secid,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "interval": "5",  # 5-minute candles
                "oi": "false",
                "fromDate": from_date_str,
                "toDate": to_date_str
            }
        
            # Implement retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=HEADERS, json=payload, timeout=10)
                    
                    # Handle rate limiting (429 errors)
                    if response.status_code == 429:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⏳ Rate limited. Waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    
                    if response.status_code != 200:
                        # Print the full response text for debugging
                        print(f"❌ Historical fetch failed for {symbol} (ID: {secid}): {response.status_code} - {response.text}")
                        break
        
                    
                    data = response.json()
                    
                    # Validate response structure
                    required_keys = {"open", "high", "low", "close", "volume", "timestamp"}
                    if not required_keys.issubset(data.keys()):
                        print(f"❌ Missing required candle fields for {symbol}")
                        break
                    
                    # Build DataFrame with proper timezone handling
                    df_vol = pd.DataFrame({
                        "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata"),
                        "open": data["open"],
                        "high": data["high"],
                        "low": data["low"],
                        "close": data["close"],
                        "volume": data["volume"]
                    })
        
                    # Extract date for grouping
                    df_vol["date"] = df_vol["timestamp"].dt.date
                    
                    # Compute average daily volume over last 5 days
                    daily_vol = df_vol.groupby("date")["volume"].sum()
                    avg_volume = daily_vol.tail(5).mean()
                
                    if pd.isna(avg_volume) or avg_volume < MIN_VOLUME:
                        print(f"⛔ Low volume: {avg_volume:,.0f} < {MIN_VOLUME:,.0f}")
                        break
                
                    volume_passed += 1
                    
                    # Step 3: Volatility check (ATR proxy) - using same data
                    df_vol["range"] = df_vol["high"] - df_vol["low"]
                    daily_range = df_vol.groupby("date")["range"].max()
                    atr = daily_range.tail(5).mean()
                
                    if pd.isna(atr) or atr < MIN_ATR:
                        print(f"⛔ Low volatility: ₹{atr:.2f} < ₹{MIN_ATR:.2f}")
                        break
                    
                    technical_passed += 1
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    print(f"⚠️ Volume/ATR check attempt {attempt+1} failed: {str(e)[:70]}")
                    if attempt == max_retries - 1:
                        raise  # Re-raise exception on final attempt
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                continue  # Skip to next stock if all retries failed
                
        except Exception as e:
            print(f"⚠️ Volume/ATR check failed after retries: {str(e)[:70]}")
            continue
            
        # ====== SMA and RSI Check (Improved) ======
        # Initialize technical indicators to None
        sma_20 = None
        rsi_value = None
        technical_ok = False
        
        try:
            # Get historical data with proper datetime formatting
            from_date = (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d 09:15:00')
            to_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d 15:30:00')
            
            # Implement retry logic for historical price API
            # Implement retry logic for historical price API
            max_retries = 3
            rejection_printed = False  # Avoid repeated rejection messages
            for attempt in range(max_retries):
                try:
                    url = "https://api.dhan.co/v2/charts/intraday"
                    payload = {
                        "securityId": secid,
                        "exchangeSegment": "NSE_EQ",
                        "instrument": "EQUITY",
                        "interval": "1",
                        "oi": "false",
                        "fromDate": from_date,
                        "toDate": to_date
                    }
            
                    time.sleep(0.5 + random.uniform(0, 0.5))
                    response = requests.post(url, headers=HEADERS, json=payload, timeout=15)
            
                    if response.status_code == 429:
                        wait_time = 5 + random.uniform(0, 2)
                        print(f"⏳ Rate limited. Waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
            
                    if response.status_code != 200:
                        if not rejection_printed:
                            print(f"❌ Historical fetch failed: {response.status_code} - {response.text[:100]}")
                            rejection_printed = True
                        continue
            
                    data = response.json()
                    required_keys = {"open", "high", "low", "close", "volume", "timestamp"}
                    if not isinstance(data, dict) or not all(key in data for key in required_keys):
                        if not rejection_printed:
                            print(f"❌ Invalid or incomplete response structure for {symbol}")
                            rejection_printed = True
                        continue
            
                    df_hist = pd.DataFrame({
                        "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata"),
                        "open": data["open"],
                        "high": data["high"],
                        "low": data["low"],
                        "close": data["close"],
                        "volume": data["volume"]
                    })
            
                    if df_hist.empty:
                        if not rejection_printed:
                            print(f"⚠️ Empty DataFrame for {symbol}")
                            rejection_printed = True
                        continue
            
                    df_hist = df_hist.sort_values('timestamp')
                    closes = df_hist['close'].astype(float)
            
                    if len(closes) < 20:
                        if not rejection_printed:
                            print(f"⛔ Insufficient data for SMA20: {len(closes)} days")
                            rejection_printed = True
                        continue
                    sma_20 = closes.tail(20).mean()
            
                    if len(closes) < 15:
                        if not rejection_printed:
                            print(f"⛔ Insufficient data for RSI: {len(closes)} days")
                            rejection_printed = True
                        continue
            
                    delta = closes.diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(14).mean().bfill()
                    avg_loss = loss.rolling(14).mean().bfill()
                    avg_loss = avg_loss.replace(0, 0.01)
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_value = rsi.iloc[-1]
            
                    if ltp < sma_20:
                        if not rejection_printed:
                            print(f"⛔ Below SMA20: ₹{ltp:.2f} < ₹{sma_20:.2f}")
                            rejection_printed = True
                        continue
                    sma_passed += 1
            
                    if rsi_value < 50 or rsi_value > 70:
                        if not rejection_printed:
                            print(f"⛔ RSI out of range: {rsi_value:.2f}")
                            rejection_printed = True
                        continue
                    rsi_passed += 1
            
                    technical_ok = True
                    break
            
                except Exception as e:
                    print(f"⚠️ Technical indicator attempt {attempt+1} error: {str(e)[:70]}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⏳ Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
            else:
                continue  # Skip to next stock if all retries failed
            
                
        except Exception as e:
            print(f"⚠️ Technical indicator error: {str(e)[:70]}")
            continue
            
        # Skip if technical checks didn't pass
        if not technical_ok:
            continue
        # ====== END SECTION ======

        
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
            "potential_profit": round(quantity * atr, 2),
            "sma_20": sma_20,          # NEW FIELD
            "rsi": rsi_value            # NEW FIELD (corrected from rsi to rsi_value)
        })
        print(f"✅ SELECTED: ₹{ltp:,.2f} | Vol: {avg_volume:,.0f} | Range: ₹{atr:.2f}")

    except Exception as e:
        print(f"⚠️ Processing error: {str(e)[:70]}")
    finally:
        # Add jitter to avoid regular request patterns
        time.sleep(0.5)

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
        "sma_passed": sma_passed,   # NEW COUNTER
        "rsi_passed": rsi_passed,   # UPDATED COUNTER
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
total_sec = elapsed.total_seconds()
minutes = int(total_sec // 60)
seconds = int(total_sec % 60)
print(f"\n⏱️ Total scan time: {minutes} min {seconds} sec")
print(f"💵 Capital available: ₹{CAPITAL:,.2f}")
print(f"📈 Potential positions: {len(results)}")
# 📝 Save all captured print outputs to a .txt log file
with open("D:/Downloads/Dhanbot/dhan_autotrader/Logs/dynamic_stock_generator.txt", "w", encoding="utf-8") as f:
    f.write(log_buffer.getvalue())