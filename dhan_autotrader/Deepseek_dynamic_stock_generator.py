# dynamic_stock_generator.py
import os
import sys
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import time
from nsepython import nsefetch
from db_logger import log_dynamic_stock_list, log_to_postgres
import io
import sys
from dhan_api import get_live_price, get_historical_price
import random 
import numpy as np

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
        url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100'
        data = nsefetch(url)
        
        symbols = []
        for item in data["data"]:
            symbol = item["symbol"].strip().upper()
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

    positive_gainers = [(s, g) for s, g in sector_gains.items() if g > 0]
    top_sectors = [s for s, _ in positive_gainers]
    
    print("📈 Top sectors today:", top_sectors)
    return [s[0] for s in top_sectors]

# ======== MARKET TREND ANALYSIS ========
def is_index_bullish(index_id, segment):
    """Check if index is bullish using RSI and MACD"""
    try:
        # Get intraday data for index
        url = "https://api.dhan.co/v2/charts/intraday"
        now = datetime.now()
        
        # Handle market hours - indices only available during trading hours
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if now < market_open or now > market_close:
            print("⚠️ Outside market hours for index data")
            return False
        
        # Use date-only format for indices
        today_str = now.strftime("%Y-%m-%d")
        payload = {
            "securityId": str(index_id),
            "exchangeSegment": segment,
            "instrument": "INDEX",
            "interval": "15",
            "fromDate": today_str,
            "toDate": today_str
        }
        
        # For indices, API expects date-only format
        print(f"🔍 Fetching index data for {today_str}")
        
        response = requests.post(url, headers=HEADERS, json=payload, timeout=10)
        if response.status_code != 200:
            error_msg = response.text[:100] if response.text else "No response text"
            print(f"⚠️ Index data fetch failed ({response.status_code}): {error_msg}")
            return False
            
        data = response.json()
        if not data or 'close' not in data or len(data['close']) < 5:
            print("⚠️ Insufficient index candle data")
            return False
            
        closes = pd.Series(data['close']).astype(float)
        
        # Calculate RSI (14-period)
        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean().bfill()
        avg_loss = loss.rolling(14).mean().bfill().replace(0, 0.01)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]
        
        # Calculate MACD (12,26,9)
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # Bullish if RSI > 50 and MACD histogram is positive and increasing
        return (rsi_value > 50) and (histogram.iloc[-1] > 0) and (histogram.iloc[-1] > histogram.iloc[-2])
        
    except Exception as e:
        print(f"⚠️ is_index_bullish failed: {str(e)[:100]}")
        return False

# ======== LOAD MASTER SECURITY LIST ========
try:
    master_df = pd.read_csv(MASTER_CSV)
    symbol_sector_map = build_sector_map(nifty100_symbols)
    master_df["base_symbol"] = master_df["SEM_TRADING_SYMBOL"].str.replace("-EQ", "").str.strip().str.upper()
    master_df["sector"] = master_df["base_symbol"].map(symbol_sector_map)
    top_sectors = get_sector_strength()
    
    # Get Nifty 50 index ID for trend check
    nifty50_row = master_df[
        master_df["base_symbol"].str.upper().str.contains("NIFTY") &
        master_df["SM_SYMBOL_NAME"].str.upper().str.contains("50")
    ]
    if not nifty50_row.empty:
        nifty50_id = str(nifty50_row.iloc[0]["SEM_SMST_SECURITY_ID"])
        nifty_segment = "IDX_I"
    else:
        print("❌ Nifty 50 index not found in master. Skipping trend check.")
        nifty50_id = None
        nifty_segment = None
    
    # Check market trend for Nifty 50 using correct segment
    if nifty50_id and nifty_segment:
        nifty_bullish = is_index_bullish(nifty50_id, nifty_segment)
    else:
        nifty_bullish = False
    
    market_status = "Bullish" if nifty_bullish else "Bearish"
    print(f"📊 Market Trend: {market_status}")
    
    nifty100_df = master_df[
        (master_df["base_symbol"].isin(nifty100_symbols))
    ]   
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
    sector = row.get("sector", "UNKNOWN")
    print(f"\n🔍 [{count}/{len(nifty100_df)}] Scanning {symbol} (Sector: {sector})")
    
    # Skip stocks not in top sectors during bearish market
    if not nifty_bullish and sector not in top_sectors:
        print(f"⏩ Skipping {symbol} - not in top sectors during bearish market")
        continue
    
    try:
        # Step 1: Get pre-market LTP
        ltp = get_live_price(symbol, secid, premarket=True)       
        
        # Fetch yesterday's close and open price from live quote API
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
            open_price = float(quote_data.get("openPrice", 0))
    
            # ✅ Smart Gap-Up Rejection Logic
            if prev_close > 0 and open_price > prev_close * 1.03:
                if ltp < open_price:  # No follow-through after gap-up
                    print(f"⛔ Gap-up trap: Open ₹{open_price:.2f} > Prev Close ₹{prev_close:.2f} but LTP ₹{ltp:.2f} dropped")
                    continue
    
            # ✅ Reject weak gap-downs
            if ltp <= prev_close * 0.995:
                print(f"⛔ Not bullish: LTP ₹{ltp:.2f} ≤ Prev Close ₹{prev_close:.2f}")
                continue
    
        except Exception as e:
            print(f"⚠️ Close fetch failed: {str(e)[:60]}")
            continue
    
        # Step 2: Volume check (last 5 trading days)
        try:
            url = "https://api.dhan.co/v2/charts/intraday"
            now = datetime.now()
            from_date = (now - timedelta(days=5)).replace(hour=9, minute=30, second=0, microsecond=0)
            to_date = (now - timedelta(days=1)).replace(hour=15, minute=30, second=0, microsecond=0)
            from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
            to_date_str = to_date.strftime("%Y-%m-%d %H:%M:%S")
            
            payload = {
                "securityId": secid,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "interval": "5",
                "oi": "false",
                "fromDate": from_date_str,
                "toDate": to_date_str
            }
        
            # Implement retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=HEADERS, json=payload, timeout=10)
                    
                    if response.status_code == 429:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⏳ Rate limited. Waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    
                    if response.status_code != 200:
                        print(f"❌ Historical fetch failed for {symbol} (ID: {secid}): {response.status_code} - {response.text}")
                        break
        
                    data = response.json()
                    required_keys = {"open", "high", "low", "close", "volume", "timestamp"}
                    if not required_keys.issubset(data.keys()):
                        print(f"❌ Missing required candle fields for {symbol}")
                        break
                    
                    df_vol = pd.DataFrame({
                        "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata"),
                        "open": data["open"],
                        "high": data["high"],
                        "low": data["low"],
                        "close": data["close"],
                        "volume": data["volume"]
                    })
        
                    df_vol["date"] = df_vol["timestamp"].dt.date
                    daily_vol = df_vol.groupby("date")["volume"].sum()
                    avg_volume = daily_vol.tail(5).mean()
                
                    if pd.isna(avg_volume) or avg_volume < MIN_VOLUME:
                        print(f"⛔ Low volume: {avg_volume:,.0f} < {MIN_VOLUME:,.0f}")
                        break
                
                    volume_passed += 1
                    df_vol["range"] = df_vol["high"] - df_vol["low"]
                    daily_range = df_vol.groupby("date")["range"].max()
                    atr = daily_range.tail(5).mean()
                
                    if pd.isna(atr) or atr < MIN_ATR:
                        print(f"⛔ Low volatility: ₹{atr:.2f} < ₹{MIN_ATR:.2f}")
                        break
                    
                    technical_passed += 1
                    break
                    
                except Exception as e:
                    print(f"⚠️ Volume/ATR check attempt {attempt+1} failed: {str(e)[:70]}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                continue
                
        except Exception as e:
            print(f"⚠️ Volume/ATR check failed after retries: {str(e)[:70]}")
            continue
            
        # ====== SMA and RSI Check ======
        sma_20 = None
        rsi_value = None
        technical_ok = False
        
        try:
            from_date = (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d 09:15:00')
            to_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d 15:30:00')
            max_retries = 3
            rejection_printed = False
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
                continue
                
        except Exception as e:
            print(f"⚠️ Technical indicator error: {str(e)[:70]}")
            continue
            
        if not technical_ok:
            continue

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
            "sma_20": sma_20,
            "rsi": rsi_value,
            "sector": sector,
            "market_trend": "Bullish" if nifty_bullish else "Bearish",
            "in_top_sector": sector in top_sectors,
            "stock_origin": "Dynamic"
        })
        print(f"✅ SELECTED: ₹{ltp:,.2f} | Vol: {avg_volume:,.0f} | Range: ₹{atr:.2f}")

    except Exception as e:
        print(f"⚠️ Processing error: {str(e)[:70]}")
    finally:
        time.sleep(0.5)

# ======== SAVE RESULTS ========
if results:
    results_df = pd.DataFrame(results)
    
    # Prioritize high volatility and volume with sector boost
    results_df["sector_boost"] = results_df["in_top_sector"].astype(int) * 0.5  # 50% boost
    results_df["priority_score"] = results_df["avg_range"] * results_df["avg_volume"] * (1 + results_df["sector_boost"])
    results_df = results_df.sort_values("priority_score", ascending=False)
    
    results_df.to_csv(OUTPUT_CSV, index=False)   
    log_to_postgres(datetime.now(), "Test_dynamic_stock_generator.py", "SUCCESS", 
                   f"{len(results_df)} stocks saved to dynamic_stock_list. Market: {'Bullish' if nifty_bullish else 'Bearish'}")
    log_dynamic_stock_list(results_df)
    
    # ====== Trending Nifty 100 Gainers ======
    print("\n🚀 Adding trending Nifty 100 gainers to boost trade pool...")
    trending_additions = []
    
    for _, row in nifty100_df.iterrows():
        symbol = row["base_symbol"]
        secid = str(row["SEM_SMST_SECURITY_ID"])
        sector = row.get("sector", "UNKNOWN")
    
        try:
            quote_url = f"https://api.dhan.co/quotes/isin?security_id={secid}&exchange=NSE"
            quote_resp = requests.get(quote_url, headers=HEADERS, timeout=5)
            if quote_resp.status_code != 200:
                continue
            data = quote_resp.json()
            open_price = float(data.get("openPrice", 0))
            ltp = float(data.get("lastTradedPrice", 0))
            if open_price <= 0 or ltp <= 0:
                continue
    
            pct_gain = ((ltp - open_price) / open_price) * 100
    
            if pct_gain < 2:
                continue
    
            if symbol in results_df["symbol"].values:
                continue
    
            # Get RSI
            from_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d 09:15:00')
            to_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d 15:30:00')
            payload = {
                "securityId": secid,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "interval": "1",
                "oi": "false",
                "fromDate": from_date,
                "toDate": to_date
            }
            response = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload, timeout=10)
            if response.status_code != 200:
                continue
            hist = response.json()
            closes = pd.Series(hist["close"])
            if closes.empty or len(closes) < 14:
                continue
            delta = closes.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean().bfill()
            avg_loss = loss.rolling(14).mean().bfill().replace(0, 0.01)
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]
    
            if rsi_val >= 70:
                continue
    
            quantity = int(CAPITAL // ltp)
            if quantity <= 0:
                continue
    
            capital_used = quantity * ltp
            trending_additions.append({
                "symbol": symbol,
                "security_id": secid,
                "ltp": ltp,
                "quantity": quantity,
                "capital_used": capital_used,
                "avg_volume": 0,
                "avg_range": 0,
                "potential_profit": round(quantity * 2, 2),
                "sma_20": None,
                "rsi": rsi_val,
                "sector": sector,
                "market_trend": "Bullish" if nifty_bullish else "Bearish",
                "in_top_sector": sector in top_sectors,
                "priority_score": pct_gain * 1000,
                "stock_origin": "Trending"
            })
    
        except Exception as e:
            print(f"⚠️ Trending check failed for {symbol}: {str(e)[:60]}")
            continue
    
    # Merge into existing result set
    if trending_additions:
        trending_df = pd.DataFrame(trending_additions)
        results_df = pd.concat([results_df, trending_df], ignore_index=True)
        results_df = results_df.sort_values("priority_score", ascending=False)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Added {len(trending_additions)} trending stocks to dynamic_stock_list.csv")
    else:
        print("❌ No trending stocks passed filters.")
    
    print(f"\n✅ Saved {len(results_df)} stocks to {OUTPUT_CSV}")
    # Save summary
    summary_path = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"
    summary_row = {
        "date": datetime.now().strftime("%m/%d/%Y %H:%M"),
        "Script_Name": "dynamic_stock_generator.py",
        "total_scanned": total_scanned,
        "affordable": affordable,
        "technical_passed": technical_passed,
        "volume_passed": volume_passed,
        "sentiment_passed": sentiment_passed,
        "sma_passed": sma_passed,
        "rsi_passed": rsi_passed,
        "final_selected": final_selected,
        "market_trend": "Bullish" if nifty_bullish else "Bearish"
    }
    
    try:
        if os.path.exists(summary_path):
            pd.DataFrame([summary_row]).to_csv(summary_path, mode='a', header=False, index=False)
        else:
            pd.DataFrame([summary_row]).to_csv(summary_path, mode='w', header=True, index=False)
    except Exception as e:
        print(f"⚠️ Could not write to filter_summary_log.csv: {e}")
    
    print(f"📊 Top 5 opportunities:")
    print(results_df[["symbol", "ltp", "quantity", "potential_profit", "sector"]].head().to_string(index=False))
else:
    print("\n❌ No stocks passed all filters")
    log_to_postgres(datetime.now(), "Test_dynamic_stock_generator.py", "WARNING", 
                   f"No stocks selected today. Market: {'Bullish' if nifty_bullish else 'Bearish'}")

# ======== PERFORMANCE METRICS ========
elapsed = datetime.now() - start_time
total_sec = elapsed.total_seconds()
minutes = int(total_sec // 60)
seconds = int(total_sec % 60)
print(f"\n⏱️ Total scan time: {minutes} min {seconds} sec")
print(f"💵 Capital available: ₹{CAPITAL:,.2f}")
print(f"📈 Potential positions: {len(results)}")
# Save log
with open("D:/Downloads/Dhanbot/dhan_autotrader/Logs/dynamic_stock_generator.txt", "w", encoding="utf-8") as f:
    f.write(log_buffer.getvalue())