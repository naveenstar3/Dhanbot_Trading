# consolidated_dynamic_stock_generator.py
import os
import sys
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import time
from nsepython import nsefetch
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator
import io
import random
from dhan_api import get_live_price
import logging

# ======== CONSTANTS ========
RSI_MIN = 45
RSI_MAX = 70
GAP_UP_THRESHOLD = 0.01  # 1% gap-up threshold
MIN_VOLUME = 300000
MIN_ATR = 1.2
SMALLCAP_MIN_VOLUME = 500000
SMALLCAP_MIN_ATR = 2.0
SMALLCAP_MAX_RSI = 70

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

sys.stdout = TeeLogger(sys.__stdout__, log_buffer)

# ======== CONFIGURATION ========
start_time = datetime.now()
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

# CAPITAL is now loaded from config.json instead of CSV
try:
    CAPITAL = float(config.get("capital", 0))
    if CAPITAL <= 0:
        raise ValueError("Capital must be greater than zero.")
    print(f"💰 Capital Loaded from config: ₹{CAPITAL:,.2f}")
except Exception as e:
    print(f"❌ Capital loading from config failed: {e}")
    sys.exit(1)


# ======== RELIABLE NIFTY 100 FETCH ========
def get_nifty100_constituents():
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
        "NIFTY PHARMA", "NIFTY REALTY", "NIFTY METAL", "NIFTY ENERGY", "NIFTY CONSUMER DURABLES",
        "NIFTY OIL & GAS", "NIFTY MEDIA", "NIFTY HEALTHCARE INDEX"
    ]
    sector_gains = {}
    market_condition = "bullish"
    
    # Get Nifty 50 performance for market condition
    try:
        nifty_url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        nifty_data = nsefetch(nifty_url)
        nifty_change = float(nifty_data["data"][0]["pChange"])
        market_condition = "bearish" if nifty_change < -0.3 else "bullish"
        print(f"📊 Market Condition: {market_condition.upper()} (Nifty Change: {nifty_change:.2f}%)")
    except Exception as e:
        print(f"⚠️ Nifty 50 fetch failed: {str(e)[:50]}")

    for sector in sector_indices:
        try:
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={sector.replace(' ', '%20')}"
            data = nsefetch(url)
            pChange = float(data["data"][0]["pChange"])
            sector_gains[sector] = pChange
        except Exception as e:
            print(f"⚠️ Sector fetch failed: {sector} → {str(e)[:50]}")

    # Dynamic sector selection
    if market_condition == "bearish":
        top_sectors = [s for s, g in sector_gains.items() if g > 0]
        if not top_sectors:
            top_sectors = sorted(sector_gains.items(), key=lambda x: x[1], reverse=True)[:3]
            top_sectors = [s[0] for s in top_sectors]
        print("🐻 Bearish market - Dynamic positive sectors:", top_sectors)
    else:
        top_sectors = [s for s, g in sector_gains.items() if g > 0]
        if not top_sectors:
            top_sectors = list(sector_gains.keys())
        print("🐂 Bullish market - Dynamic positive sectors:", top_sectors)   
    
    return top_sectors, market_condition, sector_gains
    
# ======== TECHNICAL INDICATORS ========
def calculate_macd(closes):
    if len(closes) < 26:
        return 0, 0, False
    
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    macd_crossover = macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
    
    return macd.iloc[-1], histogram.iloc[-1], macd_crossover

def calculate_rsi(closes):
    if len(closes) < 14:
        return 50
    
    delta = closes.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean().bfill()
    avg_loss = loss.rolling(14).mean().bfill().replace(0, 0.01)
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.iloc[-1]  # Return only the last RSI value

# ======== LOAD MASTER SECURITY LIST ========
try:
    master_df = pd.read_csv(MASTER_CSV)
    symbol_sector_map = build_sector_map(nifty100_symbols)
    master_df["base_symbol"] = master_df["SEM_TRADING_SYMBOL"].str.replace("-EQ", "").str.strip().str.upper()
    master_df["sector"] = master_df["base_symbol"].map(symbol_sector_map)
    top_sectors, market_condition, sector_strengths = get_sector_strength()
    
    # Filter to Nifty 100 and top sectors
    nifty100_df = master_df[
        (master_df["base_symbol"].isin(nifty100_symbols)) &
        (master_df["sector"].isin(top_sectors))
    ]
    print(f"📊 Master list filtered to {len(nifty100_df)} Nifty 100 stocks in focus sectors")
    
    if len(nifty100_df) == 0:
        print("❌ No matching securities found in master list")
        sys.exit(1)
except Exception as e:
    print(f"❌ Master CSV load failed: {e}")
    sys.exit(1)

# ======== PRE-MARKET SCAN (NIFTY100) ========
nifty100_results = []
total_scanned = 0
affordable = 0
technical_passed = 0
volume_passed = 0
sentiment_passed = 0
sma_passed = 0
rsi_passed = 0
final_selected = 0

print("\n🚀 Starting Nifty100 pre-market scan...")
for count, (_, row) in enumerate(nifty100_df.iterrows(), start=1):
    total_scanned += 1
    symbol = row["base_symbol"]
    secid = str(row["SEM_SMST_SECURITY_ID"])
    sector = row["sector"]
    print(f"\n🔍 [{count}/{len(nifty100_df)}] Scanning NIFTY100: {symbol} ({sector})")

    try:
        # Step 1: Get pre-market LTP
        ltp = get_live_price(symbol, secid, premarket=True)       
        
        # Fetch yesterday's close and open price
        try:
            quote_url = f"https://api.dhan.co/quotes/isin?security_id={secid}"
            quote_resp = requests.get(quote_url, headers=HEADERS, timeout=5)
            
            # Handle rate limiting
            if quote_resp.status_code == 429:
                wait_time = 5 + random.uniform(0, 2)
                print(f"⏳ Rate limited on quote API. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                quote_resp = requests.get(quote_url, headers=HEADERS, timeout=5)
            
            quote_data = quote_resp.json()
            prev_close = float(quote_data.get("previousClose", 0))
            open_price = float(quote_data.get("openPrice", 0))
    
            # Smart Gap-Up Rejection Logic (1% threshold)
            if prev_close > 0 and open_price > prev_close * (1 + GAP_UP_THRESHOLD):
                if ltp < open_price:  # No follow-through
                    print(f"⛔ Gap-up trap: Open ₹{open_price:.2f} > Prev Close ₹{prev_close:.2f} but LTP ₹{ltp:.2f} dropped")
                    continue
    
            # Reject weak gap-downs in bullish markets
            if market_condition == "bullish" and ltp <= prev_close * 0.995:
                print(f"⛔ Not bullish: LTP ₹{ltp:.2f} ≤ Prev Close ₹{prev_close:.2f}")
                continue
    
        except Exception as e:
            print(f"⚠️ Close fetch failed: {str(e)[:60]}")
            continue
    
        # Step 2: Volume and volatility check
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
        
            # Retry logic with exponential backoff
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
                    
                    # Volatility check (ATR proxy)
                    df_vol["range"] = df_vol["high"] - df_vol["low"]
                    daily_range = df_vol.groupby("date")["range"].max()
                    atr = daily_range.tail(5).mean()
                
                    if pd.isna(atr) or atr < MIN_ATR:
                        print(f"⛔ Low volatility: ₹{atr:.2f} < ₹{MIN_ATR:.2f}")
                        break
                    
                    technical_passed += 1
                    break  # Success
                    
                except Exception as e:
                    print(f"⚠️ Volume/ATR check attempt {attempt+1} failed: {str(e)[:70]}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                continue  # Skip to next stock
                
        except Exception as e:
            print(f"⚠️ Volume/ATR check failed after retries: {str(e)[:70]}")
            continue
            
        # ====== SMA, RSI, and MACD Check ======
        sma_20 = None
        rsi_value = None
        macd_value = None
        macd_hist = None
        macd_crossover = False
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
                            print(f"❌ Invalid response structure for {symbol}")
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
            
                    # Calculate RSI (45-70 range for all market conditions)
                    rsi_value = calculate_rsi(closes)
                    
                    # Calculate MACD
                    macd_value, macd_hist, macd_crossover = calculate_macd(closes)
                    
                    # Standardized RSI check
                    if rsi_value < RSI_MIN or rsi_value > RSI_MAX:
                        if not rejection_printed:
                            print(f"⛔ RSI out of range: {rsi_value:.2f} (Allowed: {RSI_MIN}-{RSI_MAX})")
                            rejection_printed = True
                        continue
                    rsi_passed += 1
            
                    if ltp < sma_20:
                        if not rejection_printed:
                            print(f"⛔ Below SMA20: ₹{ltp:.2f} < ₹{sma_20:.2f}")
                            rejection_printed = True
                        continue
                    sma_passed += 1
            
                    technical_ok = True
                    break
            
                except Exception as e:
                    print(f"⚠️ Technical indicator attempt {attempt+1} error: {str(e)[:70]}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⏳ Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
            else:
                continue  # Skip to next stock
                
        except Exception as e:
            print(f"⚠️ Technical indicator error: {str(e)[:70]}")
            continue
            
        if not technical_ok:
            continue
            
        # Step 4: Position sizing
        quantity = int(CAPITAL // ltp)
        if quantity <= 0:
            print(f"⛔ Unaffordable: ₹{ltp:,.2f} > ₹{CAPITAL:,.2f}")
            continue
        affordable += 1        
        capital_used = quantity * ltp
        
        final_selected += 1
        nifty100_results.append({
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
            "macd": macd_value,
            "macd_hist": macd_hist,
            "macd_crossover": int(macd_crossover),
            "sector": sector,
            "sector_strength": sector_strengths.get(sector, 0),
            "stock_origin": "Nifty100",
            "priority_score": round(atr * avg_volume, 2)  # Added priority score
        })
        print(f"✅ SELECTED: ₹{ltp:,.2f} | Vol: {avg_volume:,.0f} | Range: ₹{atr:.2f} | RSI: {rsi_value:.2f}")

    except Exception as e:
        print(f"⚠️ Processing error: {str(e)[:70]}")
    finally:
        time.sleep(0.5)  # Jitter to avoid patterns

# ======== SMALL CAP SCAN ========
print("\n🚀 Starting Small Cap scan...")
smallcap_results = []

# Filter only EQ/NSE_EQ & skip SME/PSU/ETF/REIT
symbol_col = "SEM_TRADING_SYMBOL"
series_col = "SEM_SERIES"
segment_col = "SEM_SEGMENT"
security_id_col = "SEM_SMST_SECURITY_ID"

smallcap_df = master_df[
    (master_df[series_col] == "EQ") &
    (master_df[segment_col] == "NSE_EQ") &
    (~master_df[symbol_col].str.contains("SME|PSU|ETF|REIT", case=False, na=False))
]

for _, row in smallcap_df.iterrows():
    try:
        sym = row[symbol_col]
        sec_id = str(row[security_id_col])
        sector = "UNKNOWN"

        print(f"🔍 Scanning SMALLCAP: {sym}")

        # Fetch intraday 1-min data
        today = datetime.date.today().strftime("%Y-%m-%d")
        payload = {
            "securityId": sec_id,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "expiryCode": 0,
            "fromDate": today,
            "toDate": today
        }
        
        response = requests.post(
            "https://api.dhan.co/v2/charts/intraday", 
            headers=HEADERS, 
            json=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"⛔ Failed to fetch data for {sym}: {response.status_code}")
            continue
            
        data = response.json()
        df = pd.DataFrame(data["data"])
        if df.empty or len(df) < 20:
            print(f"⛔ {sym}: Insufficient data")
            continue

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata")
        df.set_index("datetime", inplace=True)
        df = df.astype(float).sort_index()

        # Apply all filters
        ltp = df["close"].iloc[-1]
        if ltp > CAPITAL:
            print(f"⛔ Price too high: ₹{ltp:.2f} > ₹{CAPITAL:,.2f}")
            continue

        rsi = RSIIndicator(df["close"], window=14).rsi().iloc[-1]
        if rsi > SMALLCAP_MAX_RSI:
            print(f"⛔ RSI too high: {rsi:.2f} > {SMALLCAP_MAX_RSI}")
            continue

        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]
        if atr < SMALLCAP_MIN_ATR:
            print(f"⛔ Low volatility: ₹{atr:.2f} < ₹{SMALLCAP_MIN_ATR:.2f}")
            continue

        avg_vol = df["volume"].tail(5).mean()
        if avg_vol < SMALLCAP_MIN_VOLUME:
            print(f"⛔ Low volume: {avg_vol:,.0f} < {SMALLCAP_MIN_VOLUME:,.0f}")
            continue

        if df["close"].iloc[-1] < df["close"].iloc[-5]:
            print(f"⛔ Negative momentum")
            continue

        sma_20 = SMAIndicator(df["close"], window=20).sma_indicator().iloc[-1]

        qty = int(CAPITAL // ltp)
        capital_used = round(qty * ltp, 2)
        potential_profit = round(atr * qty, 2)
        priority_score = round(avg_vol * atr, 2)

        smallcap_results.append({
            "symbol": sym,
            "security_id": sec_id,
            "ltp": round(ltp, 2),
            "quantity": qty,
            "capital_used": capital_used,
            "avg_volume": int(avg_vol),
            "avg_range": round(atr, 2),
            "potential_profit": potential_profit,
            "sma_20": round(sma_20, 2),
            "rsi": round(rsi, 2),
            "macd": 0,  # Not calculated for smallcaps
            "macd_hist": 0,
            "macd_crossover": 0,
            "sector": sector,
            "sector_strength": 0,  # Not available for smallcaps
            "stock_origin": "SmallCap",
            "priority_score": priority_score
        })
        print(f"✅ SELECTED: ₹{ltp:.2f} | Vol: {avg_vol:,.0f} | Range: ₹{atr:.2f} | RSI: {rsi:.2f}")

    except Exception as e:
        print(f"❌ {sym} failed: {e}")

# ======== COMBINE RESULTS ========
print("\n🚀 Combining scan results...")
all_results = nifty100_results + smallcap_results

if all_results:
    results_df = pd.DataFrame(all_results)
    # Priority score = ATR * Volume (higher is better)
    results_df["priority_score"] = results_df["avg_range"] * results_df["avg_volume"]
    results_df = results_df.sort_values("priority_score", ascending=False)
    
    # Save to CSV with additional technical data
    results_df.to_csv(OUTPUT_CSV, index=False)   
    
    # ======== TRENDING STOCKS BOOST ========
    print("\n🚀 Adding trending stocks to boost trade pool...")
    
    trending_additions = []
    
    for _, row in nifty100_df.iterrows():
        symbol = row["base_symbol"]
        secid = str(row["SEM_SMST_SECURITY_ID"])
        sector = row.get("sector", "")
        
        # Skip if already selected
        if symbol in results_df["symbol"].values:
            continue
    
        try:
            quote_url = f"https://api.dhan.co/quotes/isin?security_id={secid}"
            quote_resp = requests.get(quote_url, headers=HEADERS, timeout=5)
            if quote_resp.status_code != 200:
                continue
            data = quote_resp.json()
            open_price = float(data.get("openPrice", 0))
            ltp = float(data.get("lastTradedPrice", 0))
            if open_price <= 0 or ltp <= 0:
                continue
    
            # Calculate percentage move
            if market_condition == "bullish":
                pct_move = ((ltp - open_price) / open_price) * 100
                min_move = 2.0  # Minimum 2% gain for bullish
            else:
                pct_move = ((open_price - ltp) / open_price) * 100
                min_move = 1.5  # Minimum 1.5% drop for bearish
            
            if abs(pct_move) < min_move:
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
            rsi_val = calculate_rsi(closes)
    
            # Adjust RSI check for market condition
            if market_condition == "bullish":
                if rsi_val >= 70:  # Overbought
                    continue
            else:
                if rsi_val <= 30:  # Oversold
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
                "macd": 0,
                "macd_hist": 0,
                "macd_crossover": 0,
                "sector": sector,
                "sector_strength": sector_strengths.get(sector, 0),
                "priority_score": abs(pct_move) * 100,  # Sort by momentum strength
                "stock_origin": "Nifty100"
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
    print(f"📊 Breakdown:")
    print(f"- Nifty100 stocks: {len(nifty100_results)}")
    print(f"- SmallCap stocks: {len(smallcap_results)}")
    print(f"- Trending stocks: {len(trending_additions)}")
    
    # 📄 Save summary to filter_summary_log.csv
    summary_path = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"
    summary_row = {
        "date": datetime.now().strftime("%m/%d/%Y %H:%M"),
        "Script_Name": "consolidated_dynamic_stock_generator.py",
        "total_scanned": total_scanned + len(smallcap_df),
        "affordable": affordable + len(smallcap_results),
        "technical_passed": technical_passed + len(smallcap_results),
        "volume_passed": volume_passed + len(smallcap_results),
        "sentiment_passed": sentiment_passed,
        "sma_passed": sma_passed + len(smallcap_results),
        "rsi_passed": rsi_passed + len(smallcap_results),
        "final_selected": len(results_df),
        "market_condition": market_condition
    }
    
    try:
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
            summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])])
            summary_df.to_csv(summary_path, index=False)
        else:
            pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    except Exception as e:
        print(f"⚠️ Could not write to filter_summary_log.csv: {e}")
    
    print(f"📊 Top 5 opportunities:")
    print(results_df[["symbol", "ltp", "quantity", "potential_profit", "priority_score", "stock_origin"]].head().to_string(index=False))
else:
    print("\n❌ No stocks passed all filters")

# ======== PERFORMANCE METRICS ========
elapsed = datetime.now() - start_time
total_sec = elapsed.total_seconds()
minutes = int(total_sec // 60)
seconds = int(total_sec % 60)
print(f"\n⏱️ Total scan time: {minutes} min {seconds} sec")
print(f"💵 Capital available: ₹{CAPITAL:,.2f}")
print(f"📈 Potential positions: {len(all_results)}")
# 📝 Save log
with open("D:/Downloads/Dhanbot/dhan_autotrader/Logs/consolidated_dynamic_stock_generator.txt", "w", encoding="utf-8") as f:
    f.write(log_buffer.getvalue())