# ========== PART 1: AUTOTRADE ENTRY SCRIPT ==========
import os
import sys
import json
import pandas as pd
import time
from datetime import datetime, timedelta, time as dtime
import pytz
from dhanhq import DhanContext, dhanhq
from decimal import Decimal, ROUND_HALF_UP
import requests
import numpy as np
from db_logger import insert_portfolio_log_to_db, log_to_postgres
import math
import io
import traceback
from scipy.stats import linregress  # Added for trend analysis

log_buffer = io.StringIO()
test_mode = False

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

# ========== Config ==========
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
MASTER_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
PORTFOLIO_LOG = "D:/Downloads/Dhanbot/dhan_autotrader/portfolio_log.csv"

# ========== Load Config ==========
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# ========== New Configuration Parameters ==========
SKIP_CHART_IF_CANDLE_FOUND = config.get("skip_chart_if_candle_found", True)
CANDLE_STRONG_THRESHOLD = config.get("candle_strong_threshold", 1.4)

# ========== Create DhanHQ global context and client ==========
context = DhanContext(config["client_id"], config["access_token"])
dhan = dhanhq(context)

# ========== Telegram from config ==========
TG_TOKEN = config["telegram_token"]
TG_CHAT_ID = config["telegram_chat_id"]


# Security IDs for NSE indices (manually curated)
INDEX_SECURITY_MAP = {
    "NIFTY 50": 26000,
    "NIFTY BANK": 26009,
    "NIFTY AUTO": 26010,
    "NIFTY IT": 26011,
    "NIFTY FMCG": 26012,
    "NIFTY FIN SERVICE": 26013,
    "NIFTY PHARMA": 26014,
    "NIFTY REALTY": 26015,
    "NIFTY ENERGY": 26018,
    "NIFTY METAL": 26019,
    "NIFTY OIL & GAS": 26020,
    "NIFTY CONSUMER DURABLES": 26022,
    "NIFTY HEALTHCARE": 26023,
    "NIFTY INFRA": 26024,
    "NIFTY MEDIA": 26025,
    "NIFTY PSU BANK": 26026,
    "NIFTY PRIVATE BANK": 26028,
    "NIFTY SERVICES SECTOR": 26037,
    "NIFTY COMMODITIES": 26045
}

# Multi-proxy stocks for each sector (3 stocks per sector)
MULTI_PROXY_MAP = {
    "NIFTY 50": [
        {"name": "RELIANCE INDUSTRIES LTD", "security_id": 2885},
        {"name": "HDFC BANK LTD", "security_id": 1333},
        {"name": "INFOSYS LIMITED", "security_id": 1594}
    ],
    "NIFTY BANK": [
        {"name": "HDFC BANK LTD", "security_id": 1333},
        {"name": "ICICI BANK LTD", "security_id": 4963},
        {"name": "STATE BANK OF INDIA", "security_id": 3045}
    ],
    "NIFTY AUTO": [
        {"name": "MARUTI SUZUKI INDIA LTD.", "security_id": 10999},
        {"name": "TATA MOTORS LTD", "security_id": 3456},
        {"name": "MAHINDRA & MAHINDRA LTD", "security_id": 2031}
    ],
    "NIFTY IT": [
        {"name": "INFOSYS LIMITED", "security_id": 1594},
        {"name": "TATA CONSULTANCY SERVICES LTD", "security_id": 11536},
        {"name": "WIPRO LIMITED", "security_id": 3787}
    ],
    "NIFTY FMCG": [
        {"name": "HINDUSTAN UNILEVER LTD.", "security_id": 1394},
        {"name": "ITC LIMITED", "security_id": 1660},
        {"name": "NESTLE INDIA LIMITED", "security_id": 17963}
    ],
    "NIFTY FIN SERVICE": [
        {"name": "BAJAJ FINANCE LIMITED", "security_id": 317},
        {"name": "HDFC BANK LTD", "security_id": 1333},
        {"name": "HDFC LIFE INSURANCE COMPANY LTD", "security_id": 467}
    ],
    "NIFTY PHARMA": [
        {"name": "SUN PHARMACEUTICAL IND L", "security_id": 3351},
        {"name": "DR. REDDYS LABORATORIES LTD", "security_id": 881},
        {"name": "CIPLA LTD", "security_id": 694}
    ],
    "NIFTY REALTY": [
        {"name": "DLF LIMITED", "security_id": 14732},
        {"name": "GODREJ PROPERTIES LTD", "security_id": 17875},
        {"name": "OBEROI REALTY LTD", "security_id": 20242}
    ],
    "NIFTY ENERGY": [
        {"name": "RELIANCE INDUSTRIES LTD", "security_id": 2885},
        {"name": "NTPC LIMITED", "security_id": 11630},
        {"name": "POWER GRID CORPORATION OF INDIA LTD", "security_id": 14977}
    ],
    "NIFTY METAL": [
        {"name": "TATA STEEL LIMITED", "security_id": 3499},
        {"name": "HINDALCO INDUSTRIES LTD", "security_id": 1363},
        {"name": "VEDANTA LIMITED", "security_id": 3063}
    ],
    "NIFTY OIL & GAS": [
        {"name": "RELIANCE INDUSTRIES LTD", "security_id": 2885},
        {"name": "OIL AND NATURAL GAS CORP.", "security_id": 2475}, 
        {"name": "GAIL (INDIA) LIMITED", "security_id": 4717}
    ],
    "NIFTY CONSUMER DURABLES": [
        {"name": "TITAN COMPANY LIMITED", "security_id": 3506},
        {"name": "VOLTAS LIMITED", "security_id": 3718},
        {"name": "BLUE STAR LIMITED", "security_id": 8311}
    ],
    "NIFTY HEALTHCARE": [
        {"name": "DIVI S LABORATORIES LTD", "security_id": 10940},
        {"name": "APOLLO HOSPITALS ENTERPRISE LTD", "security_id": 157}, 
        {"name": "FORTIS HEALTHCARE LIMITED", "security_id": 14592}
    ],
    "NIFTY INFRA": [
        {"name": "LARSEN & TOUBRO LTD.", "security_id": 11483},
        {"name": "ADANI PORTS AND SPECIAL ECONOMIC ZONE LTD", "security_id": 15083},
        {"name": "ULTRATECH CEMENT LIMITED", "security_id": 11532}
    ],
    "NIFTY MEDIA": [
        {"name": "SUN TV NETWORK LIMITED", "security_id": 13404},
        {"name": "ZEE ENTERTAINMENT ENTERPRISES LTD", "security_id": 3812},
        {"name": "PVR INOX LTD", "security_id": 13147}
    ],
    "NIFTY PSU BANK": [
        {"name": "STATE BANK OF INDIA", "security_id": 3045},
        {"name": "PUNJAB NATIONAL BANK", "security_id": 10666},
        {"name": "BANK OF BARODA", "security_id": 4668}
    ],
    "NIFTY PRIVATE BANK": [
        {"name": "ICICI BANK LTD.", "security_id": 4963},
        {"name": "AXIS BANK LTD", "security_id": 5900},
        {"name": "KOTAK MAHINDRA BANK LTD", "security_id": 1922}
    ],
    "NIFTY SERVICES SECTOR": [
        {"name": "CONTAINER CORPORATION OF INDIA LTD", "security_id": 4749},
        {"name": "IRCTC LTD", "security_id": 13611},
        {"name": "BHARTI AIRTEL LIMITED", "security_id": 10604}
    ],
    "NIFTY COMMODITIES": [
        {"name": "GRASIM INDUSTRIES LTD", "security_id": 1232},
        {"name": "JSW STEEL LIMITED", "security_id": 11723},
        {"name": "ADANI ENTERPRISES LTD", "security_id": 25}
    ]
}

def fetch_intraday_candles(security_id, symbol_name, is_index=False):
    """Fetch historical candles for stocks or indices"""
    if not security_id:
        print(f"⚠️ No security_id provided for {symbol_name}")
        return []
    
    exchange_segment = "NSE_INDEX" if is_index else "NSE_EQ"
    instrument_type = "INDEX" if is_index else "EQUITY"
    from_dt = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    to_dt = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📊 Fetching {'index' if is_index else 'stock'} data: "
          f"{symbol_name} [ID: {security_id}] ({from_dt} to {to_dt})")
    
    try:
        candles = dhan.historical_daily_data(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=from_dt,
            to_date=to_dt
        )
    except Exception as e:
        print(f"🚨 API Error for {symbol_name}: {str(e)}")
        return []
    
    if not candles or not isinstance(candles, dict) or 'data' not in candles:
        print(f"⚠️ No data returned for {symbol_name}")
        return []
    
    data = candles['data']
    if not all(key in data for key in ('open', 'high', 'low', 'close', 'volume', 'timestamp')):
        print(f"⚠️ Incomplete data structure for {symbol_name}")
        return []
    
    num_candles = len(data['close'])
    merged_candles = [
        {
            "timestamp": data["timestamp"][i],
            "open": data["open"][i],
            "high": data["high"][i],
            "low": data["low"][i],
            "close": data["close"][i],
            "volume": data["volume"][i],
        }
        for i in range(num_candles)
    ]
    
    print(f"✅ Retrieved {len(merged_candles)} candles for {symbol_name}")
    return merged_candles

def is_bullish(candles, days=5):
    """Determine bullish trend based on closing prices"""
    if not candles or len(candles) < days:
        return False
    
    # Use the last 'days' trading days
    recent_closes = [candle["close"] for candle in candles[-days:]]
    
    # Simple trend: current close > previous close (momentum)
    if recent_closes[-1] > recent_closes[-2]:
        return True
    
    # Medium-term: current close > average of last 'days' closes
    avg_close = sum(recent_closes) / len(recent_closes)
    return recent_closes[-1] > avg_close

def analyze_index(sector, security_id):
    """Analyze sector using actual index data"""
    candles = fetch_intraday_candles(security_id, sector, is_index=True)
    if not candles:
        print(f"⚠️ Could not fetch index data for {sector}")
        return None
    
    return is_bullish(candles)

def analyze_proxies(sector, proxies):
    """Analyze sector using multiple proxy stocks"""
    if not proxies:
        print(f"⚠️ No proxies defined for {sector}")
        return None
    
    bullish_count = 0
    valid_proxies = 0
    
    for proxy in proxies:
        candles = fetch_intraday_candles(proxy['security_id'], proxy['name'])
        time.sleep(1.2)  # Rate limit protection
        
        if not candles or len(candles) < 5:
            continue
            
        valid_proxies += 1
        if is_bullish(candles):
            bullish_count += 1
    
    if valid_proxies == 0:
        print(f"⚠️ No valid proxies for {sector}")
        return None
    
    return bullish_count / valid_proxies >= 0.6  # 60% threshold

# Main analysis logic
def get_sector_sentiment_map(print_table=False):
    """
    Returns: dict { sector_name (str) : True (bullish) / False (bearish/unknown) }
    Uses file cache for current day to avoid repeated index/proxy fetches.
    Cache file: sector_sentiment_cache.json (refreshes daily after first run)
    """
    CACHE_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/sector_sentiment_cache.json"
    today = datetime.now().strftime("%Y-%m-%d")
    cache = None

    # Try to load from cache
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                cache = json.load(f)
            if cache.get("date") == today and "sentiment_map" in cache:
                print(f"🟢 Loaded sector sentiment from cache ({CACHE_PATH})")
                if print_table and "summary_table" in cache:
                    print("\n" + "="*50)
                    print("SECTOR TREND SUMMARY TABLE (CACHED)")
                    print("="*50)
                    print(cache["summary_table"])
                    print("="*50)
                    print("ANALYSIS COMPLETE (CACHED)")
                    print("="*50)
                return cache["sentiment_map"]
        except Exception as e:
            print(f"⚠️ Failed to load sector sentiment cache: {e} (regenerating)")

    # No valid cache, run full analysis
    sector_results = []
    sentiment_map = {}

    for sector in INDEX_SECURITY_MAP.keys():
        print(f"\n🔍 Analyzing {sector}")

        # Try index first
        index_id = INDEX_SECURITY_MAP[sector]
        result = analyze_index(sector, index_id)
        used_proxy = False

        # Fallback to proxies if index analysis fails
        if result is None:
            print(f"🔄 Falling back to proxy analysis for {sector}")
            proxies = MULTI_PROXY_MAP.get(sector, [])
            result = analyze_proxies(sector, proxies)
            used_proxy = True

        # Final determination
        if result is None:
            trend = "UNKNOWN"
            status = "⛔"
            sentiment_map[sector] = False  # Treat UNKNOWN as not bullish
        elif result:
            trend = "BULLISH"
            status = "✅"
            sentiment_map[sector] = True
        else:
            trend = "BEARISH"
            status = "❌"
            sentiment_map[sector] = False

        sector_results.append({
            "SECTOR": sector,
            "TREND": trend,
            "STATUS": status,
            "METHOD": "PROXY" if used_proxy else "INDEX"
        })

        print(f"📌 {sector} Trend: {trend} {status} (by {'PROXY' if used_proxy else 'INDEX'})")

    # Print summary table if requested
    summary_table = ""
    if print_table:
        print("\n" + "="*50)
        print("SECTOR TREND SUMMARY TABLE")
        print("="*50)
        results_df = pd.DataFrame(sector_results)
        summary_table = results_df.to_string(index=False)
        print(summary_table)
        print("="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)

    # Save to cache for rest of the day
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump({
                "date": today,
                "sentiment_map": sentiment_map,
                "summary_table": summary_table
            }, f)
        print(f"🟢 Sector sentiment saved to cache: {CACHE_PATH}")
    except Exception as e:
        print(f"⚠️ Could not write sector sentiment cache: {e}")

    return sentiment_map


# ✅ Sector-to-NIFTY Index Mapping + Dhan Security IDs
sector_index_ids = {
    "NIFTY AUTO": "13604",
    "NIFTY BANK": "13605",
    "NIFTY ENERGY": "13607",
    "NIFTY FIN SERVICE": "13606",
    "NIFTY IT": "13609",
    "NIFTY FMCG": "13608",
    "NIFTY METAL": "13610",
    "NIFTY INFRA": "13613",
    "NIFTY PHARMA": "13611",
    "NIFTY COMMODITIES": "13612",
    "NIFTY SERVICES SECTOR": "13614"
}

# ✅ Map raw sector names to above NIFTY sectors
sector_indices = {
    "BANK": "NIFTY BANK",
    "NIFTY BANK": "NIFTY BANK",
    "BANKING": "NIFTY BANK",
    "FINANCIAL SERVICES": "NIFTY FIN SERVICE",
    "NIFTY FIN SERVICE": "NIFTY FIN SERVICE",
    "IT": "NIFTY IT",
    "NIFTY IT": "NIFTY IT",
    "IT SERVICES": "NIFTY IT",
    "AUTO": "NIFTY AUTO",
    "NIFTY AUTO": "NIFTY AUTO",
    "AUTO ANCILLARIES": "NIFTY AUTO",
    "PHARMA": "NIFTY PHARMA",
    "PHARMACEUTICALS": "NIFTY PHARMA",
    "FMCG": "NIFTY FMCG",
    "NIFTY FMCG": "NIFTY FMCG",
    "METAL": "NIFTY METAL",
    "NIFTY METAL": "NIFTY METAL",
    "ENERGY": "NIFTY ENERGY",
    "NIFTY ENERGY": "NIFTY ENERGY",
    "POWER": "NIFTY ENERGY",
    "INFRASTRUCTURE": "NIFTY INFRA",
    "NIFTY INFRA": "NIFTY INFRA",
    "INSURANCE": "NIFTY FIN SERVICE",
    "AGROCHEMICALS": "NIFTY COMMODITIES",
    "LOGISTICS": "NIFTY SERVICES SECTOR",
    "MEDIA": "NIFTY MEDIA",
    "NIFTY MEDIA": "NIFTY MEDIA"
}


# ========== Pattern Confidence Weights ==========
PATTERN_WEIGHTS = {
    # Candlestick patterns
    "Bullish Hammer": {"weight": 1.1, "vol_scale": 1.2},
    "Bullish Engulfing": {"weight": 1.3, "vol_scale": 1.5},
    "Piercing Line": {"weight": 1.2, "vol_scale": 1.3},
    "Morning Star": {"weight": 1.5, "vol_scale": 3.0},
    "Inverted Hammer": {"weight": 1.1, "vol_scale": 1.2},
    "Bullish Harami": {"weight": 1.1, "vol_scale": 1.1},
    "Three White Soldiers": {"weight": 1.3, "vol_scale": 1.5},
    "Bullish Kicker": {"weight": 1.6, "vol_scale": 2.5},
    "Breakout Marubozu": {"weight": 1.4, "vol_scale": 1.8},
    "Volume Breakout Candle": {"weight": 1.4, "vol_scale": 2.0},
    "Gap-Down Reversal": {"weight": 1.5, "vol_scale": 2.0},
    # Chart patterns
    "Cup and Handle": {"weight": 1.7, "vol_scale": 2.0},
    "Double Bottom": {"weight": 1.5, "vol_scale": 1.8},
    "Triple Bottom": {"weight": 1.6, "vol_scale": 1.9},
    "Ascending Triangle": {"weight": 1.5, "vol_scale": 1.7},
    "Bullish Pennant": {"weight": 1.4, "vol_scale": 2.0},
    "Bullish Wedge (Falling Wedge)": {"weight": 1.4, "vol_scale": 1.8},
    "Rounding Bottom": {"weight": 1.3, "vol_scale": 1.6},
    "Inverse Head and Shoulders": {"weight": 1.8, "vol_scale": 2.2},
    "Rounded Consolidation (Roundboom)": {"weight": 1.3, "vol_scale": 1.5},
    "Bullish Rectangle": {"weight": 1.4, "vol_scale": 1.7},
    "Bullish Flag": {"weight": 1.5, "vol_scale": 1.8},  # New pattern
    "Symmetrical Triangle": {"weight": 1.4, "vol_scale": 1.6},  # New pattern
    # Bearish patterns
    "Head and Shoulders": {"weight": 1.8, "vol_scale": 2.0},
    "Double Top": {"weight": 1.6, "vol_scale": 1.8},
    "Triple Top": {"weight": 1.7, "vol_scale": 1.9},
    "Descending Triangle": {"weight": 1.5, "vol_scale": 1.7},
    "Bearish Rectangle": {"weight": 1.4, "vol_scale": 1.7},
    "Bearish Wedge (Rising Wedge)": {"weight": 1.4, "vol_scale": 1.8},
    "Rounded Top": {"weight": 1.3, "vol_scale": 1.6},
    "Volume Breakout Pattern": {"weight": 1.4, "vol_scale": 2.0},
    "Distribution Zone": {"weight": 1.3, "vol_scale": 1.5}
}

# Define reversal and breakout patterns
REVERSAL_PATTERNS = {
    "Bullish Hammer", "Bullish Engulfing", "Piercing Line", "Morning Star",
    "Inverted Hammer", "Bullish Harami", "Three White Soldiers", "Bullish Kicker",
    "Gap-Down Reversal", "Double Bottom", "Triple Bottom", "Inverse Head and Shoulders",
    "Rounding Bottom"
}

BREAKOUT_PATTERNS = {
    "Breakout Marubozu", "Volume Breakout Candle", "Volume Breakout Pattern",
    "Cup and Handle", "Ascending Triangle", "Bullish Pennant",
    "Bullish Wedge (Falling Wedge)", "Bullish Rectangle", "Bullish Flag",
    "Symmetrical Triangle"
}

def log_pattern_detection(symbol, pattern_name, detected, reason=""):
    """Log pattern detection result for debugging"""
    prefix = f"{symbol}: " if symbol else ""
    if detected:
        print(f"✅ {prefix}Pattern {pattern_name} detected")
    else:
        if reason:
            print(f"❌ {prefix}Pattern {pattern_name} not detected - {reason}")
        else:
            print(f"❌ {prefix}Pattern {pattern_name} not detected")

# ========== ATR/ADR Calculation ==========
AVERAGE_TRUE_RANGE_PERIOD = 14
ADR_PERIOD = 10  # Days for Average Daily Range

def calculate_atr(candles):
    """Calculate Average True Range (ATR) for volatility measurement"""
    if len(candles) < AVERAGE_TRUE_RANGE_PERIOD + 1:
        return 0.0
        
    df = pd.DataFrame(candles)
    high, low, close = df['high'], df['low'], df['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(AVERAGE_TRUE_RANGE_PERIOD).mean().iloc[-1]
    return atr

def calculate_adr(security_id):
    """Calculate Average Daily Range (ADR) for realistic TP/SL capping"""
    try:
        from_dt = datetime.now() - timedelta(days=ADR_PERIOD + 5)
        to_dt = datetime.now()
        
        daily_data = dhan.historical_daily_data(
            security_id=str(security_id), 
            exchange_segment="NSE_EQ", 
            instrument_type="EQUITY", 
            from_date=from_dt.strftime("%Y-%m-%d"), 
            to_date=to_dt.strftime("%Y-%m-%d")
        )
        
        if not isinstance(daily_data, list) or len(daily_data) < ADR_PERIOD:
            raise ValueError(
                f"❌ Invalid ADR data for {security_id}: "
                f"Expected list of {ADR_PERIOD}+ candles, got {type(daily_data)}"
            )
            
        # Extract highs and lows from candle dictionaries
        try:
            highs = [float(candle['high']) for candle in daily_data]
            lows = [float(candle['low']) for candle in daily_data]
        except (KeyError, TypeError, ValueError) as e:
            print(f"❌ ADR data parsing failed: {e}")
            return 0.0
        daily_ranges = [abs(high - low) for high, low in zip(highs, lows)][-ADR_PERIOD:]
        return sum(daily_ranges) / len(daily_ranges) if daily_ranges else 0.0
    except Exception as e:
        print(f"❌ ADR calculation failed: {e}")
        return 0.0
        
# ========== RSI Divergence Calculation ==========
def has_rsi_divergence(highs, rsi_series, lookback=14):
    """
    Detect bearish RSI divergence (price makes new high but RSI doesn't)
    with robust validation and logging
    """
    # Validate input lengths
    if len(highs) < lookback or len(rsi_series) < lookback:
        print(f"❌ RSI divergence check failed: Need {lookback} periods, got {len(highs)} highs and {len(rsi_series)} RSI values")
        return False
    
    try:
        # Find highest high in lookback period
        lookback_highs = highs.iloc[-lookback:]
        if lookback_highs.empty:
            print(f"❌ Empty lookback window for {symbol} RSI divergence")
            return False
            
        max_high_idx = lookback_highs.idxmax()
        max_high = highs.loc[max_high_idx]
        
        # Find corresponding RSI value
        rsi_at_high = rsi_series.loc[max_high_idx]
        
        # Check current RSI vs RSI at high
        current_high = highs.iloc[-1]
        current_rsi = rsi_series.iloc[-1]
        
        divergence_detected = current_high > max_high and current_rsi < rsi_at_high
        
        if divergence_detected:
            print(f"⚠️ Bearish RSI divergence detected: "
                  f"Price {current_high:.2f} > {max_high:.2f} "
                  f"but RSI {current_rsi:.2f} < {rsi_at_high:.2f}")
        
        return divergence_detected
        
    except (IndexError, KeyError) as e:
        print(f"❌ RSI divergence calculation error: {str(e)}")
        traceback.print_exc()
        return False

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": msg}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")

def get_capital():
    try:
        return float(config.get("capital", 0.0))
    except:
        return 0.0

def has_hold():
    if not os.path.exists(PORTFOLIO_LOG):
        return False
    try:
        df = pd.read_csv(PORTFOLIO_LOG)
        today = datetime.now().strftime("%m/%d/%Y")
        return any((df['status'].str.upper() == "HOLD") & df['timestamp'].str.contains(today))
    except:
        return False

def fetch_candles(security_id, count=20, cache={}, exchange_segment="NSE_EQ", instrument_type="EQUITY"):
    """Fetch candles with caching and retry mechanism to prevent redundant calls"""
    date_str = datetime.now().strftime("%Y%m%d")
    cache_key = f"{date_str}_{security_id}_{count}_{exchange_segment}"
    if cache_key in cache:
        return cache[cache_key]

    for attempt in range(3):  # Retry up to 3 times
        try:    
            now = datetime.now()
            from_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
            to_dt = now.replace(second=0, microsecond=0)
            print(f"📡 Fetching candles for {security_id} — segment: {exchange_segment}, type: {instrument_type}")
            response = dhan.intraday_minute_data(
                security_id=str(security_id),
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=from_dt.strftime("%Y-%m-%d"),
                to_date=to_dt.strftime("%Y-%m-%d")
            )

            # Check if response is valid and extract data
            if not response or not isinstance(response, dict) or 'data' not in response:
                print(f"⚠️ Invalid response structure for {security_id}")
                return []
                
            raw_data = response['data']
            # 🔒 Defensive check for unsupported string response
            if isinstance(raw_data, str):
                if raw_data.strip() == "":
                    print(f"❌ Empty response string for {security_id} — possible Dhan API failure or invalid session")
                else:
                    print(f"⚠️ Unsupported data format for {security_id}: {type(raw_data)} - content:\n{raw_data}\n")
                return []
            
            
            candles = []  # Initialize candles to empty list

            # Handle both list and dict formats
            if isinstance(raw_data, list):
                # Process as list of candles
                for candle in raw_data:
                    try:
                        candles.append({
                            "open": candle["open"],
                            "high": candle["high"],
                            "low": candle["low"],
                            "close": candle["close"],
                            "volume": candle["volume"],
                            "timestamp": datetime.fromtimestamp(candle["timestamp"])
                        })
                    except KeyError:
                        continue
                if not candles:
                    print(f"⚠️ Empty list-style candle data for {security_id}")
                    return []
            elif isinstance(raw_data, dict):
                required_keys = ["open", "high", "low", "close", "volume", "timestamp"]
                if not all(k in raw_data and raw_data[k] for k in required_keys):
                    print(f"⚠️ Malformed dict-style candle data for {security_id}")
                    return []
                try:
                    n = len(raw_data['open'])
                    for i in range(n):
                        candles.append({
                            "open": raw_data["open"][i],
                            "high": raw_data["high"][i],
                            "low": raw_data["low"][i],
                            "close": raw_data["close"][i],
                            "volume": raw_data["volume"][i],
                            "timestamp": datetime.fromtimestamp(raw_data["timestamp"][i])
                        })
                except Exception as e:
                    print(f"❌ Error parsing dict candles for {security_id}: {e}")
                    return []
            else:
                print(f"⚠️ Unsupported data format for {security_id}: {type(raw_data)}")
                return []

            if not candles:
                print(f"⚠️ No valid parsed candles for {security_id}")
                return []
            # ✅ Candle Timestamp Freshness Check
            last_candle_time = candles[-1]["timestamp"]
            # Validate timestamp type
            if not isinstance(last_candle_time, datetime):
                raise TypeError(
                    f"❌ Invalid candle timestamp type: "
                    f"Expected datetime, got {type(last_candle_time)}"
                )
            
            # Convert last_candle_time to UTC (timezone-aware)
            if last_candle_time.tzinfo is None or last_candle_time.tzinfo.utcoffset(last_candle_time) is None:
                last_candle_time = last_candle_time.replace(tzinfo=pytz.UTC)
            
            now = datetime.now(pytz.utc)
            if (now - last_candle_time) > timedelta(minutes=2) or now.date() > last_candle_time.date():
                print(f"❌ Stale candle data for {security_id} ({last_candle_time}), forcing refresh")
                del cache[cache_key]
                return fetch_candles(security_id, count, cache, exchange_segment, instrument_type)

            cache[cache_key] = candles
            return candles     

        except Exception as e:
            if "Rate_Limit" in str(e) and attempt < 2:
                wait_time = (attempt + 1) * 10
                print(f"⚠️ Rate limit hit for {security_id}, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            print(f"❌ Error fetching candles for {security_id}: {e}")
            return []
    return []

def detect_bullish_pattern(candles, symbol=None):
    """Enhanced pattern detection with multi-pattern scoring and priority system"""
    skip_chart = False  # ✅ Initialize at the beginning to avoid scope issues
    if not candles:
        raise ValueError("❌ Empty candles passed to pattern detection")
    if len(candles) < 5:
        raise ValueError(
            f"❌ Insufficient candles ({len(candles)}) for "
            f"pattern detection on {symbol}"
        )

    # Get 15-minute trend for chart patterns
    chart_pattern_trend = None
    if len(candles) >= 75:  # Increased to 75 for 5*15min candles
        chart_pattern_trend = get_15min_trend(candles, min_candles=5)

    
    df = pd.DataFrame(candles)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    detected_patterns = []  # Store all detected patterns with scores
    
    # Volume confirmation helper
    def volume_confirmed(index=-1, multiplier=1.2, lookback=5, pattern_type=None):
        if len(df) < lookback + 1:
            return True, 1.0  # Not enough data
        
        # Pattern-specific lookback periods
        if pattern_type == "Morning Star":
            lookback = 8  # Longer period for morning star
        elif pattern_type == "Breakout":
            lookback = 3  # Shorter period for breakouts
        
        vol_avg = v.iloc[-lookback-1:-1].mean()
        
        # Fail loudly on invalid volume
        if vol_avg <= 0:
            raise ValueError(
                f"❌ Invalid volume average ({vol_avg}) for {symbol} "
                f"in {pattern_type} pattern"
            )
        
        vol_ratio = v.iloc[index] / vol_avg
        
        # For reversal patterns, allow slightly lower volume
        if "Reversal" in str(pattern_type) and vol_ratio >= multiplier * 0.8:
            return True, vol_ratio
            
        return vol_ratio >= multiplier, vol_ratio
    
    # Body/Range scoring with wick rejection
    def candle_score(index, pattern_type=None):
        body = abs(c.iloc[index] - o.iloc[index])
        candle_range = h.iloc[index] - l.iloc[index]
        if candle_range == 0:
            return 0.0  # Invalid candle
        
        body_ratio = body / candle_range if candle_range > 0 else 0
        upper_wick = h.iloc[index] - max(c.iloc[index], o.iloc[index])
        lower_wick = min(c.iloc[index], o.iloc[index]) - l.iloc[index]
        upper_ratio = upper_wick / candle_range
        lower_ratio = lower_wick / candle_range
        
        # Pattern-specific wick rejection
        if pattern_type in ["Breakout Marubozu", "Volume Breakout Candle"]:
            if upper_ratio > 0.15 or lower_ratio > 0.1:  # Max 15% upper wick, 10% lower wick
                return 0.0
        
        # Reversal patterns allow longer lower wicks
        elif pattern_type in REVERSAL_PATTERNS:
            if upper_ratio > 0.3:  # Max 30% upper wick
                return 0.0
        
        return min(1.0, body_ratio * 0.7 + (1 - (upper_ratio + lower_ratio)) * 0.3)

    # =====================
    # CANDLESTICK PATTERN DETECTION (FIRST PRIORITY)
    # =====================
    
    # Initialize Morning Star detection flag
    morning_star_detected = False
    
    # 1. Morning Star with index validation and downtrend check
    if len(c) >= 4:  # Need 4 candles to check trend
        # Validate candle indexes exist
        if len(c) < 4 or pd.isna([o.iloc[-4], c.iloc[-4], o.iloc[-3], c.iloc[-3]]).any():
            log_pattern_detection(symbol, "Morning Star", False, "Insufficient data")
        else:
            body1 = o.iloc[-3] - c.iloc[-3]
            body2 = abs(c.iloc[-2] - o.iloc[-2])
            # Strong downtrend: at least 2 of last 3 candles down
            downtrend = False
            if len(c) >= 6:
                down_count = sum([c.iloc[-6] > c.iloc[-5], c.iloc[-5] > c.iloc[-4], c.iloc[-4] > c.iloc[-3]])
                downtrend = down_count >= 2        
            if (downtrend and
                c.iloc[-3] < o.iloc[-3] and
                o.iloc[-2] < c.iloc[-3] and
                body2 < 0.3 * body1 and
                c.iloc[-1] > o.iloc[-1] and
                o.iloc[-1] > c.iloc[-2] and
                c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2):
    
                # Volume confirmation (only if all prior conditions are met)
                vol_ok, vol_ratio = volume_confirmed(multiplier=1.2, pattern_type="Morning Star")
    
                if vol_ok:
                    pattern_score = min(1.0, 0.8 + (c.iloc[-1] - o.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]))
                    detected_patterns.append(("Morning Star", pattern_score))
                    print(f"🌟 Volume Confirmed Morning Star detected (Vol Ratio: {vol_ratio:.2f}x)")
                    log_pattern_detection(symbol, "Morning Star", True)
                    morning_star_detected = True  # Set detection flag
                else:
                    log_pattern_detection(symbol, "Morning Star", False, f"Volume insufficient ({vol_ratio:.2f}x < 1.2x)")
            else:
                log_pattern_detection(symbol, "Morning Star", False, "Pattern conditions not met")
    
    # Skip other patterns if Morning Star detected
    if not morning_star_detected:
        # Conflict check: Skip if bearish candle before pattern
        if len(c) >= 2 and c.iloc[-2] < o.iloc[-2] and (o.iloc[-2] - c.iloc[-2]) > (h.iloc[-2] - l.iloc[-2]) * 0.6:
            print(f"⚠️ Bearish candle before pattern - skipping {symbol}")
            return False, None, 0.0
            
        # 2. Bullish Engulfing with reversal check
        if len(c) >= 3:  # Need 3 candles for reversal check
            engulfing_condition = (
                c.iloc[-2] < o.iloc[-2] and 
                c.iloc[-1] > o.iloc[-1] and
                c.iloc[-1] > o.iloc[-2] and 
                o.iloc[-1] < c.iloc[-2]
            )
            
            # Check for immediate bearish reversal in next candle
            next_candle_bearish = (
                len(c) >= 3 and 
                o.iloc[-3] > c.iloc[-3] and  # Previous candle was bearish
                c.iloc[-1] < o.iloc[-1]      # Current candle is bearish
            )
            
            if engulfing_condition and not next_candle_bearish:
                vol_ok, vol_ratio = volume_confirmed()
                if vol_ok:
                    pattern_score = candle_score(-1, "Bullish Engulfing") * 0.85
                    detected_patterns.append(("Bullish Engulfing", pattern_score))
                    log_pattern_detection(symbol, "Bullish Engulfing", True)
            elif engulfing_condition:
                log_pattern_detection(symbol, "Bullish Engulfing", False, "Immediate bearish reversal")
    
        # 3. Breakout Marubozu
        if len(c) >= 2:  # Now requires at least 2 candles
            body = abs(c.iloc[-1] - o.iloc[-1])
            candle_range = h.iloc[-1] - l.iloc[-1]
            body_ratio = body / candle_range if candle_range > 0 else 0
            
            # Look for no wicks and closing at high
            if (body_ratio > 0.9 and 
                min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1] < candle_range * 0.05 and
                h.iloc[-1] - max(c.iloc[-1], o.iloc[-1]) < candle_range * 0.05 and
                c.iloc[-1] > o.iloc[-1] and
                check_breakout(candles, period=1)):  # Now passing full candles array
                vol_ok, vol_ratio = volume_confirmed(multiplier=1.8)
                if vol_ok:
                    pattern_score = min(1.0, 0.85 + body_ratio * 0.5)
                    detected_patterns.append(("Breakout Marubozu", pattern_score))
                    log_pattern_detection(symbol, "Breakout Marubozu", True)
    
        # 4. Bullish Harami
        if len(c) >= 2:
            if (c.iloc[-2] < o.iloc[-2] and 
                c.iloc[-1] > o.iloc[-1] and
                o.iloc[-1] > c.iloc[-2] and 
                c.iloc[-1] < o.iloc[-2]):
                vol_ok, vol_ratio = volume_confirmed()
                if vol_ok:
                    pattern_score = candle_score(-1) * 0.75
                    detected_patterns.append(("Bullish Harami", pattern_score))
                    log_pattern_detection(symbol, "Bullish Harami", True)
    
        # 5. Three White Soldiers
        if len(c) >= 3:
            if (c.iloc[-3] > o.iloc[-3] and 
                c.iloc[-2] > o.iloc[-2] and 
                c.iloc[-1] > o.iloc[-1] and
                c.iloc[-2] > c.iloc[-3] and 
                c.iloc[-1] > c.iloc[-2]):
                vol_ok, vol_ratio = volume_confirmed()
                if vol_ok:
                    pattern_score = min(1.0, 0.9 * (vol_ratio / 2.0))
                    detected_patterns.append(("Three White Soldiers", pattern_score))
                    log_pattern_detection(symbol, "Three White Soldiers", True)
    
        # 6. Piercing Line
        if len(c) >= 2:
            if (c.iloc[-2] < o.iloc[-2] and 
                c.iloc[-1] > o.iloc[-1] and
                o.iloc[-1] < c.iloc[-2] and 
                c.iloc[-1] > (o.iloc[-2] + c.iloc[-2]) / 2):
                vol_ok, vol_ratio = volume_confirmed()
                if vol_ok:
                    pattern_score = min(1.0, 0.8 * (c.iloc[-1] - o.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]))
                    detected_patterns.append(("Piercing Line", pattern_score))
                    log_pattern_detection(symbol, "Piercing Line", True)
    
        # 7. Inverted Hammer
        if len(c) >= 1:
            body = abs(c.iloc[-1] - o.iloc[-1])
            uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
            lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
            if uw > 2 * body and lw < body:
                vol_ok, vol_ratio = volume_confirmed()
                if vol_ok:
                    pattern_score = min(1.0, 0.7 * (uw / body))
                    detected_patterns.append(("Inverted Hammer", pattern_score))
                    log_pattern_detection(symbol, "Inverted Hammer", True)
    
        # 8. Volume Breakout Candle (dual-mode detection) with wick validation
        # Mode 1: Single-candle breakout
        if len(c) >= 10:
            current_vol = v.iloc[-1]
            avg_vol = v.iloc[-10:-1].mean()
            if current_vol > 2.5 * avg_vol and c.iloc[-1] > o.iloc[-1]:
                # Check if price breaks recent high
                recent_high = max(h.iloc[-10:-1])
                # Wick validation: ensure upper wick is not too large
                candle_range = h.iloc[-1] - l.iloc[-1]
                upper_wick = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
                upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
                
                if (c.iloc[-1] > recent_high and  # Close above resistance
                    upper_wick_ratio <= 0.25):    # Max 25% upper wick allowed
                    pattern_score = min(1.0, 0.8 + (current_vol / avg_vol) / 10)
                    detected_patterns.append(("Volume Breakout Candle", pattern_score))
                    log_pattern_detection(symbol, "Volume Breakout Candle", True)
                else:
                    reason = "Failed wick validation" if upper_wick_ratio > 0.25 else "Resistance not cleared"
                    log_pattern_detection(symbol, "Volume Breakout Candle", False, reason)
        
        # Volume Breakout Pattern with retest, fade rejection and volume confirmation
        if not skip_chart and len(c) >= 20:  # Increased lookback for better resistance
            # Identify true resistance (swing high + consolidation zone)
            resistance_period = 20
            resistance_candidates = h.iloc[-resistance_period:-5]
            if len(resistance_candidates) > 0:
                resistance_level = resistance_candidates.max()
                # Verify resistance has been tested at least twice
                resistance_tests = sum(resistance_candidates >= resistance_level * 0.995)
                
                current_close = c.iloc[-1]
                current_vol = v.iloc[-1]
                avg_vol = v.iloc[-15:-1].mean() if len(v) >= 15 else 0  # Handle case with insufficient data
                
                # Breakout confirmation with stronger requirements
                breakout_condition = (
                    current_close > resistance_level * 1.005 and  # 0.5% clearance
                    avg_vol > 0 and  # Ensure we have valid volume data
                    current_vol > max(2.5 * avg_vol, 1.5 * v.iloc[-2]) and  # Strong volume surge
                    resistance_tests >= 2  # At least 2 prior tests
                )
                
                # Retest validation (check previous candle)
                retest_condition = False
                if len(c) >= 2:
                    prev_close = c.iloc[-2]
                    prev_low = l.iloc[-2]
                    # Valid retest: touched resistance then bounced
                    retest_condition = (
                        prev_low <= resistance_level * 1.005 and
                        prev_close > resistance_level * 0.995 and
                        current_close > prev_close  # Confirming upward momentum
                    )
                
                # Reject breakout fades (current candle shouldn't close near low)
                candle_range = h.iloc[-1] - l.iloc[-1]
                no_fade_condition = candle_range > 0 and (current_close - l.iloc[-1]) / candle_range > 0.33
                
                # Volume breakout confirmation
                volume_breakout = current_vol > 1.8 * v.iloc[-2]  # Volume must be 80% higher than previous
                
                if breakout_condition and (retest_condition or no_fade_condition) and volume_breakout:
                    pattern_score = min(1.0, 0.9 + (current_vol / avg_vol) / 10)  # Stronger weighting
                    
                    # Remove Volume Breakout Candle if exists
                    detected_patterns = [p for p in detected_patterns if p[0] != "Volume Breakout Candle"]
                    
                    detected_patterns.append(("Volume Breakout Pattern", pattern_score))
                    log_pattern_detection(symbol, "Volume Breakout Pattern", True)
                else:
                    reasons = []
                    if not breakout_condition: 
                        reasons.append(f"Resistance: {resistance_level:.2f}, Vol: {current_vol/avg_vol:.1f}x, Tests: {resistance_tests}")
                    if not retest_condition and len(c) >= 2: 
                        reasons.append("No valid retest")
                    if not no_fade_condition: 
                        reasons.append("Breakout fade detected")
                    if not volume_breakout:
                        reasons.append(f"Volume insufficient ({current_vol/v.iloc[-2]:.1f}x < 1.8x)")
                        
                    log_pattern_detection(symbol, "Volume Breakout Pattern", False, " | ".join(reasons))
    
        # 9. Gap-Down Reversal (with next candle confirmation)
        if len(c) >= 3:
            # Check gap down between candle -2 and candle -1
            gap_down = o.iloc[-2] < c.iloc[-3]
            gap_size = (c.iloc[-3] - o.iloc[-2]) / c.iloc[-3] if gap_down else 0
            
            # Strong reversal candle (candle -1)
            strong_reversal = c.iloc[-2] > o.iloc[-2] and (o.iloc[-2] - c.iloc[-2]) > (h.iloc[-2] - l.iloc[-2]) * 0.7
            
            # Next candle (current candle) confirmation
            confirmation = (c.iloc[-1] > o.iloc[-1] and  # Bullish
                        c.iloc[-1] > c.iloc[-2] and   # Closes above previous close
                        v.iloc[-1] > v.iloc[-2])      # Volume increases
            
            if gap_down and gap_size > 0.01 and strong_reversal and confirmation:
                vol_ok, vol_ratio = volume_confirmed(index=-1, multiplier=1.7)
                if vol_ok:
                    pattern_score = min(1.0, 0.75 + gap_size * 50)
                    detected_patterns.append(("Gap-Down Reversal", pattern_score))
                    log_pattern_detection(symbol, "Gap-Down Reversal", True)
    
        # 10. Bullish Kicker
        if len(c) >= 2:
            if (c.iloc[-2] < o.iloc[-2] and
                o.iloc[-1] > c.iloc[-2] and
                c.iloc[-1] > o.iloc[-1]):
                vol_ok, vol_ratio = volume_confirmed()
                if vol_ok:
                    pattern_score = candle_score(-1) * 0.85
                    detected_patterns.append(("Bullish Kicker", pattern_score))
                    log_pattern_detection(symbol, "Bullish Kicker", True)
        
        # 11. Bullish Hammer
        if len(c) >= 1:
            body = abs(c.iloc[-1] - o.iloc[-1])
            lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
            uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
            if lw > 2 * body and uw < body * 0.5:
                vol_ok, vol_ratio = volume_confirmed()
                if vol_ok:
                    pattern_score = min(1.0, 0.75 * (lw / body))
                    detected_patterns.append(("Bullish Hammer", pattern_score))
                    log_pattern_detection(symbol, "Bullish Hammer", True)

    # =====================
    # SKIP CHART PATTERNS IF STRONG CANDLE FOUND
    # =====================
    skip_chart = False
    strong_candle_found = False
    if SKIP_CHART_IF_CANDLE_FOUND and detected_patterns:
        # Only skip if pattern is NOT breakout type
        for pattern_name, _ in detected_patterns:
            weight = PATTERN_WEIGHTS.get(pattern_name, {"weight": 1.0})["weight"]
            if weight >= CANDLE_STRONG_THRESHOLD and pattern_name not in BREAKOUT_PATTERNS:
                skip_chart = True
                strong_candle_found = True
                print(f"⏩ Strong non-breakout pattern found ({pattern_name}), skipping chart patterns")
                break

    # =====================
    # CHART PATTERN DETECTION (CONDITIONAL)
    # =====================
    if not skip_chart:
        # 1. Cup and Handle pattern
        if not skip_chart and len(c) >= 40:
            # Dynamic index calculation for variable data length
            left_rim_idx = max(-len(c), -40)  # Use available data
            left_rim_high = h.iloc[left_rim_idx]
            
            # Find cup bottom (lowest point in the cup) - FIXED: Use lows (l) instead of highs (h)
            cup_slice = l.iloc[-35:-15]
            if not cup_slice.empty:
                cup_bottom_idx = cup_slice.idxmin()
                cup_bottom = l.iloc[cup_bottom_idx]
            else:
                cup_bottom = 0
            cup_bottom = l.iloc[cup_bottom_idx]
            
            # Find right rim (high point after cup)
            right_rim_high = h.iloc[-15]
            
            # Handle formation (small downward drift)
            handle_lows = l.iloc[-15:-5]
            handle_highs = h.iloc[-15:-5]
            
            # Cup should be U-shaped, handle should be downward drift
            if (left_rim_high * 0.98 < right_rim_high < left_rim_high * 1.02 and
                cup_bottom < left_rim_high * 0.85 and
                max(handle_highs) < right_rim_high and
                min(handle_lows) > cup_bottom and
                c.iloc[-1] > right_rim_high and  # Close above resistance
                chart_pattern_trend is not False):  # Trend confirmation
                
                vol_ok, vol_ratio = volume_confirmed(index=-1, multiplier=1.8, pattern_type="Cup and Handle")
    
                if vol_ok:
                    pattern_score = min(1.0, 0.8 + (vol_ratio / 5.0))
                    detected_patterns.append(("Cup and Handle", pattern_score))
                    log_pattern_detection(symbol, "Cup and Handle", True)
                else:
                    log_pattern_detection(symbol, "Cup and Handle", False, f"Volume insufficient ({vol_ratio:.2f} < 1.8x)")
        
        # 2. Double/Triple Bottom
        if len(c) >= 20:
            # Find significant lows
            lows = l.rolling(5).min().dropna()
            min_idxs = lows.nsmallest(3).index.tolist()
            min_idxs.sort()            
            # Enhanced with explicit validation and logging
            if len(min_idxs) >= 2:
                # Validate index positions
                if min_idxs[0] >= min_idxs[1]:
                    reason = f"Invalid min_idx range: {min_idxs[0]} to {min_idxs[1]}"
                    log_pattern_detection(symbol, "Double/Triple Bottom", False, reason)
                    # Skip pattern but continue evaluation
                    return False, None, 0.0
                
                # Validate sufficient gap between lows
                if min_idxs[1] - min_idxs[0] <= 1:
                    reason = "Not enough candles between lows to form resistance"
                    log_pattern_detection(symbol, "Double/Triple Bottom", False, reason)
                    # Skip pattern but continue evaluation
                    return False, None, 0.0
                
                # Check if lows are approximately equal
                low1 = lows.iloc[min_idxs[0]]
                low2 = lows.iloc[min_idxs[1]]
                
                # Guard against zero division
                if min(low1, low2) <= 0:
                    reason = "Invalid low price (<=0) detected"
                    log_pattern_detection(symbol, "Double/Triple Bottom", False, reason)
                    return False, None, 0.0
                    
                low_diff = abs(low1 - low2) / min(low1, low2)
        
                # Validate resistance slice with explicit error
                try:
                    resistance_slice = h.iloc[min_idxs[0]:min_idxs[1]]
                    
                    if resistance_slice.empty:
                        raise ValueError(f"Empty resistance slice for {symbol} "
                                        f"between indexes {min_idxs[0]} and {min_idxs[1]}")
                    
                    resistance = resistance_slice.max()
                    
                except IndexError as e:
                    reason = f"Index error in resistance calculation: {str(e)}"
                    log_pattern_detection(symbol, "Double/Triple Bottom", False, reason)
                    return False, None, 0.0
                    
                except ValueError as e:
                    reason = f"Invalid resistance data: {str(e)}"
                    log_pattern_detection(symbol, "Double/Triple Bottom", False, reason)
                    return False, None, 0.0
            
                if (low_diff < 0.02 and c.iloc[-1] > resistance):  # Close above resistance 
                    vol_ok, vol_ratio = volume_confirmed(multiplier=1.5, pattern_type="Double/Triple Bottom")
                    pattern_name = "Double Bottom" if len(min_idxs) == 2 else "Triple Bottom"
                    if vol_ok:
                        pattern_score = min(1.0, 0.7 + (c.iloc[-1] - resistance) / resistance * 5)
                        detected_patterns.append((pattern_name, pattern_score))
                        log_pattern_detection(symbol, pattern_name, True)
                    else:
                        log_pattern_detection(symbol, pattern_name, False, f"Volume insufficient ({vol_ratio:.2f} < 1.5x)") 
                
        
        # 3. Ascending Triangle
        if len(c) >= 15:
            # Horizontal resistance
            resistance = h.rolling(5).max().iloc[-15:].mean()
            resistance_range = h.iloc[-15:].max() - h.iloc[-15:].min()
            
            # Rising lows
            low_min = l.iloc[-15:].min()
            low_max = l.iloc[-15:].max()
            slope, _ = np.polyfit(range(15), l.iloc[-15:], 1)
            
            if (resistance_range / resistance < 0.02 and
                slope > 0 and
                (low_max - low_min) / low_min > 0.03 and
                c.iloc[-1] > resistance and  # Close above resistance
                chart_pattern_trend is not False):  # Trend confirmation
                
                vol_ok, vol_ratio = volume_confirmed(multiplier=1.4, pattern_type="Ascending Triangle")
                if vol_ok:
                    pattern_score = min(1.0, 0.75 + slope * 100)
                    detected_patterns.append(("Ascending Triangle", pattern_score))
                    log_pattern_detection(symbol, "Ascending Triangle", True)
                else:
                    log_pattern_detection(symbol, "Ascending Triangle", False, f"Volume insufficient ({vol_ratio:.2f} < 1.4x)")
        
        # 4. Inverse Head and Shoulders
        if len(c) >= 25:
            # Find left shoulder
            left_shoulder_idx = h.iloc[-25:-15].idxmax()
            left_shoulder = h.iloc[left_shoulder_idx]
            
            # Find head (lowest point)
            head_idx = l.iloc[-20:-10].idxmin()
            head_low = l.iloc[head_idx]
            
            # Find right shoulder
            right_shoulder_idx = h.iloc[-10:-5].idxmax()
            right_shoulder = h.iloc[right_shoulder_idx]
            
            # Neckline (resistance)
            neckline = (left_shoulder + right_shoulder) / 2
            
            if (left_shoulder * 0.98 < right_shoulder < left_shoulder * 1.02 and
                head_low < left_shoulder * 0.95 and
                c.iloc[-1] > neckline):  # Close above neckline
                
                vol_ok, vol_ratio = volume_confirmed(multiplier=1.6, pattern_type="Inverse Head and Shoulders")
                if vol_ok:
                    pattern_score = min(1.0, 0.85 + (c.iloc[-1] - neckline) / neckline * 10)
                    detected_patterns.append(("Inverse Head and Shoulders", pattern_score))
                    log_pattern_detection(symbol, "Inverse Head and Shoulders", True)
        
        # 5. Bullish Rectangle
        if len(c) >= 10:
            # Horizontal support and resistance
            support = l.rolling(5).min().iloc[-10:].mean()
            resistance = h.rolling(5).max().iloc[-10:].mean()
            range_pct = (resistance - support) / support
            
            # Consolidation range with breakout confirmation
            if range_pct < 0.03 and c.iloc[-1] > resistance:  # Close above resistance
                # Volume vs prior resistance tests
                resistance_tests = [v for i, candle in enumerate(candles[-20:]) 
                                   if candle['high'] >= resistance_level * 0.995]
                test_vol_avg = sum(resistance_tests) / len(resistance_tests) if resistance_tests else 0
                
                # Volume confirmation: current volume > 1.5x resistance test average
                vol_ok = current_vol > test_vol_avg * 1.5 if test_vol_avg > 0 else False
                vol_ratio = current_vol / test_vol_avg if test_vol_avg > 0 else 0
                # Price confirmation (closing above resistance)
                price_ok = c.iloc[-1] > resistance
                
                if vol_ok and price_ok:
                    pattern_score = min(1.0, 0.7 + vol_ratio / 2.0)
                    detected_patterns.append(("Bullish Rectangle", pattern_score))
                    log_pattern_detection(symbol, "Bullish Rectangle", True)
                else:
                    reason = ""
                    if not vol_ok: reason = f"Insufficient volume ({vol_ratio:.2f}x < 1.5x)"
                    if not price_ok: reason = f"Failed to close above resistance ({c.iloc[-1]:.2f} < {resistance:.2f})"
                    log_pattern_detection(symbol, "Bullish Rectangle", False, reason)
        
        # 6. Rounding Bottom
        if len(c) >= 30:
            # Fit polynomial curve to lows
            idx = np.array(range(30))
            lows = l.iloc[-30:].values
            coeffs = np.polyfit(idx, lows, 2)
            
            # Check U-shape (positive quadratic coefficient)
            if coeffs[0] > 0:
                # Check if current price is above starting point
                start_price = lows[0]
                if c.iloc[-1] > start_price:
                    pattern_score = min(1.0, 0.65 + (c.iloc[-1] - start_price) / start_price * 20)
                    detected_patterns.append(("Rounding Bottom", pattern_score))
                    log_pattern_detection(symbol, "Rounding Bottom", True)
        
        # 7. Bullish Pennant
        if len(c) >= 20:
            # Flagpole: sharp price movement
            flagpole_start = c.iloc[-20]
            flagpole_end = c.iloc[-15]
            flagpole_move = abs(flagpole_end - flagpole_start) / flagpole_start
            
            if flagpole_move > 0.05:  # At least 5% move
                # Pennant: converging trendlines with lower highs and higher lows
                highs = h.iloc[-15:]
                lows = l.iloc[-15:]
                
                # Fit trendlines
                high_slope, _ = np.polyfit(range(15), highs, 1)
                low_slope, _ = np.polyfit(range(15), lows, 1)
                
                # Pennant should have downward sloping highs and upward sloping lows
                if high_slope < 0 and low_slope > 0:
                    # Volume should decrease during pennant formation
                    vol_start = v.iloc[-15]
                    vol_end = v.iloc[-1]
                    if vol_end < vol_start * 0.7:
                        pattern_score = min(1.0, 0.7 + (flagpole_move * 10))
                        detected_patterns.append(("Bullish Pennant", pattern_score))
                        log_pattern_detection(symbol, "Bullish Pennant", True)
        
        # 8. Bullish Wedge (Falling Wedge)
        if len(c) >= 20:
            # Converging trendlines both sloping downward
            highs = h.iloc[-20:]
            lows = l.iloc[-20:]
            
            high_slope, _ = np.polyfit(range(20), highs, 1)
            low_slope, _ = np.polyfit(range(20), lows, 1)
            
            # Both should be negative but lows slope less negative (converging)
            if high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope):
                # Breakout above upper trendline
                if c.iloc[-1] > max(highs.iloc[:-1]):  # Close above resistance
                    vol_ok, vol_ratio = volume_confirmed(multiplier=1.3, pattern_type="Bullish Wedge")
                    if vol_ok:
                        pattern_score = min(1.0, 0.75 + (abs(high_slope) * 100))
                        detected_patterns.append(("Bullish Wedge (Falling Wedge)", pattern_score))
                        log_pattern_detection(symbol, "Bullish Wedge (Falling Wedge)", True)
        
        # 9. Rounded Consolidation (Roundboom)
        if len(c) >= 25:
            # Fit polynomial curve to closes
            idx = np.array(range(25))
            closes = c.iloc[-25:].values
            coeffs = np.polyfit(idx, closes, 2)
            
            # Check U-shape in consolidation (positive quadratic coefficient)
            if coeffs[0] > 0:
                # Consolidation range
                high_point = max(h.iloc[-25:])
                low_point = min(l.iloc[-25:])
                consolidation_range = (high_point - low_point) / low_point
                
                if consolidation_range < 0.08:  # Tight consolidation
                    # Volume should be higher at the edges
                    edge_vol = (v.iloc[-25] + v.iloc[-1]) / 2
                    center_vol = v.iloc[-12:-8].mean()
                    if edge_vol > center_vol * 1.5:
                        pattern_score = min(1.0, 0.7 + consolidation_range * 10)
                        detected_patterns.append(("Rounded Consolidation (Roundboom)", pattern_score))
                        log_pattern_detection(symbol, "Rounded Consolidation (Roundboom)", True)
        
        # 10. Bullish Flag (New pattern)
        if len(c) >= 15:
            # Flagpole: sharp price movement (at least 5% in 3-5 candles)
            flagpole_rise = 0
            flagpole_end_idx = None
            for i in range(5, 10):
                start_idx = -i-5
                end_idx = -i
                if start_idx < -len(c) or end_idx >= 0:
                    continue
                rise = (c.iloc[end_idx] - c.iloc[start_idx]) / c.iloc[start_idx]
                if rise > flagpole_rise and rise > 0.05:
                    flagpole_rise = rise
                    flagpole_end_idx = end_idx
            
            if flagpole_end_idx is not None:
                # Flag: consolidation with decreasing volume
                flag_highs = h.iloc[flagpole_end_idx+1:-1]
                flag_lows = l.iloc[flagpole_end_idx+1:-1]
                if len(flag_highs) < 4:  # At least 4 candles in flag
                    pass
                else:
                    resistance = flag_highs.max()
                    # Breakout with volume confirmation
                    if c.iloc[-1] > resistance and v.iloc[-1] > v.iloc[-2] * 1.5:
                        flag_vol = v.iloc[flagpole_end_idx+1:-1].mean()
                        flagpole_vol = v.iloc[flagpole_end_idx-4:flagpole_end_idx+1].mean()
                        if flag_vol < flagpole_vol * 0.7:
                            pattern_score = min(1.0, 0.7 + flagpole_rise)
                            detected_patterns.append(("Bullish Flag", pattern_score))
                            log_pattern_detection(symbol, "Bullish Flag", True)
        
        # 11. Symmetrical Triangle (New pattern)
        if len(c) >= 20:
            # Converging trendlines (at least 10 candles)
            triangle_highs = h.iloc[-15:-1]
            triangle_lows = l.iloc[-15:-1]
            
            if len(triangle_highs) < 10:
                pass
            else:
                # Fit trendlines
                idx = np.array(range(len(triangle_highs)))
                high_slope, high_intercept = np.polyfit(idx, triangle_highs, 1)
                low_slope, low_intercept = np.polyfit(idx, triangle_lows, 1)
                
                # Slopes should converge (high slope negative, low slope positive)
                if high_slope < 0 and low_slope > 0:
                    # Breakout above upper trendline
                    current_idx = len(triangle_highs)
                    upper_trendline = high_slope * current_idx + high_intercept
                    if c.iloc[-1] > upper_trendline and v.iloc[-1] > v.iloc[-2] * 1.2:
                        pattern_score = min(1.0, 0.8 + (abs(high_slope) + abs(low_slope)) / 2 * 100)
                        detected_patterns.append(("Symmetrical Triangle", pattern_score))
                        log_pattern_detection(symbol, "Symmetrical Triangle", True)

    # =====================
    # PATTERN SELECTION LOGIC
    # =====================
    if detected_patterns:
        # Calculate composite scores with priority for candlestick patterns
        scored_patterns = []
        candle_patterns = ["Morning Star", "Bullish Engulfing", "Bullish Kicker", "Breakout Marubozu"]
        max_weight = max(PATTERN_WEIGHTS.values(), key=lambda x: x["weight"])["weight"]
        
        for name, score in detected_patterns:
            weight = PATTERN_WEIGHTS.get(name, {"weight": 1.0})["weight"]
            normalized_weight = weight / max_weight
            composite_score = normalized_weight * score
            
            # Boost all strong patterns (weight > threshold)
            if weight >= CANDLE_STRONG_THRESHOLD:
                composite_score *= 1.20
                print(f"🚀 Boosting strong pattern {name} by 20%")
            
            scored_patterns.append((name, score, composite_score))
        
        # Select pattern with highest composite score
        best_pattern = max(scored_patterns, key=lambda x: x[2])
        print(f"🏆 Selected pattern: {best_pattern[0]} (Score: {best_pattern[1]:.2f}, Composite: {best_pattern[2]:.2f})")
        return True, best_pattern[0], best_pattern[1]
    
    return False, None, 0.0

def compute_rsi_macd(closes):
    delta = closes.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean().bfill()
    avg_loss = loss.rolling(14).mean().bfill().replace(0, 0.01)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    macd_crossover = macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
    return rsi.iloc[-1], histogram.iloc[-1], macd_crossover

def is_index_bullish(index_id):
    """Check bullishness for any index (NIFTY/Sector)"""
    candles = fetch_candles(
        index_id, 
        count=20,
        exchange_segment="NSE_INDEX",
        instrument_type="INDEX"
    )
    if not candles:
        print(f"❌ No candles returned for index ID {index_id} — skipping bullish check")
        return False

    closes = pd.Series([c["close"] for c in candles])
    rsi, macd_hist, macd_cross = compute_rsi_macd(closes)
    detected, _, _ = detect_bullish_pattern(candles)
    return detected and rsi > 50 and macd_hist > 0 and macd_cross

def check_breakout(candles, period=3):
    """Confirm 15-min high breakout with closing confirmation"""
    try:
        if len(candles) < period + 1:
            return False
        current_close = candles[-1]['close']
        prev_candles = candles[-period-1:-1]
        if not prev_candles:
            return False
        prev_high = max(c['high'] for c in prev_candles)
        return current_close > prev_high
    except (IndexError, ValueError, KeyError):
        return False

def check_gap_up(security_id):
    """Prevent entries after significant gap-ups with recent price check"""
    try:
        quote_data = dhan.ohlc_data(securities={"NSE_EQ": [int(security_id)]})
        quote = None
        if "NSE_EQ" in quote_data and str(security_id) in quote_data["NSE_EQ"]:
            quote = quote_data["NSE_EQ"][str(security_id)]
        elif "NSE_EQ" in quote_data and int(security_id) in quote_data["NSE_EQ"]:
            quote = quote_data["NSE_EQ"][int(security_id)]
        if not quote:
            return False

        prev_close = float(quote.get('previousClose', 0))
        today_open = float(quote.get('open', 0))
        current_price = float(quote.get('last_price', 0))

        if prev_close <= 0:
            return False

        gap_up = (today_open - prev_close) / prev_close > 0.01

        # Block all gap-ups >1% regardless of pullback
        if gap_up:
            print(f"❌ Gap-up detected: {security_id} (Open: {today_open:.2f} > Prev Close: {prev_close:.2f})")
            return True
        return False
    except Exception as e:
        print(f"❌ Error in check_gap_up for {security_id}: {e}")
        return False
            
        return gap_up
    except:
        return False

def is_near_support(candles, buffer=0.005):
    """Check if current price is near support level (within buffer%)"""
    if len(candles) < 10:
        return False
    # Find recent low (support level)
    recent_lows = [candle['low'] for candle in candles[-10:]]
    support_level = min(recent_lows)
    current_price = candles[-1]['close']
    # Check if current price is within buffer above support
    return current_price <= support_level * (1 + buffer)

def get_15min_trend(candles_1min, min_candles=5):  # Reduced minimum to 5
    """Check if 15-minute trend is bullish using linear regression slope"""
    if len(candles_1min) < min_candles * 15:
        return False
    df = pd.DataFrame(candles_1min)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Resample to 15 minutes
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_15min = df.resample('15min').apply(ohlc_dict).dropna()
    if len(df_15min) < min_candles:
        return False
    # Get closing prices
    closes = df_15min['close'].iloc[-min_candles:]
    # Calculate slope
    x = np.arange(len(closes))
    slope, _, _, _, _ = linregress(x, closes)
    # Bullish if slope positive
    return slope > 0

def detect_bearish_pattern(candles, current_pattern=None):
    """Enhanced bearish pattern detection with conflict checking, returns confidence score (0-1)"""
    if len(candles) < 3:
        return 0.0
    df = pd.DataFrame(candles)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    
    # Pattern-specific conflict rules
    if current_pattern in REVERSAL_PATTERNS:
        lookback = 2
    else:
        lookback = 3
        
    bearish_score = 0.0
    
    # 1. Doji: small body relative to range
    for i in range(-lookback,0):
        body = abs(c.iloc[i] - o.iloc[i])
        total_range = h.iloc[i] - l.iloc[i]
        if total_range == 0:
            continue
        body_ratio = body / total_range
        if body_ratio < 0.1:
            # Stronger weight for recent candles
            recency_factor = 1.0 - (abs(i) / (lookback + 1))
            bearish_score = max(bearish_score, 0.5 * recency_factor)
            
    # 2. Shooting Star: long upper wick, small body
    body = abs(c.iloc[-1] - o.iloc[-1])
    if body > 0:
        upper_wick = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        lower_wick = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
        if upper_wick > 2 * body and lower_wick < body:
            bearish_score = max(bearish_score, 0.7)
    
    # 3. Bearish Engulfing
    if len(c) >= 2:
        if (c.iloc[-2] > o.iloc[-2] and  # Previous candle green
            c.iloc[-1] < o.iloc[-1] and   # Current candle red
            o.iloc[-1] > c.iloc[-2] and 
            c.iloc[-1] < o.iloc[-2]):
            bearish_score = max(bearish_score, 0.8)
    
    # 4. Dark Cloud Cover
    if len(c) >= 2:
        prev_mid = (o.iloc[-2] + c.iloc[-2]) / 2
        if (c.iloc[-2] > o.iloc[-2] and  # Prev green
            o.iloc[-1] > c.iloc[-2] and   # Open above prev close
            c.iloc[-1] < prev_mid and     # Close below midpoint
            c.iloc[-1] > o.iloc[-2]):     # Close above prev open
            bearish_score = max(bearish_score, 0.75)
    
    # 5. Evening Star (only if current pattern is reversal)
    body_middle = abs(o.iloc[-2] - c.iloc[-2])
    body_range = h.iloc[-2] - l.iloc[-2]
    if (c.iloc[-3] > o.iloc[-3] and  # First candle green
        body_middle < 0.3 * body_range and  # Small body
        o.iloc[-1] < c.iloc[-2] and  # Third candle opens below middle
        c.iloc[-1] < (o.iloc[-3] + c.iloc[-3]) / 2):  # Closes below midpoint of first
        bearish_score = max(bearish_score, 0.9)   
    
    return bearish_score

def log_trade(symbol, security_id, action, price, qty, status, stop_pct=None, target_pct=None, stop_price=None, target_price=None, order_id="N/A", timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%m/%d/%Y %H:%M:%S")
    
    # CSV row log
    log_row = [
        symbol,
        timestamp_str,
        security_id,
        qty,
        price,
        0,  # momentum_5min placeholder
        target_pct if target_pct is not None else "",
        stop_pct if stop_pct is not None else "",
        "",  # live_price
        "",  # change_pct
        "",  # last_checked
        status,  # ✅ Correctly placed here
        "",  # exit_price
        order_id,
        target_price if target_price is not None else "",
        stop_price if stop_price is not None else ""
    ]

    # CSV logging
    with open(PORTFOLIO_LOG, "a") as f:
        f.write(",".join(map(str, log_row)) + "\n")

    # DB logging
    try:
        insert_portfolio_log_to_db(
            trade_date=timestamp,
            symbol=symbol,
            security_id=security_id,
            quantity=qty,
            buy_price=price,
            stop_pct=float(stop_pct) if stop_pct is not None else None,
            target_pct=float(target_pct) if target_pct is not None else None,
            stop_price=stop_price,
            target_price=target_price,
            status=status,
            order_id=order_id
        )
    except Exception as e:
        print("❌ DB log failed (portfolio_log):", e)

def place_order(symbol, security_id, qty, price, pattern_name, candles, tick_size_map, capital, breakout_level=None):
    if price <= 0:
        print(f"⚠️ Invalid price for {symbol}: {price}")
        return
        
    # Store breakout level if applicable
    is_breakout_trade = pattern_name in BREAKOUT_PATTERNS and breakout_level is not None
        
    # Validate tick size exists, fail loudly if missing
    if str(security_id) not in tick_size_map:
        # Fallback to price-based tick size
        if price < 100:
            tick_size_value = 0.05
        elif price < 500:
            tick_size_value = 0.10
        else:
            tick_size_value = 0.50
        print(f"⚠️ Using fallback tick size {tick_size_value} for {security_id}")
    else:
        tick_size_value = tick_size_map[str(security_id)]
    tick_size_dec = Decimal(str(tick_size_value))
    
    limit_price = Decimal(str(price)) * Decimal("1.002")
    limit_price = (limit_price / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec
    limit_price = float(limit_price)
    
    # Calculate ADR for realistic TP/SL capping
    adr = calculate_adr(security_id)
    
    # ========== SL/TP CALCULATION (MOVED UP) ==========
    # Base parameters
    base_sl_pct = 0.005
    base_tp_pct = 0.01
    
    # Pattern-specific adjustments
    pattern_conf = PATTERN_WEIGHTS.get(pattern_name, {"weight": 1.0, "vol_scale": 1.0})
    conf_weight = pattern_conf["weight"]
    vol_scale = pattern_conf["vol_scale"]
    
    base_tp_pct = 0.01 * conf_weight
    base_sl_pct = 0.005 * (2 - conf_weight/2)  # Inverse to weight
    
    # Volatility adjustment using ATR
    atr = calculate_atr(candles)
    entry_price = Decimal(str(limit_price))  # Use limit_price instead of price
    atr_multiplier = float(vol_scale) * (float(atr) / float(entry_price)) if atr > 0 else 1.0
    
    # Apply volatility scaling
    tp_pct = max(base_tp_pct, atr_multiplier)
    sl_pct = min(base_sl_pct, atr_multiplier * 0.7)
    
    # Enforce minimum SL of 1.5x ATR
    min_sl_pct = 1.5 * (atr / float(entry_price))
    if sl_pct < min_sl_pct:
        sl_pct = min_sl_pct
        print(f"🔧 Adjusted SL to minimum 1.5x ATR: {sl_pct*100:.2f}%")
    
    # Tiered time decay adjustment
    now_time = datetime.now().time()
    if now_time < dtime(13, 0):     # Before 1 PM
        time_decay = 1.0
    elif now_time < dtime(14, 30):  # 1 PM - 2:30 PM
        time_decay = 0.9
    else:                           # After 2:30 PM
        # Calculate minutes past 2:30 PM
        minutes_past = (now_time.hour - 14) * 60 + (now_time.minute - 30)
        decay_factor = min(1.0, minutes_past / 60)  # 60 minutes until close
        time_decay = max(0.5, 0.9 - (0.4 * decay_factor))
    
    tp_pct *= time_decay
    
    # Ensure minimum 1:2 risk-reward ratio
    if tp_pct / sl_pct < 2:
        tp_pct = sl_pct * 2.2
    
    # Calculate final SL and TP
    stop_loss = float(entry_price * (Decimal(1) - Decimal(sl_pct)))
    target = float(entry_price * (Decimal(1) + Decimal(tp_pct)))
    
    # Apply ADR capping
    max_move = adr * 0.3
    target = min(target, limit_price + max_move)
    stop_loss = max(stop_loss, limit_price - max_move * 0.7)
    
    # Round to tick size
    stop_loss = float((Decimal(str(stop_loss)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
    target = float((Decimal(str(target)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
    
    # Ensure TP ≠ SL
    if abs(target - stop_loss) < float(tick_size_value):
        target += float(tick_size_value)
        print(f"⚠️ Adjusted TP to avoid overlap with SL: ₹{target:.2f}")
    
    # Calculate remaining market hours (9:30-15:30)
    now = datetime.now().time()
    market_open = dtime(9, 30)
    market_close = dtime(15, 30)
    if now < market_open:
        remaining_hours = 6.0
    elif now >= market_close:
        remaining_hours = 0.0
    else:
        remaining_seconds = (market_close.hour * 3600 + market_close.minute * 60) - \
                        (now.hour * 3600 + now.minute * 60)
        remaining_hours = max(0, remaining_seconds / 3600)  # Ensures non-negative
    
    # Time feasibility check
    required_move = (target - limit_price) / limit_price
    max_allowed_move = 0.015 * (remaining_hours / 1.5) if remaining_hours > 0 else 0
    if required_move > max_allowed_move > 0:
        target = limit_price * (1 + max_allowed_move)
        target = float((Decimal(str(target)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
        send_telegram(f"⚠️ Adjusted {symbol} target to ₹{target:.2f} for time constraints")
    # ========== END SL/TP CALCULATION ==========
    
    # For breakout patterns: Ensure SL is below breakout level
    if is_breakout_trade:
        # Dynamic buffer based on volatility (ATR)
        atr = calculate_atr(candles)
        buffer = max(0.005, min(0.02, atr / price * 2))  # 0.5%-2% range
        breakout_stop = breakout_level * (1 - buffer)
        if stop_loss < breakout_stop:
            stop_loss = breakout_stop
            print(f"🔧 Adjusted SL to breakout level (₹{stop_loss:.2f}) for {symbol}")

    order = {
        "security_id": str(security_id),
        "exchange_segment": "NSE_EQ",
        "transaction_type": "BUY",
        "order_type": "LIMIT",
        "product_type": "CNC",
        "quantity": qty,
        "price": limit_price,
        "validity": "DAY"
    }
    
    try:
        res = dhan.place_order(**order)
        
        if res.get('status') == 'REJECTED':
            print(f"❌ Order rejected for {symbol}: {res.get('message', 'No message')}")
            send_telegram(f"❌ Order rejected for {symbol}: {res.get('message', 'No message')}")
            return
            
        print("✅ Order Placed:", res)
        msg = f"✅ BUY {symbol} Qty: {qty} @ ₹{limit_price}"
        if pattern_name:
            msg += f" | Pattern: {pattern_name}"
        send_telegram(msg)
        now = datetime.now()
        log_trade(
            symbol=symbol,
            security_id=security_id,
            action="BUY",
            price=limit_price,
            qty=qty,
            status="HOLD",
            stop_pct=sl_pct,
            target_pct=tp_pct,
            stop_price=stop_loss,
            target_price=target,
            order_id=res.get("data", {}).get("orderId", "N/A"),
            timestamp=now
        )
        
        # Enhanced Stop Loss and Target via Forever Order
        try:
            # Pattern confirmation for strong patterns (using initial SL/TP values)
            strong_patterns = ["Morning Star", "Bullish Engulfing", "Bullish Kicker", "Breakout Marubozu"]
            if pattern_name in strong_patterns:
                # Add confirmation check
                time.sleep(2)  # Wait for next candle
                next_candle = fetch_candles(security_id, count=1)
                if next_candle and next_candle[0]:
                    pattern_candle_close = candles[-1]['close']
                    next_close = next_candle[0]['close']
                    next_open = next_candle[0]['open']
                    next_volume = next_candle[0]['volume']
                    pattern_volume = candles[-1]['volume']
                    
                    # Confirmation conditions
                    confirmed = (
                        next_close > pattern_candle_close and 
                        next_close > next_open and 
                        next_volume > pattern_volume
                    )
                    
                    if confirmed:
                        # Pattern-specific boost factors
                        boost_factors = {
                            "Morning Star": 1.2,
                            "Bullish Engulfing": 1.15,
                            "Bullish Kicker": 1.15,
                            "Breakout Marubozu": 1.1
                        }
                        boost = boost_factors.get(pattern_name, 1.1)
                        
                        # Convert to Decimal for safe arithmetic
                        tp_pct_dec = Decimal(str(tp_pct)) * Decimal(str(boost))
                        target = float(entry_price * (Decimal(1) + tp_pct_dec))
                        
                        # Recalculate stop loss to maintain risk-reward ratio
                        risk = float(entry_price - Decimal(str(stop_loss)))
                        reward = float(Decimal(str(target)) - entry_price)
                        if reward / risk < 2.0:  # Maintain min 1:2 risk-reward
                            sl_pct = (reward * 0.45) / float(entry_price)  # 45% of reward as risk
                            stop_loss = float(entry_price * (Decimal(1) - Decimal(sl_pct)))
                            stop_loss = float((Decimal(str(stop_loss)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
                            print(f"🔁 Adjusted SL to maintain risk-reward: ₹{stop_loss:.2f}")
                        
                        target = float((Decimal(str(target)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
                        print(f"🌟 {pattern_name} confirmation - New target: ₹{target:.2f}, New SL: ₹{stop_loss:.2f}")
                        
                        # Implement break-even SL shift when profit > 1.5x risk
                        current_profit = float(Decimal(str(next_close)) - entry_price)
                        if current_profit > 1.5 * risk:
                            # Shift SL to entry price (break-even)
                            stop_loss = float(entry_price)
                            print(f"🔒 Moving to break-even SL at entry price: ₹{stop_loss:.2f}")
                    else:
                        print(f"⚠️ {pattern_name} not confirmed by next candle. Proceeding with original target.")
                else:
                    print(f"⚠️ Could not fetch next candle for {pattern_name} confirmation. Proceeding.")
        
            # Small delay to avoid overlap
            time.sleep(1.5)
        
            response = dhan.place_forever(
                security_id=str(security_id),
                exchange_segment="NSE_EQ",
                transaction_type="SELL",
                product_type="CNC",
                quantity=qty,
                price=target,
                trigger_Price=stop_loss,
                order_type="SINGLE"
            )
        
            if response.get('status') == 'success':
                print(f"🎯 SL/TP set for {symbol}: Target ₹{target:.2f}, Stop ₹{stop_loss:.2f}")
                send_telegram(
                    f"🎯 {symbol} | {pattern_name}\n"
                    f"ENTRY: ₹{limit_price:.2f} | QTY: {qty}\n"
                    f"SL: ₹{stop_loss:.2f}\n"
                    f"TARGET: ₹{target:.2f}"
                )
            else:
                print(f"⚠️ SL/TP failed for {symbol}: {response}")
                send_telegram(f"⚠️ SL/TP setup failed for {symbol}. Retrying with lower TP...")

                # 🔁 Retry with reduced TP
                target = float((Decimal(str(limit_price * 1.012)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
                print(f"🔁 Retrying Forever Order with lower TP: ₹{target:.2f}")
                response_retry = dhan.place_forever(
                    security_id=str(security_id),
                    exchange_segment="NSE_EQ",
                    transaction_type="SELL",
                    product_type="CNC",
                    quantity=qty,
                    price=target,
                    trigger_Price=stop_loss,
                    order_type="SINGLE"
                )

                if response_retry.get('status') == 'success':
                    print(f"✅ Retry success: SL/TP set for {symbol} at lower TP ₹{target:.2f}")
                    send_telegram(
                        f"✅ RETRY: {symbol} SL/TP set\n"
                        f"ENTRY: ₹{limit_price:.2f} | SL: ₹{stop_loss:.2f} | TP: ₹{target:.2f}"
                    )
                else:
                    print(f"❌ Retry also failed for SL/TP: {response_retry}")
                    send_telegram(f"❌ Retry failed: Could not set SL/TP for {symbol}")

            # Small delay to avoid overlap
            time.sleep(1.5)

            response = dhan.place_forever(
                security_id=str(security_id),
                exchange_segment="NSE_EQ",
                transaction_type="SELL",
                product_type="CNC",
                quantity=qty,
                price=target,
                trigger_Price=stop_loss,
                order_type="SINGLE"
            )

            if response.get('status') == 'success':
                print(f"🎯 SL/TP set for {symbol}: Target ₹{target:.2f}, Stop ₹{stop_loss:.2f}")
                send_telegram(
                    f"🎯 {symbol} | {pattern_name}\n"
                    f"ENTRY: ₹{limit_price:.2f} | QTY: {qty}\n"
                    f"SL: ₹{stop_loss:.2f} ({sl_pct*100:.1f}%)\n"
                    f"TARGET: ₹{target:.2f} ({tp_pct*100:.1f}%)"
                )
            else:
                print(f"⚠️ SL/TP failed for {symbol}: {response}")
                send_telegram(f"⚠️ SL/TP setup failed for {symbol}. Retrying with lower TP...")

                # 🔁 Retry with reduced TP
                target = float((Decimal(str(limit_price * 1.012)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
                print(f"🔁 Retrying Forever Order with lower TP: ₹{target:.2f}")
                response_retry = dhan.place_forever(
                    security_id=str(security_id),
                    exchange_segment="NSE_EQ",
                    transaction_type="SELL",
                    product_type="CNC",
                    quantity=qty,
                    price=target,
                    trigger_Price=stop_loss,
                    order_type="SINGLE"
                )

                if response_retry.get('status') == 'success':
                    print(f"✅ Retry success: SL/TP set for {symbol} at lower TP ₹{target:.2f}")
                    send_telegram(
                        f"✅ RETRY: {symbol} SL/TP set\n"
                        f"ENTRY: ₹{limit_price:.2f} | SL: ₹{stop_loss:.2f} | TP: ₹{target:.2f}"
                    )
                else:
                    print(f"❌ Retry also failed for SL/TP: {response_retry}")
                    send_telegram(f"❌ Retry failed: Could not set SL/TP for {symbol}")
            

        except Exception as e:
            print("⚠️ Failed to place SL/TP:", e)
            send_telegram(f"⚠️ SL/TP setup failed for {symbol}: {e}")
    except Exception as e:
        print("❌ Order Failed:", e)
        send_telegram(f"❌ Order Failed for {symbol}: {e}")
        
# ========== BREAKOUT MONITOR ==========
def monitor_breakout(security_id, breakout_level, symbol, qty):
    """
    Monitor breakout trades for 3 candles after entry
    Exit if price closes below breakout level
    """
    print(f"🔍 Starting breakout monitor for {symbol} (Level: ₹{breakout_level:.2f})")
    for i in range(3):  # Check next 3 candles
        time.sleep(60)  # Wait for next candle
        
        # Fetch latest candle with API rate limit compliance
        candles = fetch_candles(security_id, count=1)
        time.sleep(1)  # ⚠️ Added 1-second delay to comply with DHAN API rate limit
        
        if not candles:
            print(f"⚠️ Could not fetch candle for {symbol}, attempt {i+1}/3")
            continue
            
        close_price = candles[-1]['close']
        
        # Check if closed below breakout level
        if close_price < breakout_level:
            print(f"🔴 Breakout failed for {symbol}! Closing below breakout level (₹{close_price:.2f} < ₹{breakout_level:.2f})")
            try:
                # Place market sell order
                dhan.place_order(
                    security_id=str(security_id),
                    exchange_segment="NSE_EQ",
                    transaction_type="SELL",
                    order_type="MARKET",
                    product_type="CNC",
                    quantity=qty
                )
                send_telegram(f"🚨 BREAKOUT FAILURE: Sold {symbol} @ ₹{close_price:.2f} (Below breakout: ₹{breakout_level:.2f})")
                return True  # Exited position
            except Exception as e:
                print(f"❌ Failed to exit {symbol}: {e}")
                send_telegram(f"❌ BREAKOUT FAILURE EXIT ERROR: {symbol} - {str(e)}")
    
    print(f"✅ Breakout confirmed for {symbol}, stopping monitor")
    return False  # No exit triggered

def main():
    print('📌 Starting enhanced autotrade')
    
    # Precompute market close time with buffer (15:30 - 20 minutes = 15:10)
    market_close = dtime(15, 30)
    new_trade_end_time = datetime.combine(datetime.today(), dtime(15, 0))  # Extended to 3 PM for small-caps
    
    # Load master CSV once at start
    master_df = pd.read_csv(MASTER_CSV)
    print('📊 Loaded master CSV for index checks')
    
    # Create tick size map with validation
    if 'SEM_TICK_SIZE' not in master_df.columns:
        raise KeyError("❌ SEM_TICK_SIZE column missing in master CSV")
    
    if not master_df['SEM_SMST_SECURITY_ID'].notnull().all():
        raise ValueError("❌ Security IDs missing in master CSV")
    
    tick_size_map = {}
    for _, row in master_df.iterrows():
        sec_id = str(row['SEM_SMST_SECURITY_ID'])
        tick_size = row['SEM_TICK_SIZE']
        if pd.isna(tick_size):
            tick_size = 0.05  # Default tick size
        tick_size_map[sec_id] = float(tick_size)
    
    # Find NIFTY index ID once
    nifty50_row = master_df[
        master_df["SM_SYMBOL_NAME"].str.upper().str.contains("NIFTY") & 
        ~master_df["SM_SYMBOL_NAME"].str.upper().str.contains("ETF")
    ].head(1)
    
    if nifty50_row.empty:
        print("❌ NIFTY index not found in master")
        send_telegram("❌ CRITICAL: NIFTY index not found in master data")
        return
        
    nifty_id = nifty50_row.iloc[0]["SEM_SMST_SECURITY_ID"]
    
    # Continuous execution until successful trade or market close
    order_placed = False
    while not order_placed:
        # Skip if existing hold position
        if has_hold():
            print("⏩ Active hold exists - not placing new orders")
            send_telegram("⏩ Existing hold detected - no new orders placed")
            return
        
        # Skip new trades after 14:45
        if not test_mode and datetime.now() >= new_trade_end_time:
            print("⏰ New trade window closed (after 14:45)")
            send_telegram("⏰ New trade window closed - no orders placed")
            return
        
        # VIX check function
        def is_vix_ok(threshold=20.0, hard_limit=22.0):
            """
            One-shot VIX fetch per day with file-based cache to avoid NSE block.
            Returns True if VIX < threshold, False if VIX >= threshold/hard_limit.
            """
            VIX_CACHE_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/vix_cache.json"
            today = datetime.now().strftime("%Y-%m-%d")
        
            # Read cache if exists and not stale
            if os.path.exists(VIX_CACHE_PATH):
                try:
                    with open(VIX_CACHE_PATH, "r") as f:
                        cache = json.load(f)
                    if cache.get("date") == today and "vix" in cache:
                        vix_value = float(cache["vix"])
                        print(f"🟢 Loaded cached VIX: {vix_value:.2f} (from {today})")
                        if vix_value >= hard_limit:
                            print(f"🛑 VIX {vix_value:.2f} >= {hard_limit} - halting script for the day")
                            send_telegram(f"🛑 VIX {vix_value:.2f} >= {hard_limit} - halting script")
                            sys.exit(0)
                        elif vix_value >= threshold:
                            print(f"⚠️ VIX {vix_value:.2f} >= {threshold} - skipping trade")
                            send_telegram(f"⚠️ VIX {vix_value:.2f} >= {threshold} - skipping trade")
                            return False
                        else:
                            print(f"✅ VIX {vix_value:.2f} < {threshold} - proceeding")
                            return True
                except Exception as e:
                    print(f"⚠️ Failed to load VIX cache: {e} (refetching)")
        
            # If cache not found or outdated, fetch from NSE
            try:
                session = requests.Session()
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "*/*",
                    "Referer": "https://www.nseindia.com/market-data/vix",
                    "X-Requested-With": "XMLHttpRequest"
                }
                session.headers.update(headers)
                session.get("https://www.nseindia.com", timeout=10, verify=True)
                session.get("https://www.nseindia.com/market-data", timeout=10, verify=True)
                time.sleep(2)
                response = session.get("https://www.nseindia.com/api/allIndices", timeout=10)
                response.raise_for_status()
                data = response.json()
                vix_index_names = ["India VIX", "INDIA VIX", "VIX", "INDIA-VIX"]
                india_vix = None
                for idx in data["data"]:
                    if any(name.upper() in idx["index"].upper() for name in vix_index_names):
                        india_vix = idx
                        break
                if not india_vix:
                    available = ", ".join([d["index"] for d in data["data"][:5]])
                    raise ValueError(f"India VIX not found. Found: {available}...")
                vix_value = float(india_vix["last"])
        
                # Write to cache for today
                with open(VIX_CACHE_PATH, "w") as f:
                    json.dump({"date": today, "vix": vix_value}, f)
        
                print(f"🟢 VIX value {vix_value:.2f} saved to cache for {today}")
        
                if vix_value >= hard_limit:
                    print(f"🛑 VIX {vix_value:.2f} >= {hard_limit} - halting script for the day")
                    send_telegram(f"🛑 VIX {vix_value:.2f} >= {hard_limit} - halting script")
                    sys.exit(0)
                elif vix_value >= threshold:
                    print(f"⚠️ VIX {vix_value:.2f} >= {threshold} - skipping trade")
                    send_telegram(f"⚠️ VIX {vix_value:.2f} >= {threshold} - skipping trade")
                    return False
                else:
                    print(f"✅ VIX {vix_value:.2f} < {threshold} - proceeding")
                    return True
        
            except requests.exceptions.HTTPError as e:
                print(f"❌ HTTP error in VIX check: {e}")
                if e.response.status_code == 401:
                    print("💡 Tip: NSE blocked access. Try updating headers or adding delay.")
            except requests.exceptions.ConnectionError:
                print("❌ Network error: Could not connect to NSE")
            except Exception as e:
                print(f"❌ VIX check failed: {e}")
                traceback.print_exc()
        
            # Return True in case of failure IF in test_mode
            if 'test_mode' in globals() and test_mode:
                print("🧪 Test mode: Skipping VIX failure")
                return True
            else:
                return False
        

        # Apply VIX check
        if not is_vix_ok():
            time.sleep(60)
            continue

        try:
            # Load capital and stock list on each iteration
            capital = get_capital()
            print(f'💰 Capital loaded: ₹{capital:,.2f}')
            
            df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv")
            df["security_id"] = df["security_id"].astype(int).astype(str)
            print(f'📄 Loaded dynamic_stock_list.csv with {len(df)} entries')
            
            # Check market sentiment ONCE ONLY
            if 'sector_sentiment_map' not in globals():
                print("⏳ Fetching sector sentiment from proxy/Index method...")
                sector_sentiment_map = get_sector_sentiment_map(print_table=True)
            
            nifty_bullish = sector_sentiment_map.get("NIFTY 50", True)  # True if NIFTY 50 not found (fails open)
            
            unique_sectors = df['sector'].dropna().astype(str).str.strip().str.upper().unique()
            
            sector_status = {}
            for sector in unique_sectors:
                mapped_sector = sector_indices.get(sector.upper(), None)
                if not mapped_sector:
                    print(f'⚠️ Sector mapping not found for {sector}, skipping bullish check')
                    sector_status[sector] = False
                    continue
                   
                # Use proxy result (True if bullish, False otherwise)
                sector_bullish = sector_sentiment_map.get(mapped_sector, False)
                sector_status[mapped_sector] = sector_bullish
                status = "bullish" if sector_bullish else "bearish"
                print(f'📈 Sector {mapped_sector}: {status} (via Index_Check_Qwen)')                    
            
            if not nifty_bullish:
                print("📉 Overall market bearish - focusing on bullish sectors")
                send_telegram("⚠️ NIFTY bearish. Focusing on bullish sectors")
            
            # Stock evaluation loop
            candidates = []
        
            for index, row in df.iterrows():
                try:
                    if datetime.now() >= new_trade_end_time:
                        print("⏰ Time window expired - stopping evaluation")
                        break
                    
                    symbol = row["symbol"]
                    secid = row["security_id"]
                    sector = str(row.get("sector", "UNKNOWN")).strip()
                    if sector == "nan" or not sector:
                        sector = "UNKNOWN"
                    
                    # Small-cap filter - fail loudly if market_cap missing or blank
                    if 'market_cap' not in row or pd.isna(row['market_cap']):
                        raise KeyError(f"❌ Market cap data missing for {symbol} - required for small-cap trading")
                    if row['market_cap'] < 5000:  # Cr.
                        print(f'  ⚠️ Market cap too small ({row["market_cap"]} Cr) - skipping')
                        continue                   
                        
                    print(f'➡️ Evaluating {symbol} ({sector} sector)')
                    
                    # Enforce sector confirmation for small caps
                    mapped_sector = sector_indices.get(sector.strip().upper(), None)
                    
                    if not nifty_bullish:
                        if not mapped_sector:
                            print(f'  ⚠️ Sector not mapped for {symbol} ({sector}) - skipping')
                            continue
                        if not sector_status.get(mapped_sector, False):
                            if row.get('stock_origin', '') != 'Small Cap':  # Extra chance for small caps
                                print(f'  ❌ Sector bearish for {symbol} ({mapped_sector}) - skipping')
                                continue
                            print(f'  ⚠️ Allowing Small Cap {symbol} despite bearish sector')
                    # Fetch candles with rate limit control
                    try:
                        candles = fetch_candles(secid, count=75)  # Increased for chart patterns
                        time.sleep(1)
                        if not candles or len(candles) < 5:
                            print('⚠️ No candle data available, skipping...')
                            continue
                    except Exception as e:
                        if "Rate_Limit" in str(e):
                            print("⚠️ Rate limit hit, waiting 60 seconds...")
                            time.sleep(60)
                            continue
                        raise
        
                    # DYNAMIC TIME FILTER: SmallCaps get earlier entry window with strategic time-based exits
                    current_time = datetime.now().time()
                    
                    # SmallCaps get earlier entry (9:15) but must exit earlier (3:00 PM) to avoid volatility
                    if row.get('stock_origin', '') == 'Small Cap':
                        entry_time = dtime(9, 15)
                        # SmallCaps exit window closes earlier to avoid closing volatility
                        if current_time > dtime(14, 45):
                            print(f"⏰ SmallCap exit window closed (after 14:45) - skipping {symbol}")
                            continue
                    else:
                        entry_time = dtime(9, 30)
                    
                    if current_time < entry_time:
                        print(f"⏰ Too early for trading (before {entry_time.strftime('%H:%M')}) - skipping {symbol}")
                        continue
        
                    # Circuit filter (>5% move) -- FIXED: DhanHQ SDK-compliant ohlc_data
                    try:
                        quote_data = dhan.ohlc_data(securities={"NSE_EQ": [int(secid)]})
                        print(f"🪛 [DEBUG] ohlc_data response for {symbol} ({secid}): {quote_data}")  # DEBUG LINE
                        quote = None
                        
                        # FIX: Proper key path navigation for nested data blocks!
                        try:
                            # Step into nested data
                            nse_eq_block = (quote_data.get("data", {}) or {}).get("data", {}).get("NSE_EQ", {})
                            # Defensive handling for both str and int keys
                            if str(secid) in nse_eq_block:
                                quote = nse_eq_block[str(secid)]
                            elif int(secid) in nse_eq_block:
                                quote = nse_eq_block[int(secid)]
                            elif len(nse_eq_block) == 1:
                                quote = list(nse_eq_block.values())[0]
                                print(f"🪛 [DEBUG] Used fallback key for quote extraction: {list(nse_eq_block.keys())[0]}")
                        except Exception as e:
                            print(f"❌ Exception in quote extraction for {symbol}: {e}")
                        
                        if not quote:
                            print(f"❌ Could not fetch quote for {symbol} - skipping [DEBUG: NSE_EQ keys: {list((quote_data.get('data', {}) or {}).get('data', {}).get('NSE_EQ', {}).keys())}]")
                            continue

                        # Circuit filter (>5% move)
                        prev_close = float(quote.get('previousClose', 0))
                        # FIX: Dhan API uses 'last_price' not 'ltp'
                        ltp = float(quote.get('last_price', 0))  # <-- CORRECTED
                        if prev_close > 0:  # Prevent division by zero
                            current_move = abs(ltp - prev_close) / prev_close * 100
                            if current_move > 5:
                                print(f'⛔ Circuit filter: {symbol} moved {current_move:.2f}% (max 5%) - skipping')
                                continue
                        
                        # Gap-up filter (must pass the current LTP in quote)
                        if check_gap_up(secid):
                            print(f'⏫ Gap-up detected: {symbol}')
                            continue
                        
                        # Bid-ask spread
                        bid_ask_spread = abs(float(quote.get('bestBidPrice', 0)) - float(quote.get('bestAskPrice', 0)))
                        if ltp <= 0:
                            print(f"❌ Invalid LTP for {symbol} - skipping")
                            continue

                        spread_pct = bid_ask_spread / ltp * 100
                        if spread_pct > 1.0:  # 1% threshold
                            print(f'⚠️ High bid-ask spread ({spread_pct:.2f}%) for {symbol} - skipping')
                            continue

                        # Turnover check (₹ value)
                        recent_volumes = [c["volume"] for c in candles[-5:]] if len(candles) >= 5 else [candles[-1]["volume"]]
                        avg_recent_volume = sum(recent_volumes) / len(recent_volumes)
                        turnover = avg_recent_volume * ltp
                        
                        # 🕒 Dynamic volume thresholds based on stock origin and time
                        now = datetime.now().time()
                        stock_type = str(row.get('stock_origin', 'Small Cap')).strip()
                        
                        # Base thresholds by stock type (using stock_origin values)
                        volume_thresholds = {
                            'Large Cap': {'early': 8000, 'mid': 12000, 'late': 15000},
                            'Mid Cap': {'early': 5000, 'mid': 8000, 'late': 10000},
                            'Small Cap': {'early': 2000, 'mid': 3000, 'late': 4000},
                            'UNKNOWN': {'early': 3000, 'mid': 5000, 'late': 7000}  # Default for unexpected values
                        }
                        
                        # Get appropriate thresholds (default to Small Cap if unknown type)
                        thresholds = volume_thresholds.get(stock_type, volume_thresholds['Small Cap'])
                        
                        # Apply volatility scaling (higher thresholds in high volatility)
                        market_volatility = calculate_atr(candles[-30:]) if len(candles) >= 30 else 0
                        volatility_factor = 1.0
                        if ltp > 0 and market_volatility > (ltp * 0.015): # >1.5% ATR
                            volatility_factor = 1.3
                        
                        # Select threshold based on time
                        if now < dtime(10, 30):
                            min_volume = thresholds['early'] * volatility_factor
                        elif now < dtime(13, 30):
                            min_volume = thresholds['mid'] * volatility_factor
                        else:
                            min_volume = thresholds['late'] * volatility_factor
                        
                        # Dynamic turnover requirement (scaled by stock type)
                        min_turnover = {
                            'Large Cap': 1000000,
                            'Mid Cap': 750000,
                            'Small Cap': 500000
                        }.get(stock_type, 500000)  # Default to Small Cap threshold
                        
                        if avg_recent_volume < min_volume:
                            print(f'❌ Avg volume too low: {avg_recent_volume:.0f} < {min_volume:.0f} ({stock_type}) - skipping {symbol}')
                            continue
                        
                        if turnover < min_turnover:
                            print(f'❌ Turnover too low: ₹{turnover:,.2f} < ₹{min_turnover:,.0f} ({stock_type}) - skipping {symbol}')
                            continue
                        
                        
                    except KeyError as e:
                        print(f"❌ Missing quote data for {symbol}: {str(e)}")
                        continue
                    except Exception as e:
                        print(f"❌ Quote fetch or liquidity check failed for {symbol}: {e}")
                        continue

        
                    # Bullish pattern detection
                    detected, pattern_name, pattern_score = detect_bullish_pattern(candles, symbol)
                    
                    if not detected:
                        print('📉 No bullish pattern detected, skipping...')
                        continue
                    
                    # Enhanced bearish pattern conflict filter with pattern-specific checks
                    if detect_bearish_pattern(candles, pattern_name):
                        print(f'🚫 Conflicting bearish pattern detected near {pattern_name}: {symbol} - skipping...')
                        continue
                    
                    # Support zone confirmation (only for reversal patterns)
                    if pattern_name in REVERSAL_PATTERNS:
                        if not is_near_support(candles, buffer=0.015):  # 1.5% buffer
                            print(f'🚫 Not near support (within 1.5%): {symbol} - skipping...')
                            continue
                    
                    # Modified 15min trend check with reduced requirements
                    if len(candles) >= 75:  # Minimum 75 candles (5*15min)
                        if not get_15min_trend(candles, min_candles=5):
                            print(f'📉 15-minute trend not bullish for {symbol} - skipping...')
                            continue
                    else:
                        print(f'⚠️ Insufficient data for trend check on {symbol} - skipping...')
                        continue
                    
                    # Calculate composite score
                    weight = PATTERN_WEIGHTS.get(pattern_name, {"weight": 1.0})["weight"]
                    composite_score = weight * pattern_score
                    
                    # ========== VOLUME ANALYSIS (NEW) ==========
                    # Cache to avoid redundant API calls
                    volume_cache = {}
                    
                    def get_avg_volume(security_id, symbol):
                        """Get 30D avg volume with caching and fallback"""
                        if security_id in volume_cache:
                            return volume_cache[security_id]
                        
                        try:
                            # Fetch 45 days data (buffer for weekends/holidays)
                            hist_data = dhan.historical_daily_data(
                                security_id=str(security_id),
                                exchange_segment="NSE_EQ",
                                instrument_type="EQUITY",
                                from_date=(datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d"),
                                to_date=datetime.now().strftime("%Y-%m-%d")
                            )
                            
                            if hist_data and isinstance(hist_data, list):
                                df_hist = pd.DataFrame(hist_data)
                                avg_vol = df_hist['volume'].tail(30).mean()  # Last 30 trading days
                                volume_cache[security_id] = max(avg_vol, 100000)  # Floor at 1L shares
                            else:
                                volume_cache[security_id] = 100000  # Fallback minimum
                        except Exception as e:
                            print(f"⚠️ Volume data unavailable for {symbol}, using fallback: {e}")
                            volume_cache[security_id] = 100000
                        
                        return volume_cache[security_id]
                    
                    # Calculate volume metrics
                    current_volume = candles[-1]['volume'] if candles else 0
                    avg_30d_volume = get_avg_volume(secid, symbol)
                    
                    # Adaptive spike threshold (lower in high volatility)
                    market_volatility = calculate_atr(candles[-30:]) if len(candles) >= 30 else 0
                    spike_multiplier = 3.0 if market_volatility < (price * 0.01) else 2.2  # Dynamic adjustment
                    
                    # Volume spike logic
                    if current_volume > spike_multiplier * avg_30d_volume:
                        composite_score *= 1.5
                        print(f'🚀 Volume spike: {current_volume} (Avg30D: {avg_30d_volume:.0f}, {current_volume/avg_30d_volume:.1f}x)')
                    else:
                        print(f'📊 Volume: {current_volume} (Avg30D: {avg_30d_volume:.0f})')
                    # ========== END VOLUME ANALYSIS ==========
                    
                    print(f'📊 Pattern: {pattern_name}, Score: {pattern_score:.2f}, Composite: {composite_score:.2f}')
                    
                    # Technical indicators with RSI divergence check
                    closes = pd.Series([c["close"] for c in candles])
                    highs = pd.Series([c["high"] for c in candles])
                    rsi, macd_hist, macd_cross = compute_rsi_macd(closes)
                    
                    # ✅ FIXED: Robust RSI Divergence Detection to avoid float indexing
                    def has_rsi_divergence(highs, rsi, lookback=14):
                        if len(highs) < lookback or len(rsi) < lookback:
                            return False
                        
                        highs_lookback = highs.iloc[-lookback:]
                        rsi_lookback = rsi.iloc[-lookback:]
                        
                        if highs_lookback.empty or rsi_lookback.empty:
                            return False
                    
                        max_high_idx = highs_lookback.idxmax()
                        max_high = highs.loc[max_high_idx]
                        rsi_at_high = rsi.loc[max_high_idx]
                        
                        current_high = highs.iloc[-1]
                        current_rsi = rsi.iloc[-1]
                        
                        return current_high > max_high and current_rsi < rsi_at_high
                    
                    
                    rsi_divergence = has_rsi_divergence(highs, rsi) if len(closes) >= 14 else False
                    print(f'📊 RSI: {rsi:.2f}, MACD Hist: {macd_hist:.4f}, MACD Cross: {macd_cross}, Divergence: {rsi_divergence}')
                    
                    # Later in the validation...
                    if rsi_divergence:
                        print(f'🚫 Bearish RSI divergence detected for {symbol}, skipping...')
                        continue
                    
                    # Enhanced RSI validation with pattern awareness
                    if any(keyword in pattern_name for keyword in ["Reversal", "Bottom", "Round"]):
                        # Wider range for reversal patterns (30-80)
                        if not (30 <= rsi <= 80):
                            print(f'❌ RSI out of range for reversal pattern ({rsi:.2f}), skipping...')
                            continue
                        # Additional momentum confirmation for deep oversold
                        if rsi < 35:
                            # Require MACD confirmation for extreme RSI
                            if not (macd_hist > 0 or macd_cross):
                                print(f'⚠️ Deep oversold (RSI:{rsi:.2f}) without MACD confirmation, skipping...')
                                continue
                    else:
                        # Standard range for continuation patterns
                        if not (45 <= rsi <= 70):
                            print(f'❌ RSI out of range ({rsi:.2f}), skipping...')
                            continue
                    
                    # Enhanced MACD validation with pattern awareness
                    if "Bottom" in pattern_name or "Reversal" in pattern_name:
                        # For reversal patterns, accept positive histogram OR crossover
                        if not (macd_hist > 0 or macd_cross):
                            print('❌ MACD filter failed for reversal (need positive hist OR cross), skipping...')
                            continue
                    else:
                        # For continuation patterns, maintain strict crossover requirement
                        if not (macd_hist > 0 and macd_cross):
                            print('❌ MACD filter failed (need positive hist AND cross), skipping...')
                            continue
                            
                    # Pattern-specific validation rules
                    if pattern_name == "Triple Bottom":
                        # For Triple Bottom, require stronger volume confirmation
                        vol_ok, vol_ratio = volume_confirmed(index=-1, multiplier=1.8, pattern_type=pattern_name)
                        if not vol_ok:
                            print(f'❌ Triple Bottom volume insufficient ({vol_ratio:.2f}x < 1.8x), skipping...')
                            continue
                    
                    elif pattern_name == "Morning Star":
                        # For Morning Star, require closing above 50% of pattern range
                        star_low = min(l.iloc[-3], l.iloc[-2], l.iloc[-1])
                        star_high = max(h.iloc[-3], h.iloc[-2], h.iloc[-1])
                        if c.iloc[-1] < (star_low + (star_high - star_low) * 0.5):
                            print('❌ Morning Star close below 50% of pattern range, skipping...')
                            continue
                    
                    # ======== RESISTANCE CHECK ========
                    # Calculate resistance (max of last 20 candles)
                    resistance_period = 20
                    if len(candles) >= resistance_period:
                        resistance_level = max(candle['high'] for candle in candles[-resistance_period:])
                    else:
                        resistance_level = max(candle['high'] for candle in candles)
                    
                    # Store breakout level if pattern is breakout type
                    breakout_level = None
                    if pattern_name.upper() in {p.upper() for p in BREAKOUT_PATTERNS}:
                        breakout_level = resistance_level
                    
                    # Skip if within 1.2% of resistance without breakout
                    RESISTANCE_BUFFER = 0.012
                    current_price = candles[-1]['close']
                    if (pattern_name not in BREAKOUT_PATTERNS and 
                        current_price >= resistance_level * (1 - RESISTANCE_BUFFER)):
                        print(f'🚫 Within 1.2% of resistance ({resistance_level:.2f}) without breakout pattern')
                        continue
                    
                    # For breakout patterns, require 1.5% clearance
                    if (pattern_name in BREAKOUT_PATTERNS and 
                        current_price < resistance_level * 1.015):
                        print(f'🚫 Breakout pattern requires 1.5% clearance above resistance ({resistance_level*1.015:.2f})')
                        continue
                    # ======== END RESISTANCE CHECK ========
                    
                    # Position sizing with resistance discount factor
                    price = current_price
                    if price <= 0:
                        print(f'⚠️ Invalid price for {symbol}: {price}, skipping...')
                        continue
                    
                    # Apply resistance discount to position size
                    resistance_distance = 1 - (price / resistance_level)
                    position_discount = max(0.3, min(1.0, resistance_distance * 2))  # Scale 0.5% distance -> 100% allocation
                   
                    # For breakout patterns, require 1% clearance above resistance
                    if pattern_name.upper() in {p.upper() for p in BREAKOUT_PATTERNS}:
                        if price < resistance_level * 1.01:
                            print(f'🚫 Breakout pattern requires 1% clearance above resistance ({resistance_level:.2f})')
                            continue
                    
                    # Calculate max quantity with volatility adjustment
                    atr = calculate_atr(candles)
                    risk_per_share = 1.5 * atr  # Risk = 1.5x ATR
                    max_risk_per_trade = capital * 0.01  # 1% of capital
                    max_shares = max_risk_per_trade / risk_per_share
                    
                    # Apply resistance discount
                    max_investment = min(capital * position_discount, max_shares * price)
                    base_qty = int(max_investment // price)
                    if base_qty < 1:
                        raise ValueError(
                            f"❌ Unaffordable position: {symbol} price ₹{price:.2f} "
                            f"exceeds allocation ₹{max_investment:.2f}"
                        )
                 
                    print(f'📉 Resistance discount: {position_discount*100:.1f}% | Allocation: ₹{max_investment:.2f}')
                    
                    if base_qty > 0:
                        # Apply pattern confidence weighting
                        adj_qty = max(1, int(base_qty * weight * pattern_score))
                        investment_value = price * adj_qty
                        print(f'💸 Final Price: ₹{price:.2f}, Base Qty: {base_qty}, Adj Qty: {adj_qty}, Investment: ₹{investment_value:.2f}')
                    else:
                        print(f'⚠️ {symbol} price ₹{price:.2f} exceeds allocated capital, skipping')
                        continue
        
                    # Add to ranked candidates
                    candidates.append({
                        "symbol": symbol,
                        "security_id": secid,
                        "qty": adj_qty,
                        "price": price,
                        "pattern": pattern_name,
                        "candles": candles,
                        "score": pattern_score,
                        "confidence": weight,
                        "composite_score": composite_score,
                        "breakout_level": breakout_level 
                    })
        
                except Exception as e:
                    print(f'⚠️ {row.get("symbol", "UNKNOWN")} evaluation failed: {str(e)}')
                    continue
        
            # Pick best from ranked candidates if any
            if candidates:
                # Rank by composite score
                best = sorted(candidates, key=lambda x: x["composite_score"], reverse=True)[0]
                print(f"🚀 Best pick: {best['symbol']} with Qty: {best['qty']} and Composite Score: {best['composite_score']:.2f}")
                
                place_order(
                    best["symbol"], 
                    best["security_id"], 
                    best["qty"], 
                    best["price"], 
                    best["pattern"], 
                    best["candles"], 
                    tick_size_map, 
                    capital,
                    breakout_level=best["breakout_level"]
                )
                
                # Start breakout monitor if applicable
                if best["pattern"] in BREAKOUT_PATTERNS and best["breakout_level"] is not None:
                    monitor_breakout(
                        best["security_id"],# Time feasibility check
                        best["breakout_level"],
                        best["symbol"],
                        best["qty"]
                    )
                
                send_telegram(f"✅ Order placed for {best['symbol']} with Qty: {best['qty']}.")
                print("✅ Order placed. Exiting script.")
                reason_msg = f"{best['pattern']} formed. {best['symbol']} ({best['qty']} qty) selected"
                log_time = datetime.now()
                order_placed = True  # Break loop after successful placement
                
                try:
                    # CSV logging
                    with open("D:/Downloads/Dhanbot/dhan_autotrader/bot_execution_log.csv", "a") as flog:
                        flog.write(f"{log_time.strftime('%Y-%m-%d %H:%M:%S')},autotrade.py,SUCCESS,\"{reason_msg}\"\n")
                
                    # DB logging with timestamp
                    log_to_postgres(
                        timestamp=log_time,
                        script="autotrade.py",
                        status="BUY",
                        message=reason_msg
                    )
                except Exception as e:
                    print(f"⚠️ Failed to log bot_execution reason: {e}")
            else:
                print("❌ No valid trades found this iteration. Retrying in 2 minutes...")
                send_telegram("🔄 No valid trades found. Rescanning in 2 minutes...")
                time.sleep(120)  # Wait 60 sec before next scan
                
        except Exception as e:
            print(f"⚠️ Main loop error: {e}")
            traceback.print_exc()
            time.sleep(10)  # Wait 10 Sec after error before retrying

# Save execution log
try:
    log_path = "D:/Downloads/Dhanbot/dhan_autotrader/Logs/New_Autotrade_Entry.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(log_buffer.getvalue())
except Exception as e:
    print(f"⚠️ Failed to write log file: {e}")

if __name__ == "__main__":
    main()