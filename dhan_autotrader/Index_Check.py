import pandas as pd
from dhanhq import DhanContext, dhanhq
from datetime import datetime, timedelta
import json
import time
import os

CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
MASTER_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"

# Load configuration and initialize DHAN API
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
context = DhanContext(config["client_id"], config["access_token"])
dhan = dhanhq(context)
master_df = pd.read_csv(MASTER_CSV)

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
        {"name": "TATA MOTORS LTD", "security_id": 3435},
        {"name": "MAHINDRA & MAHINDRA LTD", "security_id": 2031}
    ],
    "NIFTY IT": [
        {"name": "INFOSYS LIMITED", "security_id": 1594},
        {"name": "TATA CONSULTANCY SERVICES LTD", "security_id": 3413},
        {"name": "WIPRO LIMITED", "security_id": 1348}
    ],
    "NIFTY FMCG": [
        {"name": "HINDUSTAN UNILEVER LTD.", "security_id": 1394},
        {"name": "ITC LIMITED", "security_id": 1660},
        {"name": "NESTLE INDIA LIMITED", "security_id": 4595}
    ],
    "NIFTY FIN SERVICE": [
        {"name": "BAJAJ FINANCE LIMITED", "security_id": 317},
        {"name": "HDFC BANK LTD", "security_id": 1333},
        {"name": "HDFC LIFE INSURANCE COMPANY LTD", "security_id": 7929}
    ],
    "NIFTY PHARMA": [
        {"name": "SUN PHARMACEUTICAL IND L", "security_id": 3351},
        {"name": "DR. REDDYS LABORATORIES LTD", "security_id": 910},
        {"name": "CIPLA LTD", "security_id": 599}
    ],
    "NIFTY REALTY": [
        {"name": "DLF LIMITED", "security_id": 14732},
        {"name": "GODREJ PROPERTIES LTD", "security_id": 2607},
        {"name": "OBEROI REALTY LTD", "security_id": 11184}
    ],
    "NIFTY ENERGY": [
        {"name": "RELIANCE INDUSTRIES LTD", "security_id": 2885},
        {"name": "NTPC LIMITED", "security_id": 11630},
        {"name": "POWER GRID CORPORATION OF INDIA LTD", "security_id": 383}
    ],
    "NIFTY METAL": [
        {"name": "TATA STEEL LIMITED", "security_id": 3499},
        {"name": "HINDALCO INDUSTRIES LTD", "security_id": 1363},
        {"name": "VEDANTA LIMITED", "security_id": 3063}
    ],
    "NIFTY OIL & GAS": [
        {"name": "RELIANCE INDUSTRIES LTD", "security_id": 2885},
        {"name": "ONGC", "security_id": 1181},
        {"name": "GAIL (INDIA) LIMITED", "security_id": 1201}
    ],
    "NIFTY CONSUMER DURABLES": [
        {"name": "TITAN COMPANY LIMITED", "security_id": 3506},
        {"name": "VOLTAS LIMITED", "security_id": 1324},
        {"name": "BLUE STAR LIMITED", "security_id": 505}
    ],
    "NIFTY HEALTHCARE": [
        {"name": "DIVI S LABORATORIES LTD", "security_id": 10940},
        {"name": "APOLLO HOSPITALS ENTERPRISE LTD", "security_id": 157},
        {"name": "FORTIS HEALTHCARE LIMITED", "security_id": 1053}
    ],
    "NIFTY INFRA": [
        {"name": "LARSEN & TOUBRO LTD.", "security_id": 11483},
        {"name": "ADANI PORTS AND SPECIAL ECONOMIC ZONE LTD", "security_id": 15083},
        {"name": "ULTRATECH CEMENT LIMITED", "security_id": 11536}
    ],
    "NIFTY MEDIA": [
        {"name": "SUN TV NETWORK LIMITED", "security_id": 13404},
        {"name": "ZEE ENTERTAINMENT ENTERPRISES LTD", "security_id": 1385},
        {"name": "PVR INOX LTD", "security_id": 11827}
    ],
    "NIFTY PSU BANK": [
        {"name": "STATE BANK OF INDIA", "security_id": 3045},
        {"name": "PUNJAB NATIONAL BANK", "security_id": 257},
        {"name": "BANK OF BARODA", "security_id": 4747}
    ],
    "NIFTY PRIVATE BANK": [
        {"name": "ICICI BANK LTD.", "security_id": 4963},
        {"name": "AXIS BANK LTD", "security_id": 5900},
        {"name": "KOTAK MAHINDRA BANK LTD", "security_id": 1922}
    ],
    "NIFTY SERVICES SECTOR": [
        {"name": "CONTAINER CORPORATION OF INDIA LTD", "security_id": 1217},
        {"name": "INDIABULLS HOUSING FINANCE LTD", "security_id": 10217},
        {"name": "ADANI ENTERPRISES LTD", "security_id": 25}
    ],
    "NIFTY COMMODITIES": [
        {"name": "GRASIM INDUSTRIES LTD", "security_id": 1232},
        {"name": "JSW STEEL LIMITED", "security_id": 11723},
        {"name": "ADANI ENTERPRISES LTD", "security_id": 25}
    ]
}

def fetch_candles(security_id, symbol_name, is_index=False):
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
    candles = fetch_candles(security_id, sector, is_index=True)
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
        candles = fetch_candles(proxy['security_id'], proxy['name'])
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
print("\n" + "="*50)
print(f"SECTOR ANALYSIS REPORT - {datetime.now().strftime('%d %b %Y %H:%M')}")
print("="*50 + "\n")

for sector in INDEX_SECURITY_MAP.keys():
    print(f"\n🔍 Analyzing {sector}")
    
    # Try index first
    index_id = INDEX_SECURITY_MAP[sector]
    result = analyze_index(sector, index_id)
    
    # Fallback to proxies if index analysis fails
    if result is None:
        print(f"🔄 Falling back to proxy analysis for {sector}")
        proxies = MULTI_PROXY_MAP.get(sector, [])
        result = analyze_proxies(sector, proxies)
    
    # Final determination
    if result is None:
        print(f"⛔ Unable to determine trend for {sector}")
    else:
        status = "BULLISH ✅" if result else "BEARISH ❌"
        print(f"📌 {sector} Trend: {status}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)