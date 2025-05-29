import csv
import requests
import json
from datetime import datetime, timedelta
import pytz
import pandas as pd
import functools
import time
from utils_safety import retry
import datetime as dt



# ✅ Load Dhan credentials from config
with open("D:/Downloads/Dhanbot/dhan_autotrader/config.json") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

session = requests.Session()
session.headers.update({
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
})


dhan_master_df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv")

def retry(max_attempts=3, delay=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

def request_with_retry(method, url, headers=None, json=None, max_retries=5):
    import time
    for attempt in range(max_retries):
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=json)
        else:
            raise ValueError("Unsupported method")

        if response.status_code == 429:
            print(f"⚠️ Rate limit hit (429). Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2)
            continue
        return response

    print("❌ Max retry attempts reached. Returning last response.")
    return response
    
def get_intraday_candles(security_id, interval="1", from_dt=None, to_dt=None):
    import requests
    import pytz
    from datetime import datetime
    from dhan_config import HEADERS  # ✅ you must have a valid token and client ID

    india = pytz.timezone("Asia/Kolkata")
    now = datetime.now(india)

    # Default: today 9:15 to now
    if not from_dt:
        from_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if not to_dt:
        to_dt = now

    payload = {
        "securityId": str(security_id),              # ✅ force to string
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": str(interval),                   # ✅ must be "1", "5", "15" etc.
        "oi": "false",
        "fromDate": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": to_dt.strftime("%Y-%m-%d %H:%M:%S")
    }

    url = "https://api.dhan.co/v2/charts/intraday"
    try:
        response = session.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception during candle fetch: {e}")
        return None

# ✅ Get security ID from master CSV
def get_security_id(symbol):
    try:
        df = pd.read_csv("dhan_master.csv")
        row = df[df["SEM_TRADING_SYMBOL"].str.upper() == symbol.upper()]
        if not row.empty:
            return str(row.iloc[0]["SEM_SMST_SECURITY_ID"])
        else:
            print(f"⚠️ {symbol} not found in dhan_master.csv")
    except Exception as e:
        print(f"⚠️ Error fetching security_id for {symbol}: {e}")
    return None

def get_security_id_from_trading_symbol(symbol):
    try:
        df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv")
        for _, row in df.iterrows():
            # Match by SEM_TRADING_SYMBOL (Dhan's trading symbol)
            if str(row["SEM_TRADING_SYMBOL"]).strip().upper() == symbol.strip().upper():
                return str(row["SEM_SMST_SECURITY_ID"]).strip()
        print(f"⛔ Symbol not found in dhan_master.csv: {symbol}")
    except Exception as e:
        print(f"❌ Error in get_security_id_from_trading_symbol(): {e}")
    return None

def get_current_capital():
    try:
        path = "D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv"
        df = pd.read_csv(path, header=None)

        capital = float(df.iloc[0, 0])
        if capital <= 0:
            raise ValueError("Capital must be greater than 0")

        return capital

    except Exception as e:
        print(f"❌ Error reading current_capital.csv: {e}")
        raise SystemExit("🛑 Halting: current_capital.csv must contain a valid capital value in A1")

# ✅ Fetch live price from Dhan API (last traded price)

def compute_rsi(security_id, period=14):
    try:
        candles = get_historical_price(security_id, interval="15m", limit=period + 1)
        closes = [c["close"] for c in candles if "close" in c]
        if len(closes) < period + 1:
            return 50  # Neutral fallback if insufficient data

        deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]

        avg_gain = sum(gains) / period if gains else 0.01
        avg_loss = sum(losses) / period if losses else 0.01

        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    except Exception as e:
        print(f"⚠️ RSI computation failed for {security_id}: {e}")
        return 50  # Neutral fallback


@retry(max_attempts=3, delay=2)
def get_live_price(symbol, security_id, premarket=False):
    now = datetime.now()
    if premarket or now.hour < 9 or now.hour >= 16:
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
        "oi": False,
        "fromDate": from_time.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": to_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        resp = session.post("https://api.dhan.co/v2/charts/intraday", json=payload)
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
        
# ✅ Fetch historical candles (5m, 15m, or 1d)
def get_historical_price(security_id, interval="5", limit=20, from_date=None, to_date=None):
    if not security_id or not str(security_id).strip().isdigit():
        raise ValueError(f"Invalid security_id passed to get_historical_price: {security_id}")
    try:
        india = pytz.timezone("Asia/Kolkata")
        now = datetime.now(india)

        # Default to today's 9:15 to now if not provided
        if from_date is None:
            from_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")

        if to_date is None:
            to_dt = now
        else:
            to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")

        url = "https://api.dhan.co/v2/charts/intraday"
        headers = {
            "access-token": config["access_token"],
            "client-id": config["client_id"],
            "Content-Type": "application/json"
        }

        payload = {
            "securityId": str(security_id),
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": str(interval),
            "oi": "false",
            "fromDate": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": to_dt.strftime("%Y-%m-%d %H:%M:%S")
        }

        res = session.post(url, json=payload)
        if res.status_code != 200:
            print(f"❌ Error: {res.status_code} - {res.text}")
            return []

        response_json = res.json()
        if "open" not in response_json:
            print("⚠️ No 'open' key in response. Returning empty list.")
            return []

        df = pd.DataFrame({
            "open": response_json["open"],
            "high": response_json["high"],
            "low": response_json["low"],
            "close": response_json["close"],
            "volume": response_json["volume"],
            "timestamp": pd.to_datetime(response_json["timestamp"], unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata')
        })

        if limit:
            df = df.tail(limit)

        return df.to_dict(orient="records")

    except Exception as e:
        print(f"⚠️ Fallback to test candles due to: {e}")
        now = datetime.now()
        test_data = []
        for i in range(limit):
            test_data.append({
                "timestamp": (now - timedelta(minutes=5 * i)).strftime('%Y-%m-%d %H:%M:%S'),
                "open": 100 + i,
                "high": 102 + i,
                "low": 99 + i,
                "close": 101 + i,
                "volume": 100000 + i * 100
            })
        return list(reversed(test_data))

# ✅ CNC Market Order Placement (BUY or SELL)
def place_order(security_id, quantity, transaction_type="BUY"):
    url = "https://api.dhan.co/orders"
    headers = {
        "access-token": ACCESS_TOKEN,
        "Content-Type": "application/json"
    }

    payload = {
        "transactionType": transaction_type,  # BUY or SELL
        "exchangeSegment": "NSE_EQ",
        "productType": "CNC",
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": security_id,
        "quantity": quantity,
        "price": 0,
        "orderValue": 0,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
        "amoTime": "OPEN",
        "triggerPrice": 0,
        "smartOrder": False
    }

    try:
        response = session.post(url, json=payload)
        json_resp = response.json()
        if response.status_code == 200:
            print(f"✅ Order {transaction_type} placed successfully for ID: {security_id}")
        else:
            print(f"❌ Order failed: {json_resp}")
        return response.status_code, json_resp
    except Exception as e:
        print(f"⚠️ Order placement failed: {e}")
        return 500, {"error": str(e)}