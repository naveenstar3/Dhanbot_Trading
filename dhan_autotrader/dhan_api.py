import csv
import requests
import json
from datetime import datetime
import pytz
import pandas as pd

# ✅ Load Dhan credentials from config
with open("D:/Downloads/Dhanbot/dhan_autotrader/dhan_config.json") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

dhan_master_df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv")

# ✅ Get security ID from master CSV
def get_security_id(symbol):
    try:
        df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv")
        for _, row in df.iterrows():
            sm_symbol = str(row.get("SM_SYMBOL_NAME", "")).strip().upper()
            trading_symbol = str(row.get("SEM_TRADING_SYMBOL", "")).strip().upper()
            if symbol.strip().upper() in [sm_symbol, trading_symbol]:
                return str(row["SEM_SMST_SECURITY_ID"]).strip()
        print(f"⛔ Symbol not found in dhan_master.csv: {symbol}")
    except Exception as e:
        print(f"❌ Error in get_security_id(): {e}")
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
        df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv")
        if "capital" in df.columns and not df.empty:
            return float(df["capital"].iloc[0])
        else:
            print("⚠️ current_capital.csv missing 'capital' column or empty.")
            return 0
    except Exception as e:
        print(f"⚠️ Failed to load capital: {e}")
        return 0

# ✅ Fetch live price from Dhan API (last traded price)
def get_live_price(symbol):
    try:
        security_id = get_security_id(symbol)
        if not security_id:
            raise Exception("Security ID not found")

        url = f"https://api.dhan.co/market-feed/quotes/{security_id}?exchangeSegment=NSE_EQ"
        headers = {
            "access-token": ACCESS_TOKEN,
            "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)
        data = response.json().get("data", {})
        return float(data.get("lastTradedPrice", 0)) / 100

    except Exception as e:
        print(f"⚠️ Live price fetch error for {symbol}: {e}")
        return 0

# ✅ Fetch historical candles (5m, 15m, or 1d)
def get_historical_price(security_id, interval="5m", limit=15):
    try:
        url = f"https://api.dhan.co/chart/intraday/{security_id}?exchangeSegment=NSE_EQ&instrumentId={security_id}&interval={interval}&limit={limit}"
        headers = {
            "access-token": ACCESS_TOKEN,
            "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)
        raw_candles = response.json().get("data", [])
        formatted = []

        for row in raw_candles:
            formatted.append({
                "datetime": row["startTime"],
                "open": float(row["openPrice"]) / 100,
                "high": float(row["highPrice"]) / 100,
                "low": float(row["lowPrice"]) / 100,
                "close": float(row["closePrice"]) / 100,
                "volume": int(row["volume"])
            })

        return formatted

    except Exception as e:
        print(f"⚠️ Historical price error for {security_id}: {e}")
        return []

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
        response = requests.post(url, headers=headers, json=payload)
        json_resp = response.json()
        if response.status_code == 200:
            print(f"✅ Order {transaction_type} placed successfully for ID: {security_id}")
        else:
            print(f"❌ Order failed: {json_resp}")
        return response.status_code, json_resp
    except Exception as e:
        print(f"⚠️ Order placement failed: {e}")
        return 500, {"error": str(e)}
