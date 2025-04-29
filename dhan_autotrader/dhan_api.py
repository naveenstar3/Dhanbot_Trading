import requests
import yfinance as yf
from datetime import datetime, timedelta

# ✅ Order Placement Function
def place_order(access_token, security_id, quantity, transaction_type="BUY"):
    url = "https://api.dhan.co/orders"
    headers = {
        "access-token": access_token,
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

    response = requests.post(url, headers=headers, json=payload)

    try:
        json_resp = response.json()
        if response.status_code == 200:
            print(f"✅ Order {transaction_type} placed successfully.")
        else:
            print(f"❌ Order failed: {json_resp}")
        return response.status_code, json_resp
    except:
        print("⚠️ Failed to parse response")
        return response.status_code, {}
        
# ✅ Updated Live Price Fetcher (Yahoo Finance)
def get_live_price(symbol):
    try:
        stock = yf.Ticker(symbol + ".NS")
        data = stock.history(period="1d", interval="1m")
        if data.empty or data["Close"].isnull().all():
            raise Exception("No data received or Close price missing.")
        return float(round(data["Close"].dropna().iloc[-1], 2))
    except Exception as e:
        print(f"⚠️ Error fetching price for {symbol}: {e}")
        return None

# ✅ Historical price fetcher for intraday momentum check
def get_historical_price(symbol, minutes_ago=5):
    try:
        import pytz
        end = datetime.now(pytz.utc)
        start = end - timedelta(minutes=minutes_ago + 1)

        data = yf.download(
        tickers=f"{symbol}.NS",
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=True
        )


        if data.empty or "Close" not in data.columns:
            raise Exception("No data available")

        # Make sure index is timezone-aware and in UTC
        data.index = data.index.tz_convert("UTC")

        # Filter rows before 'start' time
        filtered = data[data.index <= start]

        if filtered.empty:
            raise Exception("No historical data available")

        # ✅ FIX HERE: no .iloc[0] needed after selecting last non-null Close
        close_price = filtered["Close"].dropna().iloc[-1]
        return round(float(close_price.iloc[0]), 2)

    except Exception as e:
        print(f"⚠️ Error in get_historical_price({symbol}): {e}")
        return 0

