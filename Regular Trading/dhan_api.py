import requests
import yfinance as yf

# ✅ Order Placement Function
def place_order(access_token, security_id, quantity):
    url = "https://api.dhan.co/orders"
    headers = {
        "access-token": access_token,
        "Content-Type": "application/json"
    }

    payload = {
        "transactionType": "BUY",
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
    return response.status_code, response.json()

# ✅ Updated Live Price Fetcher (Yahoo Finance)
def get_live_price(symbol):
    try:
        stock = yf.Ticker(symbol + ".NS")
        data = stock.history(period="1d", interval="1m")
        if data.empty or data["Close"].isnull().all():
            raise Exception("No data received or Close price missing.")
        return round(data["Close"].dropna().iloc[-1], 2)
    except Exception as e:
        print(f"⚠️ Error fetching price for {symbol}: {e}")
        return None  # ← Return None instead of dummy price


