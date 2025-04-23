import requests
import json
import csv
from datetime import datetime, time
from dhan_api import get_live_price

# ✅ Load config (token & client ID)
with open("dhan_config.json") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
BASE_URL = "https://api.dhan.co/orders"

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# ✅ Get real-time capital from Dhan
def get_available_margin():
    margin_url = "https://api.dhan.co/reports/daily-funds"
    try:
        response = requests.get(margin_url, headers=HEADERS)
        data = response.json()
        return float(data["availableMargin"])
    except Exception as e:
        print(f"⚠️ Unable to fetch live margin: {e}")
        return 3700  # fallback if API fails

# ✅ Verified PSU stock list with correct Dhan security IDs
STOCK_LIST = [
    {"symbol": "NHPC", "security_id": "3505"},
    {"symbol": "IRFC", "security_id": "15919"},
    {"symbol": "NLCINDIA", "security_id": "10158"},
    {"symbol": "BEL", "security_id": "11740"},
    {"symbol": "BHEL", "security_id": "1343"}
]

def log_live_prices(price_data):
    filename = "live_prices_log.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for entry in price_data:
            writer.writerow([now, entry["symbol"], entry["price"]])

def is_market_open():
    now = datetime.now().time()
    return time(9, 15) <= now <= time(15, 30)

def run_autotrade():
    print("🚀 Starting AutoTrader (Live Price Mode)")

    if not is_market_open():
        print("⏰ Market is currently closed. Exiting without trading.")
        return

    CAPITAL = get_available_margin()
    price_log = []

    for stock in STOCK_LIST:
        try:
            live_price = get_live_price(stock["symbol"])
            price_log.append({"symbol": stock["symbol"], "price": live_price})
            qty = int(CAPITAL // live_price)
            if qty >= 1:
                payload = {
                    "transactionType": "BUY",
                    "exchangeSegment": "NSE_EQ",
                    "productType": "CNC",
                    "orderType": "MARKET",
                    "validity": "DAY",
                    "securityId": stock["security_id"],
                    "tradingSymbol": stock["symbol"],
                    "quantity": qty,
                    "price": 0,
                    "disclosedQuantity": 0,
                    "afterMarketOrder": False,
                    "amoTime": "OPEN",
                    "triggerPrice": 0,
                    "smartOrder": False
                }

                print(f"\n📦 Placing order: {stock['symbol']} — {qty} shares @ ₹{live_price}")
                response = requests.post(BASE_URL, headers=HEADERS, json=payload)
                if response.status_code == 200:
                    print(f"✅ {stock['symbol']} order placed successfully!")
                    print("Response:", response.json())
                    break
                else:
                    print(f"❌ Failed: {response.status_code} | {response.text}")
            else:
                print(f"⚠️ Not enough funds to buy even 1 share of {stock['symbol']} @ ₹{live_price}")

        except Exception as e:
            print(f"❗ Error placing order for {stock['symbol']}: {e}")

    log_live_prices(price_log)
    print("📊 Live prices saved to live_prices_log.csv")
    input("\n✅ Script finished. Press Enter to exit...")

if __name__ == "__main__":
    run_autotrade()
