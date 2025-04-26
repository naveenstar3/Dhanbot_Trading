import csv
import json
import requests
from datetime import datetime, time
from dhan_api import get_live_price

PORTFOLIO_LOG = "portfolio_log.csv"
GROWTH_LOG = "growth_log.csv"
CURRENT_CAPITAL_FILE = "current_capital.csv"

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

# ✅ Load capital from file or user input
def get_available_capital():
    try:
        with open(CURRENT_CAPITAL_FILE, "r") as f:
            capital = float(f.read().strip())
            print(f"💰 Capital being used for trade: ₹{capital}")
            return capital
    except:
        capital = float(input("Enter your starting capital: "))
        with open(CURRENT_CAPITAL_FILE, "w") as f:
            f.write(str(capital))
        return capital

# ✅ Fetch PSU stocks with lotSize 1 and in affordable range
def get_psu_stocks(capital):
    PSU_SYMBOLS = ["NHPC", "IRFC", "BEL", "BHEL", "NLCINDIA"]
    psu_stocks = []
    try:
        response = requests.get("https://images.dhan.co/api-data/api-scrip-master.csv")
        lines = response.text.splitlines()
        reader = csv.DictReader(lines)
        for row in reader:
            try:
                if row["SEM_TRADING_SYMBOL"] not in PSU_SYMBOLS:
                    continue
                symbol = row["SEM_TRADING_SYMBOL"]
                security_id = row["SEM_SMST_SECURITY_ID"]
                lot_size = int(float(row["SEM_LOT_UNITS"]))
                price = get_live_price(symbol)
                required_cost = lot_size * price

                if capital >= required_cost:
                    psu_stocks.append({
                        "symbol": symbol,
                        "security_id": security_id,
                        "lot_size": lot_size,
                        "price": price
                    })
            except Exception as e:
                print(f"❗ Skipping {row.get('SEM_TRADING_SYMBOL', 'UNKNOWN')}: {e}")
    except Exception as e:
        print(f"⚠️ Error fetching PSU stocks: {e}")
    return psu_stocks

def run_autotrade():
    print("🚀 Starting AutoTrader (Capital File Mode)")
    now = datetime.now().time()
    if not time(9, 15) <= now <= time(15, 30):
        print("⏰ Market is currently closed. Exiting without trading.")
        return

    capital = get_available_capital()
    stock_list = get_psu_stocks(capital)
    price_log = []

    for stock in stock_list:
        try:
            price_log.append({"symbol": stock["symbol"], "price": stock["price"]})
            required_cost = stock["lot_size"] * stock["price"]
            if capital < required_cost:
                print(f"⛔ Skipping {stock['symbol']} — requires ₹{round(required_cost, 2)} but capital is ₹{capital}")
                continue

            qty = (int(capital // stock["price"]) // stock["lot_size"]) * stock["lot_size"]
            if qty >= stock["lot_size"]:
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

                print(f"\n📦 Placing order: {stock['symbol']} — {qty} shares @ ₹{stock['price']}")
                response = requests.post(BASE_URL, headers=HEADERS, json=payload)
                if response.status_code == 200:
                    print(f"✅ {stock['symbol']} order placed successfully!")
                    with open("portfolio_log.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        stock["symbol"],
                        stock["security_id"],
                        qty,
                        stock["price"],
                        round(stock["price"] * 1.015, 2),  # Target price (1.5% above)
                        round(stock["price"] * 0.99, 2),   # Stop loss price (1% below)
                        "", "", "", "HOLD"
                    ])
                    break
                else:
                    print(f"❌ Failed: {response.status_code} | {response.text}")
            else:
                print(f"⚠️ Not enough capital for lot size of {stock['symbol']} (Lot size: {stock['lot_size']})")

        except Exception as e:
            print(f"❗ Error placing order for {stock['symbol']}: {e}")

    # Optional: log to CSV
    with open("live_prices_log.csv", mode='a', newline='') as f:
        writer = csv.writer(f)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for entry in price_log:
            writer.writerow([now, entry["symbol"], entry["price"]])

    print("\n✅ Script finished. Exiting...")

if __name__ == "__main__":
    run_autotrade()
