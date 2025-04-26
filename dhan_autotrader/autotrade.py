import csv
import json
import requests
from datetime import datetime, time, timedelta
from dhan_api import get_live_price, get_historical_price
from psu_stock_loader import fetch_psu_rows_from_master

PORTFOLIO_LOG = "portfolio_log.csv"
LIVE_LOG = "live_prices_log.csv"
CURRENT_CAPITAL_FILE = "current_capital.csv"
GROWTH_LOG = "growth_log.csv"
BASE_URL = "https://api.dhan.co/orders"

# ✅ Load Dhan credentials
with open("dhan_config.json") as f:
    config = json.load(f)

HEADERS = {
    "access-token": config["access_token"],
    "client-id": config["client_id"],
    "Content-Type": "application/json"
}

# ✅ Capital calculation with reinvestment logic
def get_available_capital():
    try:
        with open(CURRENT_CAPITAL_FILE, "r") as f:
            base_capital = float(f.read().strip())
    except:
        base_capital = float(input("Enter your starting capital: "))
        with open(CURRENT_CAPITAL_FILE, "w") as f:
            f.write(str(base_capital))

    try:
        with open(GROWTH_LOG, newline="") as f:
            rows = list(csv.DictReader(f))
            if rows:
                last_growth = float(rows[-1].get("profits_realized", 0))
                if last_growth >= 5:
                    base_capital += last_growth
                    print(f"🔼 Reinvested ₹{last_growth} profit into capital")
                else:
                    print(f"⏹️ Holding capital, last profit ₹{last_growth} < ₹5")
    except:
        pass

    print(f"💰 Capital for today: ₹{round(base_capital, 2)}")
    return base_capital

# ✅ Avoid trade if unsold position exists
def has_open_position():
    today = datetime.now().date()
    try:
        with open(PORTFOLIO_LOG, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status", "").upper() != "SOLD":
                    ts_str = row.get("timestamp", "")
                    try:
                        entry_date = datetime.strptime(ts_str, "%m/%d/%Y %H:%M").date()
                        if entry_date == today:
                            return True
                    except:
                        continue
    except FileNotFoundError:
        return False
    return False

# ✅ Market hours check
def is_market_open():
    now = datetime.now().time()
    return time(9, 15) <= now <= time(15, 30)

# ✅ Save price snapshot
def log_live_prices(price_data):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LIVE_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        for entry in price_data:
            writer.writerow([now, entry["symbol"], entry["price"]])

# ✅ Main auto-trading function
def run_autotrade():
    print("🚀 Starting AutoTrade...")
    if not is_market_open():
        print("⏹️ Market closed. Skipping trade.")
        return

    if has_open_position():
        print("⚠️ Unsold position found. Skipping today's trade.")
        return

    capital = get_available_capital()
    psu_candidates = fetch_psu_rows_from_master()

    price_log = []
    momentum_stocks = []

    for stock in psu_candidates:
        try:
            current_price = get_live_price(stock["symbol"])
            prev_price = get_historical_price(stock["symbol"], minutes_ago=5)

            if prev_price <= 0:
                continue

            momentum = ((current_price - prev_price) / prev_price) * 100
            qty = (int(capital // current_price) // stock["lot_size"]) * stock["lot_size"]

            if qty >= stock["lot_size"]:
                momentum_stocks.append({
                    **stock,
                    "price": current_price,
                    "momentum": momentum,
                    "quantity": qty
                })
                price_log.append({"symbol": stock["symbol"], "price": current_price})
        except Exception as e:
            print(f"❌ Failed for {stock['symbol']}: {e}")

    log_live_prices(price_log)
    if not momentum_stocks:
        print("⚠️ No eligible PSU stocks to trade today.")
        return

    top_stock = sorted(momentum_stocks, key=lambda x: x["momentum"], reverse=True)[0]

    payload = {
        "transactionType": "BUY",
        "exchangeSegment": "NSE_EQ",
        "productType": "CNC",
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": top_stock["security_id"],
        "tradingSymbol": top_stock["symbol"],
        "quantity": top_stock["quantity"],
        "price": 0,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
        "amoTime": "OPEN",
        "triggerPrice": 0,
        "smartOrder": False
    }

    print(f"📦 Placing order: {top_stock['symbol']} — {top_stock['quantity']} shares @ ₹{top_stock['price']} [Momentum: {round(top_stock['momentum'], 2)}%]")
    response = requests.post(BASE_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        print(f"✅ Order placed for {top_stock['symbol']} successfully!")

        # ✅ Log the trade to portfolio_log.csv
        timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")
        with open(PORTFOLIO_LOG, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                top_stock["symbol"],
                top_stock["price"],
                "",
                "HOLD",
                round(top_stock["momentum"], 2),
                1.5,
                1,
                top_stock["security_id"],
                top_stock["quantity"],
                timestamp,
                ""
            ])
        print("📝 Trade logged to portfolio_log.csv")
    else:
        print(f"❌ Order failed: {response.status_code} | {response.text}")

if __name__ == "__main__":
    run_autotrade()
