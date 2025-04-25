import csv
import datetime
import requests
import json
from dhan_api import get_live_price

# Configuration
LOG_FILE = "portfolio_log.csv"
SELL_LOG_FILE = "sell_log.csv"

# ✅ Load Dhan credentials
with open("dhan_config.json") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
SELL_URL = "https://api.dhan.co/orders"
HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

def place_sell_order(security_id, symbol, quantity):
    payload = {
        "transactionType": "SELL",
        "exchangeSegment": "NSE_EQ",
        "productType": "CNC",
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": security_id,
        "tradingSymbol": symbol,
        "quantity": quantity,
        "price": 0,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
        "amoTime": "OPEN",
        "triggerPrice": 0,
        "smartOrder": False
    }
    try:
        response = requests.post(SELL_URL, headers=HEADERS, json=payload)
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}

def log_sell(symbol, security_id, quantity, exit_price, reason):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SELL_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, symbol, security_id, quantity, exit_price, reason])

def check_portfolio():
    updated_rows = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "SOLD":
                updated_rows.append(row)
                continue

            symbol = row["symbol"]
            security_id = str(row["security_id"]).strip()
            buy_price = float(row["buy_price"])
            quantity = int(row["quantity"])
            target_pct = float(row["target_pct"])
            stop_pct = float(row["stop_pct"])

            try:
                live_price = get_live_price(symbol)
            except Exception as e:
                print(f"⚠️ Skipping {symbol}: {e}")
                continue

            change_pct = ((live_price - buy_price) / buy_price) * 100
            status = "HOLD"
            exit_price = ""

            if change_pct >= target_pct:
                status = "TARGET HIT"
                exit_price = live_price
                code, response = place_sell_order(security_id, symbol, quantity)
                if code == 200:
                    status = "SOLD"
                    log_sell(symbol, security_id, quantity, live_price, "TARGET HIT")
                    print(f"✅ SOLD {symbol} at ₹{live_price} (Target Hit)")
                else:
                    print(f"❌ SELL failed for {symbol}: {response}")
            elif change_pct <= -abs(stop_pct):
                status = "STOP LOSS"
                exit_price = live_price
                code, response = place_sell_order(security_id, symbol, quantity)
                if code == 200:
                    status = "SOLD"
                    log_sell(symbol, security_id, quantity, live_price, "STOP LOSS")
                    print(f"✅ SOLD {symbol} at ₹{live_price} (Stop Loss)")
                else:
                    print(f"❌ SELL failed for {symbol}: {response}")

            row.update({
                "live_price": round(live_price, 2),
                "change_pct": round(change_pct, 2),
                "last_checked": now,
                "status": status,
                "exit_price": exit_price
            })
            updated_rows.append(row)

    if updated_rows:
        fieldnames = updated_rows[0].keys()
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in updated_rows:
                writer.writerow(row)
        print("✅ Portfolio updated and SELLs processed.")
    else:
        print("⚠️ No valid updates available. No rows written.")

if __name__ == "__main__":
    check_portfolio()
