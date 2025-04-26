import csv
import datetime
import requests
import json
import os
from dhanhq import DhanContext, dhanhq
from dhan_api import get_live_price

PORTFOLIO_LOG = "portfolio_log.csv"
SELL_LOG = "sell_log.csv"

# ✅ Load Dhan credentials
with open("dhan_config.json") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

# ✅ Initialize Dhan SDK
context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(context)

# ✅ Place a SELL order using official SDK
def place_sell_order(security_id, symbol, quantity):
    try:
        response = dhan.place_order(
            security_id=security_id,
            exchange_segment="NSE",
            transaction_type="SELL",
            quantity=quantity,
            order_type="MARKET",
            product_type="CNC",
            price=0
        )
        return 200, response
    except Exception as e:
        return 500, {"error": str(e)}

# ✅ Log to sell_log.csv
def log_sell(symbol, security_id, quantity, exit_price, reason):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SELL_LOG, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, symbol, security_id, quantity, exit_price, reason])

# ✅ Market hours checker
def is_market_open():
    now = datetime.datetime.now().time()
    return datetime.time(9, 30) <= now <= datetime.time(15, 30)

# ✅ Main portfolio evaluation
def check_portfolio():
    if not is_market_open():
        print("⏹️ Market closed. Skipping auto-sell.")
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    headers = ["timestamp", "symbol", "security_id", "quantity", "buy_price", "target_pct", "stop_pct", "live_price", "change_pct", "last_checked", "status", "exit_price"]

    if not os.path.exists(PORTFOLIO_LOG) or os.stat(PORTFOLIO_LOG).st_size == 0:
        with open(PORTFOLIO_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print("📁 Created new portfolio_log.csv with headers only.")
        return

    with open(PORTFOLIO_LOG, newline="") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

    updated_rows = []

    for row in existing_rows:
        if row.get("status") == "SOLD":
            updated_rows.append(row)
            continue

        symbol = row["symbol"]
        security_id = str(row["security_id"]).strip()
        buy_price = float(row["buy_price"])
        quantity = int(row["quantity"])
        target_pct = float(row.get("target_pct", 1.5))
        stop_pct = float(row.get("stop_pct", 1))

        try:
            live_price = get_live_price(symbol)
        except Exception as e:
            print(f"⚠️ Skipping {symbol}: {e}")
            updated_rows.append(row)
            continue

        change_pct = ((live_price - buy_price) / buy_price) * 100
        status = row.get("status", "HOLD")
        exit_price = row.get("exit_price", "")

        if change_pct >= target_pct:
            code, response = place_sell_order(security_id, symbol, quantity)
            if code == 200 and "order_id" in response:
                status = "SOLD"
                exit_price = live_price
                log_sell(symbol, security_id, quantity, live_price, "TARGET HIT")
                print(f"✅ SOLD {symbol} at ₹{live_price} (Target Hit)")
            else:
                print(f"❌ SELL failed or blocked for {symbol}: {response}")
                status = "HOLD"
                exit_price = ""
        elif change_pct <= -abs(stop_pct):
            code, response = place_sell_order(security_id, symbol, quantity)
            if code == 200 and "order_id" in response:
                status = "SOLD"
                exit_price = live_price
                log_sell(symbol, security_id, quantity, live_price, "STOP LOSS")
                print(f"✅ SOLD {symbol} at ₹{live_price} (Stop Loss)")
            else:
                print(f"❌ SELL failed or blocked for {symbol}: {response}")
                status = "HOLD"
                exit_price = ""

        row.update({
            "live_price": round(live_price, 2),
            "change_pct": round(change_pct, 2),
            "last_checked": now,
            "status": status,
            "exit_price": exit_price
        })
        updated_rows.append(row)

    if updated_rows:
        with open(PORTFOLIO_LOG, "w", newline="", encoding="utf-8") as f:
            fieldnames = headers
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        print("✅ Portfolio updated. One record per stock maintained.")
    else:
        print("⚠️ No rows processed.")

if __name__ == "__main__":
    check_portfolio()
