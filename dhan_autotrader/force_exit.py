# 📂 force_exit.py — Final Version (Safe Forced Exit at 3:25 PM)

import csv
import datetime
import requests
import json
import os
import pytz
import time as systime
from dhanhq import DhanContext, dhanhq
from dhan_api import get_live_price
from config import *

# ✅ Load Dhan credentials
with open("dhan_config.json") as f:
    config_data = json.load(f)

ACCESS_TOKEN = config_data["access_token"]
CLIENT_ID = config_data["client_id"]

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# ✅ Initialize Dhan SDK
context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(context)

# ✅ Telegram Constants
TELEGRAM_TOKEN = "7557430361:AAFZKf4KBL3fScf6C67quomwCrpVbZxQmdQ"
TELEGRAM_CHAT_ID = "5086097664"

# ✅ Telegram Notification Function
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"⚠️ Telegram send error: {e}")

# ✅ Place Sell Order (Handles TEST_MODE internally)
def place_sell_order(security_id, symbol, quantity):
    if TEST_MODE:
        print(f"🛠️ [TEST MODE] Simulating SELL order for {symbol}")
        return 200, {"order_id": "TEST_ORDER_ID_SELL"}
    else:
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

# ✅ Fetch Trade Book
def get_trade_book():
    if TEST_MODE:
        print("🛠️ [TEST MODE] Simulating Trade Book fetch...")
        return [{"order_id": "TEST_ORDER_ID_SELL", "status": "TRADED"}]
    else:
        try:
            response = requests.get(TRADE_BOOK_URL, headers=HEADERS)
            if response.status_code == 200:
                return response.json()["data"]
            else:
                print(f"⚠️ Failed to fetch trade book: {response.status_code}")
                return []
        except Exception as e:
            print(f"⚠️ Exception during trade_book fetch: {e}")
            return []

# ✅ Forced Exit Logic
def force_exit():
    print("🚨 Starting Forced Exit check...")

    if not os.path.exists(PORTFOLIO_LOG) or os.stat(PORTFOLIO_LOG).st_size == 0:
        print("⚠️ No portfolio_log.csv found or empty. Skipping force exit.")
        return

    with open(PORTFOLIO_LOG, newline="") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")
    headers = reader.fieldnames if reader.fieldnames else []

    updated_rows = []

    for row in existing_rows:
        if row.get("status", "").upper() == "SOLD":
            updated_rows.append(row)
            continue

        symbol = row["symbol"]
        security_id = str(row["security_id"]).strip()
        quantity = int(row["quantity"])

        try:
            live_price = get_live_price(symbol)
        except Exception as e:
            print(f"⚠️ Error fetching live price for {symbol}: {e}")
            updated_rows.append(row)
            continue
            
        # 🔍 Validate order inputs
        if not security_id or quantity <= 0:
            print(f"❌ Invalid order params: symbol={symbol}, security_id={security_id}, qty={quantity}")
            updated_rows.append(row)
            continue
        
        print(f"🧪 Placing SELL: {symbol}, ID={security_id}, Qty={quantity}")


        # ✅ Place forced sell order
        code, response = place_sell_order(security_id, symbol, quantity)
        if code == 200 and "order_id" in response:
            systime.sleep(2)
            order_id = response["order_id"]
            trade_book = get_trade_book()
            matching_trades = [trade for trade in trade_book if trade.get("order_id") == order_id]

            if matching_trades and matching_trades[0]["status"].upper() == "TRADED":
                row.update({
                    "live_price": round(live_price, 2),
                    "change_pct": "",
                    "last_checked": now,
                    "status": "SOLD",
                    "exit_price": round(live_price, 2)
                })
                updated_rows.append(row)
                print(f"✅ Force SOLD {symbol} at ₹{round(live_price,2)}")
                send_telegram_message(f"✅ Force SOLD {symbol} at ₹{round(live_price,2)} (EOD Exit)")
            else:
                print(f"⚠️ Force sell order placed but not traded for {symbol}.")
                updated_rows.append(row)
        else:
            print(f"❌ Force SELL failed for {symbol}: {response}")
            updated_rows.append(row)

    if updated_rows:
        with open(PORTFOLIO_LOG, "w", newline="", encoding="utf-8") as f:
            fieldnames = headers
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        print("✅ Portfolio updated after force exit.")

    print("🚨 Force Exit complete.")

# ✅ Main
if __name__ == "__main__":
    force_exit()
