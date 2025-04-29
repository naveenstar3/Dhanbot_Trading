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

# ✅ Startup Mode Alert
if TEST_MODE:
    print("🛠️ Running in TEST MODE. No real sell orders will happen.")
else:
    print("🚀 Running in LIVE PRODUCTION MODE. Real sell orders will happen!")

# ✅ Estimate Net Profit After Charges
def estimate_net_profit(buy_price, sell_price, quantity):
    gross_profit = (sell_price - buy_price) * quantity

    brokerage_total = BROKERAGE_PER_ORDER * 2  # buy + sell
    gst_on_brokerage = brokerage_total * (GST_PERCENTAGE / 100)
    stt_sell = sell_price * quantity * (STT_PERCENTAGE / 100)
    exchg_txn_charge = (buy_price + sell_price) * quantity * (EXCHANGE_TXN_CHARGE_PERCENTAGE / 100)
    sebi_charge = (buy_price + sell_price) * quantity * (SEBI_CHARGE_PERCENTAGE / 100)

    total_charges = brokerage_total + gst_on_brokerage + stt_sell + exchg_txn_charge + sebi_charge + DP_CHARGE_PER_SELL

    net_profit = gross_profit - total_charges
    return net_profit

# ✅ Place sell order (Handles TEST_MODE internally)
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

# ✅ Log Sell action
def log_sell(symbol, security_id, quantity, exit_price, reason):
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    with open(SELL_LOG, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, symbol, security_id, quantity, exit_price, reason])

# ✅ Market Open Check
def is_market_open():
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).time()
    return datetime.time(9, 30) <= now <= datetime.time(15, 30)

# ✅ Main Portfolio Evaluation
def check_portfolio():
    if not is_market_open():
        print("⏹️ Market closed. Skipping auto-sell.")
        return

    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")
    headers = ["timestamp", "symbol", "security_id", "quantity", "buy_price", "momentum_5min", "target_pct", "stop_pct", "live_price", "change_pct", "last_checked", "status", "exit_price"]

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

        # ✅ Net Profit Check
        net_profit = estimate_net_profit(buy_price, live_price, quantity)

        reason = ""
        should_sell = False

        # ✅ Dynamic Target or Stop Hit Check
        if change_pct >= target_pct:
            reason = "TARGET HIT"
            should_sell = True
        elif change_pct <= -stop_pct:
            reason = "STOP LOSS"
            should_sell = True

        # ✅ Minimum Net Profit Requirement Check
        if should_sell and net_profit >= MINIMUM_NET_PROFIT_REQUIRED:
            code, response = place_sell_order(security_id, symbol, quantity)
            if code == 200 and "order_id" in response:
                systime.sleep(2)
                order_id = response["order_id"]
                trade_book = get_trade_book()
                matching_trades = [trade for trade in trade_book if trade.get("order_id") == order_id]

                if matching_trades and matching_trades[0]["status"].upper() == "TRADED":
                    status = "SOLD"
                    exit_price = live_price
                    log_sell(symbol, security_id, quantity, live_price, reason)
                    print(f"✅ SOLD {symbol} at ₹{live_price} ({reason}) Net Profit: ₹{round(net_profit,2)}")
                    send_telegram_message(f"✅ SOLD {symbol} at ₹{live_price} ({reason}) Net Profit: ₹{round(net_profit,2)}")
                else:
                    print(f"⚠️ Sell order placed but NOT TRADED yet for {symbol}. Holding.")
            else:
                print(f"❌ SELL failed for {symbol}: {response}")
        else:
            print(f"⚠️ Holding {symbol}. Change% {round(change_pct,2)}%. Net Profit ₹{round(net_profit,2)}.")

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
        print("✅ Portfolio updated.")
    else:
        print("⚠️ No rows processed.")

if __name__ == "__main__":
    check_portfolio()
