# 📂 force_exit.py — Final Version (Cleaned & Verified)

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
from utils_logger import log_bot_action

# ✅ Load Dhan credentials
with open("dhan_config.json") as f:
    config_data = json.load(f)

ACCESS_TOKEN = config_data["access_token"]
CLIENT_ID = config_data["client_id"]

context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(context)

TELEGRAM_TOKEN = "7557430361:AAFZKf4KBL3fScf6C67quomwCrpVbZxQmdQ"
TELEGRAM_CHAT_ID = "5086097664"

def log_bot_action(script_name, action, status, message):
    log_file = "bot_execution_log.csv"
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    headers = ["timestamp", "script_name", "action", "status", "message"]
    new_row = [now, script_name, action, status, message]
    file_exists = os.path.exists(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(new_row)

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"⚠️ Telegram send error: {e}")

def place_sell_order(security_id, symbol, quantity, exchange_segment):
    if TEST_MODE:
        print(f"🛠️ [TEST MODE] Simulating SELL order for {symbol}")
        return 200, {"order_id": "TEST_ORDER_ID_SELL"}
    try:
        seg = dhan.NSE if exchange_segment.upper() == "NSE_EQ" else dhan.BSE
        response = dhan.place_order(
            security_id=security_id,
            exchange_segment=seg,
            transaction_type=dhan.SELL,
            quantity=quantity,
            order_type=dhan.MARKET,
            product_type=dhan.CNC,
            price=0
        )
        return 200, response
    except Exception as e:
        return 500, {"error": str(e)}

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

        symbol = row["symbol"].strip().upper()
        security_id = row.get("security_id", "").strip()
        exchange_segment = "NSE_EQ"

        if not security_id:
            print(f"❌ Missing security_id for {symbol}")
            log_bot_action("force_exit.py", "FORCED SELL", "❌ FAILED", f"{symbol} - Missing ID")
            updated_rows.append(row)
            continue

        quantity = int(row["quantity"])

        try:
            live_price = get_live_price(symbol)
        except Exception as e:
            print(f"⚠️ Error fetching live price for {symbol}: {e}")
            updated_rows.append(row)
            continue

        print(f"🧚 Placing SELL: {symbol}, ID={security_id}, Qty={quantity}")
        code, response = place_sell_order(security_id, symbol, quantity, exchange_segment)

        if code == 200 and "order_id" in response:
            systime.sleep(2)
            order_id = response["order_id"]

            max_retries = 5
            trade_status = None
            matching_trades = []

            for attempt in range(max_retries):
                trade_book = get_trade_book()
                matching_trades = [t for t in trade_book if t.get("order_id") == order_id]
                if matching_trades:
                    trade_status = matching_trades[0].get("status", "").upper()
                    if trade_status == "TRADED":
                        break
                systime.sleep(2)

            if trade_status == "TRADED":
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
                log_bot_action("force_exit.py", "FORCED SELL", "✅ TRADED", f"{symbol} @ ₹{round(live_price, 2)} (EOD Exit)")

                buy_price = float(row.get("buy_price", 0))
                net_profit = (live_price - buy_price) * quantity
                profit_status = "✅ PROFIT" if net_profit > 0 else "❌ LOSS"
                profit_pct = round(((live_price - buy_price) / buy_price) * 100, 2)

                summary_msg = (
                    f"📊 Forced Trade Summary ({datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')})\n"
                    f"Stock: {symbol}\n"
                    f"Buy Price: ₹{round(buy_price, 2)}\n"
                    f"Sell Price: ₹{round(live_price, 2)}\n"
                    f"Qty: {quantity}\n"
                    f"Net Profit: ₹{round(net_profit, 2)}\n"
                    f"Profit %: {profit_pct}%\n"
                    f"Status: {profit_status}"
                )
                send_telegram_message(summary_msg)
            else:
                print(f"⚠️ Order {order_id} did not complete after retries: status={trade_status}")
                log_bot_action("force_exit.py", "FORCED SELL", "⚠️ NOT TRADED", f"{symbol} - Status: {trade_status}")
                updated_rows.append(row)
        else:
            print(f"❌ Force SELL failed for {symbol}: {response}")
            log_bot_action("force_exit.py", "FORCED SELL", "❌ FAILED", f"{symbol} - API Error: {response}")
            updated_rows.append(row)

    if updated_rows:
        with open(PORTFOLIO_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(updated_rows)
        print("✅ Portfolio updated after force exit.")

    hold_stocks = [row for row in updated_rows if row.get("status", "").upper() != "SOLD"]
    if hold_stocks:
        msg = f"🚨 Final Check: {len(hold_stocks)} stock(s) still in HOLD\n"
        for r in hold_stocks:
            msg += f"Symbol: {r['symbol']}\n"
        msg += "⚠️ Please check manually before next trade day."
        send_telegram_message(msg)

    print("🚨 Force Exit complete.")

if __name__ == "__main__":
    force_exit()