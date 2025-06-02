
# 📂 force_exit.py — Final Version (Cleaned & Verified)

import csv
from datetime import datetime, time as dtime
import requests
import json
import os
import pytz
import time as systime
from dhanhq import DhanContext, dhanhq
from dhan_api import get_live_price
from config import *
from utils_logger import log_bot_action
from utils_safety import safe_read_csv

now = datetime.now(pytz.timezone("Asia/Kolkata")).time()
if now >= dtime(15, 30):
    print("⏳ Market is closed. Skipping forced exit.")
    exit()

# ✅ Load Dhan credentials
with open("config.json") as f:
    config_data = json.load(f)

ACCESS_TOKEN = config_data["access_token"]
CLIENT_ID = config_data["client_id"]
TELEGRAM_TOKEN = config_data.get("telegram_token")
TELEGRAM_CHAT_ID = config_data.get("telegram_chat_id")

context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(context)

def log_bot_action(script_name, action, status, message):
    log_file = "bot_execution_log.csv"
    now = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
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

def force_exit():
    print("🚨 Starting Forced Exit check...")

    if not os.path.exists(PORTFOLIO_LOG) or os.stat(PORTFOLIO_LOG).st_size == 0:
        print("⚠️ No portfolio_log.csv found or empty. Skipping force exit.")
        return

    raw_lines = safe_read_csv(PORTFOLIO_LOG)
    reader = csv.DictReader(raw_lines)
    existing_rows = list(reader)

    now = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")
    headers = reader.fieldnames if reader.fieldnames else []
    updated_rows = []

    for row in existing_rows:
        if row.get("status", "").upper() == "SOLD":
            updated_rows.append(row)
            continue

        symbol = row["symbol"].strip().upper()
        security_id = row.get("security_id", "").strip()

        if not security_id:
            print(f"❌ Missing security_id for {symbol}")
            log_bot_action("force_exit.py", "FORCED SELL", "❌ FAILED", f"{symbol} - Missing ID")
            updated_rows.append(row)
            continue

        quantity = int(row["quantity"])

        try:
            live_price = get_live_price(symbol, security_id)
        except Exception as e:
            print(f"⚠️ Error fetching live price for {symbol}: {e}")
            updated_rows.append(row)
            continue

        print(f"🧚 Placing SELL: {symbol}, ID={security_id}, Qty={quantity}")
        exchange_segment = "NSE_EQ"
        code, response = place_sell_order(security_id, symbol, quantity, exchange_segment)
        

        if code == 200 and "order_id" in response:
            order_id = response["order_id"]
        
            time.sleep(3)  # Delay before checking status
            final_status = get_order_status(order_id)
            status_text = final_status.get("data", {}).get("orderStatus", "UNKNOWN")

            print(f"📟 Order ID {order_id} current status: {status_text}")

            if status_text.upper() in ["TRADED", "COMPLETED", "FILLED", "TRANSIT"]:
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
                net_profit = estimate_net_profit(buy_price, live_price, quantity)
                profit_status = "✅ PROFIT" if net_profit > 0 else "❌ LOSS"
                profit_pct = round(((live_price - buy_price) / buy_price) * 100, 2)

                summary_msg = (
                    f"📊 Forced Trade Summary ({datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')})\n"
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
                print(f"⚠️ Order {order_id} did not complete after retries: status={status_text}")
                log_bot_action("force_exit.py", "FORCED SELL", "⚠️ NOT TRADED", f"{symbol} - Status: {status_text}")
                updated_rows.append(row)
        else:
            if response.get("status") == "success":
                order_id = response['data'].get('orderId')
                print(f"✅ Force SELL placed for {symbol}: Order ID {order_id}")
                row.update({
                    "live_price": round(live_price, 2),
                    "change_pct": "",
                    "last_checked": now,
                    "status": "SOLD",
                    "exit_price": round(live_price, 2)
                })
            else:
                print(f"❌ Force SELL failed for {symbol}: {response}")
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
        try:
            total_profit = 0
            with open("sell_log.csv", newline="") as f:
                rows = list(csv.DictReader(f))
                for row in rows:
                    if row["reason"].strip().upper() == "FORCE EXIT":
                        try:
                            qty = int(row["qty"])
                            sell_price = float(row["sell_price"])
                            entry_price = float(row["entry_price"])
                            profit = (sell_price - entry_price) * qty
                            total_profit += profit
                        except:
                            continue

            with open("current_capital.csv", "r") as f:
                current_cap = float(f.read().strip())

            new_cap = round(current_cap + total_profit, 2)
            with open("current_capital.csv", "w") as f:
                f.write(str(new_cap))

            log_bot_action("force_exit.py", "CAPITAL SYNC", "✅ DONE", f"Capital updated to ₹{new_cap} after forced exit.")
        except Exception as e:
            log_bot_action("force_exit.py", "CAPITAL SYNC", "❌ ERROR", str(e))
            send_telegram_message(f"❌ Capital sync failed in force_exit.py: {e}")

    print("🚨 Force Exit complete.")


if __name__ == "__main__":
    if os.path.exists("emergency_exit.txt"):
        send_telegram_message("⛔ Emergency Exit active. Skipping force sell.")
        log_bot_action("force_exit.py", "SKIPPED", "EMERGENCY EXIT", "Force sell skipped due to emergency.")
    else:
        force_exit()
