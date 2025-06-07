import csv
import datetime
import requests
import json
import os
import pytz
import time as systime
from dhanhq import DhanContext, dhanhq
from dhan_api import get_live_price, get_intraday_candles, get_historical_price, get_security_id
from config import *
from utils_logger import log_bot_action
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils_safety import safe_read_csv
import pandas as pd
import portalocker 
from db_logger import insert_live_trail_to_db, insert_portfolio_log_to_db, update_portfolio_log_to_db

# ✅ Trailing Exit Config
LIVE_BUFFER_FILE = "live_trail_BUFFER.csv"

# ✅ Load Dhan credentials
with open("config.json") as f:
    config_data = json.load(f)

ACCESS_TOKEN = config_data["access_token"]
CLIENT_ID = config_data["client_id"]
TELEGRAM_TOKEN = config_data.get("telegram_token")
TELEGRAM_CHAT_ID = config_data.get("telegram_chat_id")

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# ✅ Initialize Dhan SDK
context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(context)

# ✅ Bot Execution Logger
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

def force_exit_hold_stocks():
    print("🚨 [Force Exit] Triggered at 3:25 PM...")

    if not os.path.exists(PORTFOLIO_LOG) or os.stat(PORTFOLIO_LOG).st_size == 0:
        print("⚠️ portfolio_log.csv missing or empty.")
        return

    raw_lines = safe_read_csv(PORTFOLIO_LOG)
    reader = csv.DictReader(raw_lines)
    existing_rows = list(reader)
    headers = reader.fieldnames if reader.fieldnames else []
    updated_rows = []

    for row in existing_rows:
        if row.get("status", "").upper() == "SOLD":
            updated_rows.append(row)
            continue

        symbol = row["symbol"].strip().upper()
        security_id = row.get("security_id", "").strip()
        if not security_id:
            print(f"❌ [Force Exit] Missing security_id for {symbol}")
            updated_rows.append(row)
            continue

        quantity = int(row["quantity"])
        try:
            live_price = get_live_price(symbol, security_id)
        except Exception as e:
            print(f"⚠️ [Force Exit] Live price error for {symbol}: {e}")
            updated_rows.append(row)
            continue

        code, response = place_sell_order(security_id, symbol, quantity, "NSE_EQ")
        now_dt = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        now_str = now_dt.strftime("%Y-%m-%d %H:%M")

        if code == 200 and "order_id" in response:
            row.update({
                "live_price": round(live_price, 2),
                "change_pct": "",
                "last_checked": now_str,
                "status": "SOLD",
                "exit_price": round(live_price, 2)
            })

            # ✅ DB log update: convert `now_dt` to UTC if needed
            update_portfolio_log_to_db(
                trade_date=now_dt,
                symbol=symbol,
                security_id=security_id,
                quantity=quantity,
                buy_price=float(row.get("buy_price", 0)),
                stop_pct=float(row.get("stop_pct", 1)),
                exit_price=round(live_price, 2),
                live_price=round(live_price, 2),
                status="SOLD"
            )

            updated_rows.append(row)

            send_telegram_message(f"✅ Force SOLD {symbol} at ₹{round(live_price,2)} (EOD Exit)")
            log_bot_action("portfolio_tracker.py", "FORCED SELL", "✅ TRADED", f"{symbol} @ ₹{round(live_price, 2)}")
        else:
            print(f"❌ [Force Exit] Failed for {symbol}: {response}")
            updated_rows.append(row)

    if updated_rows:
        with open(PORTFOLIO_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(updated_rows)
        print("✅ [Force Exit] portfolio_log.csv updated after forced exit.")
        
def monitor_hold_positions():
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    if now.hour == 15 and now.minute >= 0 and now.minute < 5:  # Trigger at 3:00–3:04 PM
        if not os.path.exists(PORTFOLIO_LOG):
            return
        with open(PORTFOLIO_LOG, newline="") as f:
            reader = csv.DictReader(f)
            hold_stocks = [row["symbol"] for row in reader if row["status"].upper() == "HOLD"]
        
        if hold_stocks:
            msg = f"⏳ {len(hold_stocks)} stock(s) still HOLD at 3:00 PM: {', '.join(hold_stocks)}"
            print(msg)
            send_telegram_message(msg)
            log_bot_action("portfolio_tracker.py", "3PM check", "⚠️ HOLDING", msg)

def get_dynamic_minimum_net_profit(capital):
    """
    Returns scaled minimum net profit:
    - Minimum ₹5
    - Scales as 0.1% of current capital
    """
    return max(5, round(capital * 0.001, 2))
    
# ✅ Peak Exhaustion Detection — enhanced smart sell logic
def is_peak_exhausted(symbol, target_pct=0.07, grace_band=0.01, min_retries=5):
    try:
        now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        if now.hour < 14 or (now.hour == 14 and now.minute < 15):
            return False  # too early to judge exhaustion

        if not os.path.exists(LIVE_BUFFER_FILE):
            return False

        recent_records = []
        with open(LIVE_BUFFER_FILE, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 4:
                    continue
                ts_str, sym, price, change = row
                if sym.strip().upper() != symbol.upper():
                    continue
                ts = pytz.timezone("Asia/Kolkata").localize(datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
                if (now - ts).seconds <= 3600:  # last 60 min
                    recent_records.append(float(change))

        if len(recent_records) < 10:
            return False

        near_target_hits = [x for x in recent_records if (target_pct - grace_band) <= x < target_pct]
        if len(near_target_hits) >= min_retries:
            return True

    except Exception as e:
        print(f"⚠️ Error in peak exhaustion check for {symbol}: {e}")
    return False
    
def log_live_trail(symbol, live_price, change_pct):
    try:
        print(f"📦 Entering log_live_trail for {symbol} at price {live_price} with change_pct {change_pct}")
        print(f"🔒 Attempting to acquire lock on {LIVE_BUFFER_FILE}")

        file_empty = not os.path.exists(LIVE_BUFFER_FILE) or os.stat(LIVE_BUFFER_FILE).st_size == 0

        with portalocker.Lock(LIVE_BUFFER_FILE, mode='a', timeout=5, newline='') as f:
            print(f"✍️ Locked and opening {LIVE_BUFFER_FILE} in append mode")
            writer = csv.writer(f)

            # Add header if file is new/empty
            if file_empty:
                writer.writerow(["timestamp", "symbol", "price", "change_pct"])

            now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([now, symbol, round(live_price, 2), round(change_pct, 2)])
            insert_live_trail_to_db(now, symbol, round(live_price, 2), round(change_pct, 2))
            print(f"✅ Successfully wrote to {LIVE_BUFFER_FILE}")

    except PermissionError as e:
        print(f"⛔ Permission denied when logging {symbol} to {LIVE_BUFFER_FILE}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error in log_live_trail for {symbol}: {e}")


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
def place_sell_order(security_id, symbol, quantity, exchange_segment="NSE_EQ"):
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
    
# ✅ Intelligent Early Exit Analyzer
def should_exit_early(symbol, current_price):
    try:
        records = []
        with open(LIVE_BUFFER_FILE, newline='') as f:
            reader = csv.reader(f)
            for row_ in reader:
                if len(row_) != 4:
                    continue
                ts_str, sym, price, change = row_
                if sym.strip().upper() == symbol.upper():
                    try:
                        records.append((ts_str, float(change)))
                    except:
                        continue

        if not records or len(records) < 4:
            return False

        max_change = max(records, key=lambda x: x[1])
        peak_time = max_change[0]
        peak_change = max_change[1]

        now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        current_change = records[-1][1]

        if now.hour >= 14 and now.minute >= 45:
            if (peak_change - current_change) >= 0.4:
                return True

        # ✅ RSI check using Dhan historical API
        from dhan_api import get_historical_price
        security_id = get_security_id(symbol)
        raw_data = get_historical_price(security_id, interval="15")
        if not raw_data or "close" not in raw_data:
            return False

        df = pd.DataFrame({"close": raw_data["close"]})
        rsi_series = calculate_rsi(df['close'])
        if not rsi_series.empty and rsi_series.iloc[-1] > 70:
            print(f"⚠️ RSI triggered exit: {symbol} has RSI {round(rsi_series.iloc[-1], 2)}")
            return True

        return False

    except Exception as e:
        print(f"⚠️ Error in early exit check for {symbol}: {e}")
        return False

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ✅ Main Portfolio Evaluation
def is_nse_trading_day():
    today = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).date()
    if today.weekday() >= 5:
        return False

    year = today.year
    fname = f"nse_holidays_{year}.csv"

    if not os.path.exists(fname):
        try:
            print(f"📥 Downloading NSE holiday calendar for {year}...")
            url = "https://www.nseindia.com/api/holiday-master?type=trading"
            headers = {"User-Agent": "Mozilla/5.0"}
            s = requests.Session()
            s.headers.update(headers)
            r = s.get(url, timeout=10)
            data = r.json()
            if "Trading" not in data:
                raise ValueError("Missing 'Trading' key in NSE holiday API response")
            holidays = data["Trading"]           
            dates = [datetime.datetime.strptime(d["date"], "%d-%b-%Y").date() for d in holidays if str(year) in d["date"]]
            pd.DataFrame({"date": dates}).to_csv(fname, index=False)
        except Exception as e:
            print(f"⚠️ NSE holiday fetch failed: {e}")
            return True

    try:
        hdf = pd.read_csv(fname)
        holiday_dates = pd.to_datetime(hdf["date"]).dt.date.tolist()
        return today not in holiday_dates
    except:
        return True

def check_portfolio():
    if not is_market_open() or not is_nse_trading_day():
        print("⏹️ Market closed or holiday. Skipping auto-sell.")
        log_bot_action("portfolio_tracker.py", "market_status", "INFO", "Skipped: Market closed or holiday.")
        return

    monitor_hold_positions()
    ist_now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    if ist_now.hour == 9 and ist_now.minute <= 30 and os.path.exists(LIVE_BUFFER_FILE):
        os.remove(LIVE_BUFFER_FILE)
        print("🧹 Cleared live_trail_BUFFER.csv for new day.")

    now = ist_now.strftime("%Y-%m-%d %H:%M")
    headers = ["timestamp", "symbol", "security_id", "quantity", "buy_price", "momentum_5min",
               "target_pct", "stop_pct", "live_price", "change_pct", "last_checked", "status", "exit_price"]

    if not os.path.exists(PORTFOLIO_LOG) or os.stat(PORTFOLIO_LOG).st_size == 0:
        with open(PORTFOLIO_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print("📁 Created new portfolio_log.csv with headers only.")
        return

    from utils_safety import safe_read_csv
    raw_lines = safe_read_csv(PORTFOLIO_LOG)
    reader = csv.DictReader(raw_lines)
    existing_rows = list(reader)
    
    def process_sell_logic(row):
        print(f"🔍 Starting processing for {row.get('symbol')}")
        if row.get("status") == "SOLD":
            return row

        try:
            symbol = row["symbol"]
            security_id = str(row["security_id"]).strip()
            buy_price = float(row["buy_price"])
            quantity = int(row["quantity"])
            target_pct = float(row.get("target_pct", 1.5))
            stop_pct = float(row.get("stop_pct", 1))

            security_id = get_security_id(symbol)
            print(f"📌 Calling get_live_price with: {security_id}")
            live_price = get_live_price(symbol, security_id)
            print(f"✅ Live price for {symbol} = {live_price}")
            
            if live_price is None:
                print(f"⚠️ Failed to get live price for {symbol}")
                return row
            change_pct = ((live_price - buy_price) / buy_price) * 100
            log_live_trail(symbol, live_price, change_pct)

            status = row.get("status", "HOLD")
            exit_price = row.get("exit_price", "")
            now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")

            net_profit = estimate_net_profit(buy_price, live_price, quantity)
            capital_in_stock = buy_price * quantity
            MINIMUM_NET_PROFIT_REQUIRED = get_dynamic_minimum_net_profit(capital_in_stock)
            max_rupee_loss = capital_in_stock * 0.004
            actual_loss = (buy_price - live_price) * quantity

            reason = ""
            should_sell = False
            
            records = []
            if os.path.exists(LIVE_BUFFER_FILE):
                with open(LIVE_BUFFER_FILE, newline='') as f:
                    reader = csv.reader(f)
                    for row_ in reader:
                        if len(row_) != 4:
                            continue
                        ts_str, sym, price, change = row_
                        if sym.strip().upper() == symbol.upper():
                            try:
                                records.append((ts_str, float(change)))
                            except:
                                continue
            
            max_trail = max([r[1] for r in records if r[1] is not None], default=0)
            trailing_drop = max_trail - change_pct
            
            if change_pct >= target_pct:
                reason = "TARGET HIT"
                should_sell = True
            # 🧠 Predictive peak logic using linear regression
            if len(records) >= 15:
                import numpy as np
                from sklearn.linear_model import LinearRegression
            
                X = np.array(list(range(len(records)))).reshape(-1, 1)
                y = np.array([r[1] for r in records]).reshape(-1, 1)
            
                model = LinearRegression()
                model.fit(X, y)
            
                future_index = len(records) + 5  # Predict 5 mins ahead
                predicted_peak = float(model.predict([[future_index]])[0])
                potential_upside = predicted_peak - change_pct
            
                print(f"🔮 AI Prediction: Current={round(change_pct,2)}%, Predicted Peak={round(predicted_peak,2)}%")
            
                if potential_upside < 0.2 and trailing_drop > 0.1:
                    reason = f"AI EXIT: Predicted peak only {round(predicted_peak,2)}%, current={round(change_pct,2)}%"
                    should_sell = True
            
            # Fallback to original peak drop
            elif max_trail >= 0.5 and trailing_drop >= 0.25:
                reason = f"PEAK DROP: Peaked at {round(max_trail,2)}%, now at {round(change_pct,2)}%"
                should_sell = True            
            
            elif change_pct <= -stop_pct:
                reason = "STOP LOSS"
                should_sell = True
            elif should_exit_early(symbol, live_price):
                reason = "SMART EXIT: Dropped from peak after 2:45 PM"
                should_sell = True
            elif actual_loss >= max_rupee_loss:
                reason = f"FORCED EXIT: Max ₹ loss {round(actual_loss, 2)} > {round(max_rupee_loss, 2)}"
                should_sell = True
            elif is_peak_exhausted(symbol, target_pct=target_pct):
                reason = "SMART EXIT: Price exhausted near target multiple times"
                should_sell = True

            if should_sell and net_profit >= MINIMUM_NET_PROFIT_REQUIRED:
                exchange_segment = "NSE_EQ"
                code, response = place_sell_order(security_id, symbol, quantity, exchange_segment)
                if code == 200 and "order_id" in response:
                    order_id = response["order_id"]
                    trade_status = None
                    for _ in range(5):
                        trade_book = get_trade_book()
                        matching_trades = [t for t in trade_book if t.get("order_id") == order_id]
                        if matching_trades:
                            trade_status = matching_trades[0].get("status", "").upper()
                            if trade_status == "TRADED":
                                break
                        systime.sleep(2)

                    if trade_status == "TRADED":
                       status = "SOLD"
                       exit_price = live_price
                       log_sell(symbol, security_id, quantity, live_price, reason)
                       print(f"✅ SOLD {symbol} at ₹{live_price} ({reason}) Net Profit: ₹{round(net_profit, 2)}")
                       send_telegram_message(f"✅ SOLD {symbol} at ₹{live_price} ({reason}) Net Profit: ₹{round(net_profit, 2)}")
                       log_bot_action("portfolio_tracker.py", "SELL executed", "✅ TRADED", f"{symbol} @ ₹{round(live_price, 2)} | Reason: {reason}")
                   
                       # ✅ DB Update — convert timestamp to IST
                       update_portfolio_log_to_db(
                           trade_date=datetime.datetime.now(pytz.timezone("Asia/Kolkata")),
                           symbol=symbol,
                           security_id=security_id,
                           quantity=quantity,
                           buy_price=buy_price,
                           stop_pct=stop_pct,
                           exit_price=exit_price,
                           live_price=live_price,
                           status="SOLD"
                       )
                   
                       profit_status = "✅ PROFIT" if net_profit > 0 else "❌ LOSS"
                       profit_pct = round(((exit_price - buy_price) / buy_price) * 100, 2)
                       summary_msg = (
                           f"📊 Trade Summary ({now})\n"
                           f"Stock: {symbol}\n"
                           f"Buy Price: ₹{round(buy_price, 2)}\n"
                           f"Sell Price: ₹{round(exit_price, 2)}\n"
                           f"Qty: {quantity}\n"
                           f"Net Profit: ₹{round(net_profit, 2)}\n"
                           f"Profit %: {profit_pct}%\n"
                           f"Status: {profit_status}"
                       )
                       send_telegram_message(summary_msg)
                   
                    else:
                        print(f"⚠️ Sell order placed but NOT TRADED yet for {symbol}. Holding.")
                else:
                    print(f"❌ SELL failed for {symbol}: {response}")
                    send_telegram_message(f"❌ SELL failed for {symbol}: {response}")
            else:
                hold_reason = ""
                if not should_sell:
                    hold_reason = "🚫 Conditions not met: Target/Stop/Peak not triggered."
                elif net_profit < MINIMUM_NET_PROFIT_REQUIRED:
                    hold_reason = (
                        f"💸 Blocked by Net Profit Rule: ₹{round(net_profit, 2)} < ₹{round(MINIMUM_NET_PROFIT_REQUIRED, 2)}"
                    )
                else:
                    hold_reason = "❓ Unknown reason. Needs manual review."

                print(f"⚠️ HOLDING {symbol}. Change%: {round(change_pct, 2)}%. "
                      f"Net Profit: ₹{round(net_profit, 2)}. Reason: {hold_reason}")
                log_bot_action("portfolio_tracker.py", "SELL skipped", "⚠️ HOLD", f"{symbol} | {hold_reason}")

            row.update({
                "live_price": round(live_price, 2),
                "change_pct": round(change_pct, 2),
                "last_checked": now,
                "status": status,
                "exit_price": exit_price
            })
            return row

        except Exception as e:
            print(f"⚠️ Error processing {row.get('symbol')}: {e}")
            return row

    # ✅ Execute all in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_sell_logic, existing_rows))
        updated_rows = [r for r in results if r]

    if updated_rows:
        with open(PORTFOLIO_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(updated_rows)
        print("✅ Portfolio updated.")

        # ✅ Also log to PostgreSQL DB for each row
        for row in updated_rows:
            if row.get("status") == "HOLD":
                trade_date = row.get("timestamp") or datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                insert_portfolio_log_to_db(
                    trade_date,
                    row["symbol"],
                    row["security_id"],
                    int(row["quantity"]),
                    float(row["buy_price"]),
                    float(row.get("stop_pct", 1))
                )
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).time()
    if now >= datetime.time(15, 24) and now < datetime.time(15, 30):
        force_exit_hold_stocks()
    
    
if __name__ == "__main__":
    if os.path.exists("emergency_exit.txt"):
        send_telegram_message("⛔ Emergency Exit active. Skipping HOLD monitoring.")
        log_bot_action("portfolio_tracker.py", "SKIPPED", "EMERGENCY EXIT", "Monitoring skipped due to emergency exit.")
    else:
        check_portfolio()
