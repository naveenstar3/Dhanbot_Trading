import csv
import datetime
import requests
import json
import os
import pytz
import time as systime
import pandas as pd
import portalocker
import numpy as np
import sys
import io
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
from dhanhq import DhanContext, dhanhq
from dhan_api import get_live_price, get_historical_price
from config import *
from utils_logger import log_bot_action
from utils_safety import safe_read_csv
from db_logger import insert_live_trail_to_db, insert_portfolio_log_to_db

# ✅ Enable TeeLogger to capture print logs to file
log_buffer = io.StringIO()

class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = TeeLogger(sys.__stdout__, log_buffer)

# ✅ Trailing Exit Config
LIVE_BUFFER_FILE = "live_trail_BUFFER.csv"
MIN_HOLDING_MINUTES = 15  # Minimum time after purchase before selling is allowed

# ✅ Load Dhan credentials
with open("config.json") as f:
    config_data = json.load(f)

ACCESS_TOKEN = config_data["access_token"]
CLIENT_ID = config_data["client_id"]
TRADE_BOOK_URL = config_data.get("trade_book_url", "https://api.dhan.co/trade-book")
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
        with portalocker.Lock(LIVE_BUFFER_FILE, 'r', timeout=5) as f:  # Added locking
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
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
    
def log_live_trail(symbol, live_price, change_pct, order_id=None):
    try:
        # Purge file if symbol being tracked is different from current
        now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        if now.hour == 9 and now.minute <= 30 and os.path.exists(LIVE_BUFFER_FILE):
            print(f"🧹 Clearing buffer at market open for new day: {LIVE_BUFFER_FILE}")
            with portalocker.Lock(LIVE_BUFFER_FILE, 'w', timeout=5) as f:
                f.write("timestamp,symbol,price,change_pct\n")  # Removed order_id

        # Append new price trail
        with portalocker.Lock(LIVE_BUFFER_FILE, 'a', timeout=5, newline='') as f:
            if os.stat(LIVE_BUFFER_FILE).st_size == 0:
                f.write("timestamp,symbol,price,change_pct\n")  # Removed order_id
            writer = csv.writer(f)
            now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([now, symbol, round(live_price, 2), round(change_pct, 2)])  # Removed order_id
            # Keep DB logging with order_id
            insert_live_trail_to_db(now, symbol, round(live_price, 2), round(change_pct, 2), order_id=order_id)
            print(f"✅ Logged {symbol} trail to {LIVE_BUFFER_FILE}")

    except Exception as e:
        print(f"❌ Error in log_live_trail for {symbol}: {e}")

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
        if not os.path.exists(LIVE_BUFFER_FILE):
            return False
            
        with portalocker.Lock(LIVE_BUFFER_FILE, 'r', timeout=5) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                ts_str, sym, price, change = row
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
        security_id = get_security_id(symbol)
        candles = get_historical_price(security_id, interval="15")
        if not candles:
            return False
            
        closes = [candle['close'] for candle in candles if 'close' in candle]
        if len(closes) < 14:
            return False
            
        df = pd.DataFrame({"close": closes})
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

# Initialize traded symbols set
if 'attempted_security_ids' not in globals():
    attempted_security_ids = set()

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
               "target_pct", "stop_pct", "live_price", "change_pct", "last_checked", "status", "exit_price", "order_id"]

    if not os.path.exists(PORTFOLIO_LOG) or os.stat(PORTFOLIO_LOG).st_size == 0:
        with open(PORTFOLIO_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print("📁 Created new portfolio_log.csv with headers only.")
        return
    
    try:
        with portalocker.Lock(PORTFOLIO_LOG, 'r', timeout=5) as f:
            # Verify file is not empty
            if os.stat(PORTFOLIO_LOG).st_size == 0:
                print("⚠️ Skipping: portfolio_log.csv is empty.")
                return
                
            # Read and parse CSV content
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            
            if not existing_rows:
                print("⚠️ Skipping: No valid rows found in portfolio_log.csv")
                return
                
    except Exception as e:
        print(f"⚠️ Error reading portfolio_log.csv: {e}")
        return
    
    def process_sell_logic(row):
        print(f"🔍 Starting processing for {row.get('symbol')}")
        # Single symbol-based check - prevents duplicate processing
        if row.get("status") in ["PROFIT", "STOP LOSS", "FORCE_EXIT"]:
            print(f"🛑 {row.get('symbol')} already closed. Skipping.")
            return row
        
        security_id = str(row.get("security_id", "")).strip()
        if not security_id:
            print(f"⚠️ Skipping {row.get('symbol')} - missing security ID")
            return row
            
        if security_id in attempted_security_ids:
            print(f"🛑 {row.get('symbol')} already attempted this session. Skipping.")
            return row

        try:
            symbol = row["symbol"]
            buy_price = float(row["buy_price"])
            quantity = int(row["quantity"])
            target_pct = float(row.get("target_pct", 1.5))
            stop_pct = float(row.get("stop_pct", 1))

            print(f"📌 Calling get_live_price for {symbol}")
            live_price = get_live_price(symbol, security_id)
            print(f"✅ Live price for {symbol} = {live_price}")
            
            if live_price is None:
                print(f"⚠️ Failed to get live price for {symbol}")
                return row
            change_pct = ((live_price - buy_price) / buy_price) * 100
            log_live_trail(symbol, live_price, change_pct, order_id=row.get("order_id", ""))

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
                with portalocker.Lock(LIVE_BUFFER_FILE, 'r', timeout=5) as f:
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
            
            # EOD Auto-Exit at 3:25 PM
            current_time = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
            if current_time.hour == 15 and current_time.minute >= 25:
                reason = "EOD AUTO-EXIT"
                should_sell = True
            elif change_pct >= target_pct:
                reason = "TARGET HIT"
                should_sell = True
            # 🧠 Predictive peak logic using linear regression
            elif len(records) >= 15:
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
            elif is_peak_exhausted(symbol, target_pct=target_pct):
                reason = "SMART EXIT: Price exhausted near target multiple times"
                should_sell = True

            # ✅ Allow loss-side exit even if profit < min threshold
            is_loss_exit = reason in ["STOP LOSS", "FORCED EXIT: Max ₹ loss", "EOD AUTO-EXIT"]            
            # ✅ Smart SELL trigger: allow exit if price hit near target multiple times
            near_target_hits = [r for r in records if target_pct - 0.1 <= r[1] <= target_pct + 0.05]
            frequent_peaks = len(near_target_hits) >= 2
            
            if frequent_peaks:
                print(f"🔁 Smart SELL allowed for {symbol} due to repeated target hits ({len(near_target_hits)}x)")
            

            # ✅ Final guard: Never SELL unless a valid strategy has triggered
            if not should_sell or not reason:
                print(f"🚫 {symbol}: SELL skipped – No strategy condition met.")
                return row

            # ✅ SELL Allowed
            if net_profit >= MINIMUM_NET_PROFIT_REQUIRED or is_loss_exit or frequent_peaks:      
                attempted_security_ids.add(security_id)
                exchange_segment = "NSE_EQ"
                code, response = place_sell_order(security_id, symbol, quantity, exchange_segment)
                if code == 200:
                    order_id = str(response.get("order_id") or response.get("data", {}).get("orderId", ""))
                    trade_status = None
                    for _ in range(5):
                        trade_book = get_trade_book()
                        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                        matching_trades = [
                            t for t in trade_book
                            if t.get("symbol", "").upper() == symbol.upper()
                            and t.get("transactionType", "").upper() == "SELL"
                            and str(t.get("orderDateTime", "")).startswith(today_str)
                        ]
                        
                        if matching_trades:
                            matching_trades.sort(key=lambda x: x.get("orderDateTime", ""), reverse=True)
                            trade_status = matching_trades[0].get("status", "").upper()
                            if trade_status in ["TRADED", "COMPLETE"]:
                                break
                        systime.sleep(2)
                                
                    if trade_status in ["TRADED", "COMPLETE"]:
                        status = "PROFIT" if net_profit >= 0 else "STOP LOSS"
                        exit_price = live_price
                        log_sell(symbol, security_id, quantity, live_price, reason)
                        print(f"✅ SOLD {symbol} at ₹{live_price} ({reason}) Net Profit: ₹{round(net_profit, 2)}")
                        send_telegram_message(f"✅ SOLD {symbol} at ₹{live_price} ({reason}) Net Profit: ₹{round(net_profit, 2)}")
                        log_bot_action("portfolio_tracker.py", "SELL executed", "✅ TRADED", f"{symbol} @ ₹{round(live_price, 2)} | Reason: {reason}")
                    
                        # ✅ DB Update
                        insert_portfolio_log_to_db(
                            datetime.datetime.now(pytz.timezone("Asia/Kolkata")),
                            symbol,
                            security_id,
                            quantity,
                            buy_price,
                            stop_pct,
                            status=status,
                            exit_price=exit_price,
                            order_id=str(order_id)
                        )
                                          
                        # ✅ Trade summary with % and status
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
                    
                    elif trade_status in ["TRANSIT", "UNKNOWN"]:
                        print(f"⏳ Sell order for {symbol} is in status: {trade_status}. Will verify next run.")
                        log_bot_action("portfolio_tracker.py", "SELL pending", f"⏳ {trade_status}", f"{symbol} sell placed. Awaiting trade book confirmation.")
                        send_telegram_message(f"⏳ Sell order placed for {symbol} but status is '{trade_status}'. Rechecking later.")
                    
                        # Track pending security IDs instead of symbols
                        attempted_security_ids.add(security_id)
                        row.update({
                            "status": "HOLD",
                            "exit_price": "",
                            "live_price": live_price,
                            "change_pct": change_pct,
                            "last_checked": now
                        })
                    
                        # ✅ NEW GUARD: Block reprocessing of same order in next run
                        print(f"🛡️ {symbol} has unconfirmed SELL order {order_id}. Skipping reprocessing.")
                        return row
                    
                    else:
                        print(f"⚠️ Sell order placed but NOT TRADED yet for {symbol}. Holding.")
                    
                        # ✅ NEW: Don't block future retries — avoid adding to attempted_security_ids
                        pending_security_id = str(security_id)
                        if pending_security_id:
                            print(f"🕓 Delaying confirmation for {symbol} | security_id = {pending_security_id}")
                            row.update({
                                "status": "HOLD",
                                "exit_price": "",
                                "live_price": live_price,
                                "change_pct": change_pct,
                                "last_checked": now,
                                "security_id": pending_security_id
                            })
                        return row
                    
                else:
                    print(f"❌ SELL failed for {symbol}: {response}")
                    send_telegram_message(f"❌ SELL failed for {symbol}: {response}")
                    
                    order_id = response.get("data", {}).get("orderId", "")
                    if row.get("status") not in ["PROFIT", "STOP LOSS", "FORCE_EXIT"]:
                        row.update({
                            "status": "HOLD",
                            "exit_price": "",
                            "order_id": str(order_id) if order_id else ""
                        }) 
                    return row
                                   
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

            # ✅ Always update exit_price and audit data
            if status in ["PROFIT", "STOP LOSS", "FORCE_EXIT"]:
                row.update({
                    "live_price": round(live_price, 2),
                    "change_pct": round(change_pct, 2),
                    "last_checked": now,
                    "status": status,
                    "exit_price": round(live_price, 2)
                })
            else:
                row.update({
                    "live_price": round(live_price, 2),
                    "change_pct": round(change_pct, 2),
                    "last_checked": now,
                    "exit_price": round(net_profit, 2)
                })
            
            return row

        except Exception as e:
            print(f"⚠️ Error processing {row.get('symbol')}: {e}")
            return row

    # ✅ Execute all in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        # ✅ Only process active HOLD rows
        hold_rows = [r for r in existing_rows if r.get("status", "").upper() == "HOLD"]
        results = list(executor.map(process_sell_logic, hold_rows))
        updated_rows = [r for r in results if r]

    if updated_rows:
        # 🔁 Load original rows
        with open(PORTFOLIO_LOG, newline="", encoding="utf-8") as f:
            original = list(csv.DictReader(f))
    
        # 🧠 Build updated map
        updated_map = {row["symbol"]: row for row in updated_rows}
    
        # 🛡️ Merge updates into original rows
        final_rows = []
        for row in original:
            symbol = row.get("symbol")
            if symbol in updated_map and row.get("status") not in ["PROFIT", "STOP LOSS", "FORCE_EXIT"]:
                final_rows.append(updated_map[symbol])
            else:
                final_rows.append(row)           
    
        # ✅ Convert updated_rows to DataFrame
        updated_df = pd.DataFrame(updated_rows)
        
        # ✅ Read original file
        if os.path.exists(PORTFOLIO_LOG):
            original_df = pd.read_csv(PORTFOLIO_LOG)
        else:
            original_df = pd.DataFrame(columns=headers)
        
        # ✅ Merge by symbol — keep SOLD entries untouched
        original_df.set_index("symbol", inplace=True)
        updated_df.set_index("symbol", inplace=True)
        
        # ✅ Update only HOLD rows
        updated_df = updated_df[~updated_df["status"].isin(["PROFIT", "STOP LOSS", "FORCE_EXIT"])]
        for symbol in updated_df.index:
            if (
                symbol in original_df.index and 
                original_df.loc[symbol]["status"] not in ["PROFIT", "STOP LOSS", "FORCE_EXIT"]
            ):
                # Create temporary copy for dtype conversion
                updated_row = updated_df.loc[symbol].copy()
                for col in original_df.columns:
                    if col in updated_row.index:
                        try:
                            # ✅ Permanent fix: forcibly cast with fallback
                            updated_row[col] = pd.Series([updated_row[col]]).astype(original_df[col].dtype).iloc[0]
                        except Exception as dtype_err:
                            print(f"⚠️ Column '{col}' dtype cast failed. Skipping dtype alignment: {dtype_err}")
        
                original_df.loc[symbol] = updated_row
            elif symbol not in original_df.index:
                if updated_df.loc[symbol]["status"] not in ["PROFIT", "STOP LOSS", "FORCE_EXIT"]:
                    # Create temporary copy for dtype conversion
                    updated_row = updated_df.loc[symbol].copy()
                    for col in original_df.columns:
                        if col in updated_row.index:
                            try:
                                # ✅ Permanent fix: forcibly cast with fallback
                                updated_row[col] = pd.Series([updated_row[col]]).astype(original_df[col].dtype).iloc[0]
                            except Exception as dtype_err:
                                print(f"⚠️ Column '{col}' dtype cast failed. Skipping dtype alignment: {dtype_err}")
        
                    original_df.loc[symbol] = updated_row
        
        
        # ✅ Save back safely
        original_df.reset_index(inplace=True)
        original_df["order_id"] = original_df["order_id"].astype(str)
        original_df.to_csv(PORTFOLIO_LOG, index=False)
        print("✅ Portfolio safely updated without overwriting new entries.")

# ✅ Final log dump to file
try:
    with open("D:/Downloads/Dhanbot/dhan_autotrader/Logs/portfolio_tracker_log.txt", "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
except Exception as e:
    print(f"⚠️ Log write failed: {e}")    

if __name__ == "__main__":
    if os.path.exists("emergency_exit.txt"):
        send_telegram_message("⛔ Emergency Exit active. Skipping HOLD monitoring.")
        log_bot_action("portfolio_tracker.py", "SKIPPED", "EMERGENCY EXIT", "Monitoring skipped due to emergency exit.")
    else:
        check_portfolio()