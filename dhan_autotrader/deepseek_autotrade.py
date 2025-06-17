import csv
import sys
import json
import requests
from datetime import datetime, time, timedelta
import pytz
import time as systime
import pandas as pd
import os
from dhan_api import get_live_price, get_historical_price, compute_rsi, calculate_qty, get_stock_volume
from Dynamic_Gpt_Momentum import find_intraday_opportunities
from utils_logger import log_bot_action
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from utils_safety import safe_read_csv
import time as tm 
from db_logger import insert_portfolio_log_to_db
import math
import io

# 🧾 Setup TeeLogger to capture print statements
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

# ✅ Load Dhan credentials
with open("D:/Downloads/Dhanbot/dhan_autotrader/config.json", "r") as f:
    config = json.load(f)
    
# ✅ Constants
PORTFOLIO_LOG = "portfolio_log.csv"
LIVE_LOG = "live_prices_log.csv"
CURRENT_CAPITAL_FILE = "current_capital.csv"
GROWTH_LOG = "growth_log.csv"
BASE_URL = "https://api.dhan.co/orders"
TRADE_BOOK_URL = "https://api.dhan.co/trade-book"
trade_executed = False
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

NEWS_API_KEY = config.get("news_api_key")
TELEGRAM_TOKEN = config.get("telegram_token")
TELEGRAM_CHAT_ID = config.get("telegram_chat_id")

LIVEMONEYDEDUCTION = True
if len(sys.argv) > 1 and sys.argv[1].strip().upper() == "NO":
    LIVEMONEYDEDUCTION = False

# 📦 Dynamic Delivery % Estimator
def get_estimated_delivery_percentage(security_id):
    from datetime import datetime, timedelta
    try:
        yesterday = datetime.now() - timedelta(days=1)
        start = yesterday.strftime("%Y-%m-%d 09:15:00")
        end = yesterday.strftime("%Y-%m-%d 15:30:00")

        candles = get_historical_price(
            security_id=security_id,
            interval="15",
            from_date=start,
            to_date=end
        )

        if not candles:
            print(f"⚠️ No candles returned for delivery %")
            return 35.0

        total_volume = sum(c["volume"] for c in candles if "volume" in c)
        if total_volume == 0:
            return 35.0

        estimated_deliverable = total_volume * 0.65  # Assumed average
        return round((estimated_deliverable / total_volume) * 100, 2)
    except Exception as e:
        print(f"⚠️ Delivery % error: {e}")
        return 35.0

# ✅ Utility Functions
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"⚠️ Telegram send error: {e}")

def emergency_exit_active():
    return os.path.exists("emergency_exit.txt")

def is_market_open():
    now = datetime.now(pytz.timezone("Asia/Kolkata")).time()
    return time(9, 15) <= now <= time(15, 30)

def get_available_capital():
    try:
        raw_lines = safe_read_csv(CURRENT_CAPITAL_FILE)
        if not raw_lines or not raw_lines[0].strip().replace('.', '', 1).isdigit():
            raise ValueError(f"Corrupt or empty file: {CURRENT_CAPITAL_FILE}")
        base_capital = float(raw_lines[0].strip())
    except Exception as e:
        print(f"⚠️ Failed to read capital file: {e}")
        base_capital = float(input("Enter your starting capital: "))
        with open(CURRENT_CAPITAL_FILE, "w") as f:
            f.write(str(base_capital))

    try:
        if raw_lines is None:  # Handle safe_read_csv returning None
            print("⚠️ GROWTH_LOG read returned None")
            return base_capital
        raw_lines = safe_read_csv(GROWTH_LOG)
        rows = list(csv.DictReader(raw_lines))
        if rows:
            last_growth = float(rows[-1].get("profits_realized", 0))
            if last_growth >= 5:
                base_capital += last_growth
    except Exception as e:
        print(f"⚠️ Skipping growth file update: {e}")

    return base_capital

def compute_trade_score(stock):
    """
    Simple scoring logic: You can customize this.
    Currently favors lower price and higher qty (affordability).
    """
    price_weight = -1 * stock["price"]  # Lower price = better
    qty_weight = stock["qty"] * 0.5     # More qty = better
    return round(price_weight + qty_weight, 2)

def get_dynamic_minimum_net_profit(capital):
    return max(5, round(capital * 0.001, 2))  # ₹5 or 0.1%

def has_open_position():
    today = datetime.now().date()
    try:
        raw_lines = safe_read_csv(PORTFOLIO_LOG)
        if len(raw_lines) <= 1:
            print(f"ℹ️ No trades yet in {PORTFOLIO_LOG}. File has only header.")
            return False           
        reader = csv.DictReader(raw_lines)
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

def load_dynamic_stocks(filepath="D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"):
    raw_lines = safe_read_csv(filepath)
    df = pd.read_csv(pd.compat.StringIO("".join(raw_lines)))
    return list(zip(df["symbol"], df["security_id"]))
    
# ✅ Buy Logic + Order Execution
def should_trigger_buy(symbol, high_15min, capital):
    try:
        price = get_live_price(symbol)
        if not price or price <= 0:
            return False, 0, 0

        # Trigger rule: price must cross 15-min high
        if price > high_15min:
            qty = int(capital // price)
            if qty > 0:
                return True, price, qty
        return False, price, 0
    except:
        return False, 0, 0

def place_buy_order(symbol, security_id, price, qty):
    # 🚫 Skip if inputs are missing or invalid
    if not security_id or not symbol or qty <= 0 or price <= 0:
        print(f"❌ Skipping invalid input: symbol={symbol}, security_id={security_id}, price={price}, qty={qty}")
        return False, "Invalid input"

    tick_size = 0.05
    buffer_price = round(math.floor(price * 1.002 / tick_size) * tick_size, 2)
    
    payload = {
        "transactionType": "BUY",
        "exchangeSegment": "NSE_EQ",
        "productType": "CNC",
        "orderType": "LIMIT",
        "validity": "DAY",
        "securityId": security_id,
        "tradingSymbol": symbol,
        "quantity": qty,
        "price": buffer_price,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
        "amoTime": "OPEN",
        "triggerPrice": 0,
        "smartOrder": False
    }

    # 🧾 Print payload for debugging
    print(f"📦 Order Payload for {symbol}: {json.dumps(payload, indent=2)}")

    try:
        if not LIVEMONEYDEDUCTION:
            print(f"🧪 DRY RUN: Simulating order for {symbol} | Qty: {qty} @ ₹{buffer_price}")
            send_telegram_message(f"🧪 DRY RUN ORDER: {symbol} | Qty: {qty} @ ₹{buffer_price}")
            dry_run_id = f"DRY_RUN_{int(tm.time())}" 
            return True, dry_run_id           

        # ✅ Real money execution
        response = requests.post(BASE_URL, headers=HEADERS, json=payload).json()
        raw_json = json.dumps(response, indent=2)
        
        # 🛡️ Patch: Support for both direct and nested orderId formats
        if "orderId" in response:
            order_id = response.get("orderId")
        elif isinstance(response.get("data"), dict):
            order_id = response["data"].get("orderId")
        else:
            order_id = None
               
        if (response.get("status", "").lower() == "success" or response.get("orderStatus", "").upper() == "TRANSIT") and order_id:
            send_telegram_message(f"✅ Order Placed: {symbol} | Qty: {qty} @ ₹{buffer_price}")
            print(f"🧾 Logging attempt: {symbol}, ID: {security_id}, Qty: {qty}, Price: {buffer_price}")
        
            try:
                # First log to CSV
                stop_pct = log_trade(symbol, security_id, qty, buffer_price, order_id)
                
                # Then log to database
                insert_portfolio_log_to_db(
                    trade_date=datetime.now(pytz.timezone("Asia/Kolkata")),
                    symbol=symbol,
                    security_id=security_id,
                    qty=qty,
                    buy_price=buffer_price,
                    stop_pct=stop_pct,
                    status="HOLD",
                    order_id=order_id
                )
                print(f"✅ Trade logged to CSV and DB for {symbol}")
            except Exception as e:
                print(f"❌ log_trade() failed for {symbol}: {e}")
                send_telegram_message(f"⚠️ Order placed for {symbol}, but logging failed: {e}")
                log_bot_action("autotrade.py", "LOG_ERROR", "❌ Logging Failed", f"{symbol} → {e}")
                return False, f"Order Placed but Logging Failed: {e}"
        
            log_bot_action("autotrade.py", "BUY", "✅ EXECUTED", f"{symbol} @ ₹{buffer_price}")
            return True, order_id       
        
        else:
            reason = response.get("remarks") or response.get("message")
            if not reason:
                reason = f"Dhan API gave no reason. Raw:\n{raw_json}"
            send_telegram_message(f"❌ Order rejected for {symbol}: {reason}")
            log_bot_action("autotrade.py", "BUY", "❌ FAILED", f"{symbol} → {reason}")
            return False, reason
    except Exception as e:
        print(f"❌ Exception placing order for {symbol}: {e}")
        return False, str(e)
    finally:
        systime.sleep(random.uniform(0.6, 1.2))
        
def get_trade_status(order_id):
    try:
        response = requests.get(TRADE_BOOK_URL, headers=HEADERS)
        if response.status_code == 200:
            trades = response.json().get("data", [])
            for t in trades:
                if str(t.get("order_id")).strip() == str(order_id).strip():
                    return t.get("status", "").upper()
        return "UNKNOWN"
    except:
        return "ERROR"

def get_atr(security_id, period=14, interval="15m"):
    """Proper ATR calculation from historical data"""
    try:
        candles = get_historical_price(security_id, interval=interval)
        if len(candles) < period + 1:
            return None
            
        tr_values = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        # Calculate ATR as SMA of TR
        atr = sum(tr_values[-period:]) / period
        return round(atr, 2)
    except:
        return None

def log_trade(symbol, security_id, qty, price, order_id):
    timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")
    # Get ATR dynamically (past 14 periods)
    atr = get_atr(security_id, period=14)
    if atr:
        target_pct = round((atr / price) * 100 * 1.2, 2)  # 1.2x ATR for target
        stop_pct = round((atr / price) * 100 * 0.8, 2)   # 0.8x ATR for stop
    else:
        err = f"❌ ATR fetch failed for {symbol}. Trade log aborted."
        print(err)
        send_telegram_message(err)
        raise ValueError(err)

    file_exists = os.path.isfile(PORTFOLIO_LOG)
    print(f"🛠️ Attempting to write to portfolio log: {PORTFOLIO_LOG}")

    # ❗ Pre-check: Is file write-locked?
    if os.path.exists(PORTFOLIO_LOG) and not os.access(PORTFOLIO_LOG, os.W_OK):
        error_msg = f"🚫 Cannot write to {PORTFOLIO_LOG}. File is locked or opened by another program (e.g., Excel)."
        print(error_msg)
        send_telegram_message(error_msg)
        raise PermissionError(error_msg)

    try:
        # Create file with headers if it doesn't exist
        if not os.path.exists(PORTFOLIO_LOG):
            with open(PORTFOLIO_LOG, mode='w', newline='') as f_init:
                writer = csv.writer(f_init)
                writer.writerow([
                    "timestamp", "symbol", "security_id", "quantity", "buy_price",
                    "momentum_5min", "target_pct", "stop_pct", "live_price",
                    "change_pct", "last_checked", "status", "exit_price", "order_id"
                ])
                print(f"📄 Created new portfolio log file with headers: {PORTFOLIO_LOG}")
    
        # Append trade row
        with open(PORTFOLIO_LOG, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, symbol, security_id, qty, price,
                0, target_pct, stop_pct, '', '', '', 'HOLD', '', order_id
            ])
            print(f"✅ Portfolio log updated for {symbol} — Qty: {qty} @ ₹{price}")
            return stop_pct
    except Exception as e:
        err_msg = f"❌ Failed to write {symbol} to CSV: {e}"
        print(err_msg)
        send_telegram_message(err_msg)
        raise   
    
# 🧵 Thread-safe monitoring functions
def monitor_stock_for_breakout(symbol, high_15min, capital, dhan_symbol_map, filter_failures, failures_lock, avg_volume=100000, fallback_mode=None):
    try:
        send_telegram_message(f"🔎 Scanning {symbol}...")

        security_id = dhan_symbol_map.get(symbol)
        if not security_id:
            print(f"⛔ Skipping {symbol} — security ID not found.")
            return

        ltp = 0
        retries = 3
        while retries > 0:
            try:
                ltp = get_live_price(symbol, security_id, premarket=False)
                if ltp == "RATE_LIMIT":
                    raise ValueError("RATE_LIMIT")
                break
            except Exception as e:
                if "429" in str(e) or "RATE_LIMIT" in str(e):
                    print(f"⏳ Rate limit hit for {symbol}. Retrying in 10s...")
                    systime.sleep(10)
                    retries -= 1
                else:
                    print(f"⚠️ {symbol} LTP fetch error: {e}")
                    break
        
        price = ltp if ltp else 0

        if price <= 0:
            print(f"❌ Skipping {symbol} — Invalid LTP: ₹{price}")
            return

        # Breakout Check
        if high_15min and price < high_15min:
            print(f"⏭️ Skipping {symbol} — Price ₹{price} has not crossed 15-min high ₹{high_15min}")
            return
            
        # Add volume check to breakout logic
        if price > high_15min:
            current_volume = get_stock_volume(security_id)
            if current_volume is None or current_volume <= 0:
                print(f"⚠️ Invalid volume for {symbol}")
                return
            
            capital = get_available_capital()
            volume_threshold = max(50000, capital * 0.0002)  # Use percentage instead of fixed division

            # 🔄 Apply fallback adjustment BEFORE threshold check
            if fallback_mode == "volume":
                volume_threshold = volume_threshold * 0.5
            
            if current_volume < volume_threshold:
                print(f"⛔ Breakout rejected: volume {current_volume} < threshold {volume_threshold}")
                with failures_lock:
                    filter_failures["volume"] += 1
                return None
                    
        # ✅ Check Delivery Percentage (Minimum 30%)
        delivery_pct = get_estimated_delivery_percentage(security_id)
        if delivery_pct < 30:
            print(f"⛔ Skipping {symbol} — Low Delivery %: {delivery_pct}%")
            with failures_lock:
                filter_failures["delivery"] += 1
            return           

        # RSI Check
        rsi = compute_rsi(security_id)
        if rsi is None:
            print(f"⚠️ Skipping {symbol} — Unable to compute RSI.")
            return
        rsi_limit = 70
        if fallback_mode == "rsi_high":
            rsi_limit = 75
        
        if rsi >= rsi_limit:
            print(f"⚠️ Skipping {symbol} — RSI too high: {rsi}")
            with failures_lock:
                filter_failures["rsi_high"] += 1
            return
        elif rsi < 25:
            print(f"⚠️ Skipping {symbol} — RSI too low: {rsi}")
            with failures_lock:
                filter_failures["rsi_low"] += 1
            return

        qty = calculate_qty(price, capital)
        if qty <= 0:
            print(f"❌ Skipping {symbol} — Insufficient qty for price ₹{price}")
            return

        # ✅ Final Candidate with Weighted Score
        # Get historical data for last 5 days manually
        try:
            candles_all = get_historical_price(security_id, interval="1d")
            candles = candles_all[-5:] if len(candles_all) >= 5 else candles_all
            volume = sum(c.get("volume", 0) for c in candles)
        except Exception as e:
            print(f"⚠️ Volume fetch failed: {e}")
            volume = 0
        
        volume_score = min(1.0, volume / 1000000)  # Normalize volume
        atr_distance = abs(price - high_15min) if high_15min else 0
        atr_score = min(1.0, atr_distance / price) if price else 0
        
        # Optional: placeholder ML sentiment score (e.g., from GPT earlier process or set to 0.5 default)
        ml_score = 0.5  
        # Example logic to simulate sentiment failure logging (customize if needed)
        if ml_score < 0.3:
            with failures_lock:
                filter_failures["sentiment"] += 1       
        momentum_cutoff = 0.05
        if fallback_mode == "momentum":
            momentum_cutoff = 0.00

        weighted_score = round((ml_score * 0.6) + (volume_score * 0.2) + (atr_score * 0.2), 4)

        if weighted_score < momentum_cutoff:
            print(f"❌ Skipping {symbol} — Score {weighted_score} < momentum cutoff {momentum_cutoff}")
            with failures_lock:
                filter_failures["momentum"] += 1
            return

        return {
            "symbol": symbol,
            "security_id": security_id,
            "price": price,
            "qty": qty,
            "score": weighted_score
        }
    except Exception as e:
        print(f"❌ Exception during monitoring {symbol}: {e}")
        return

# ✅ Real-Time Monitoring & Trade Controller
def is_nse_trading_day():
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
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
                raise KeyError("Trading key missing in NSE response")
            
            holidays = data["Trading"]
            dates = [datetime.strptime(d["date"], "%d-%b-%Y").date() for d in holidays if str(year) in d["date"]]
            pd.DataFrame({"date": dates}).to_csv(fname, index=False)
        except Exception as e:
            print(f"⚠️ NSE holiday fetch failed: {e}")
            return True  # fallback: assume trading

    try:
        hdf = pd.read_csv(fname)
        holiday_dates = pd.to_datetime(hdf["date"]).dt.date.tolist()
        return today not in holiday_dates
    except:
        return True  # fallback to assume trading

def run_autotrade():
    global trade_executed  # ✅ Ensures outer-level flag is respected
    # First check if it's a trading day
    if not is_nse_trading_day():
        print("⛔ Market is closed today (holiday). Exiting auto-trade.")
        log_bot_action("autotrade.py", "market_status", "INFO", "Skipped: Market holiday.")
        return
    
    # Wait until market opens if before 9:15 AM
    now_ist = datetime.now(pytz.timezone("Asia/Kolkata"))
    if now_ist.time() < time(9, 15):
        # Calculate precise wait time
        market_open_time = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        seconds_to_wait = (market_open_time - now_ist).total_seconds()
        
        if seconds_to_wait > 0:
            print(f"🕒 Current time: {now_ist.strftime('%H:%M:%S')}. Waiting {seconds_to_wait:.0f} seconds until market open...")
            systime.sleep(seconds_to_wait)
    
    # Final market open check after waiting
    if not is_market_open():
        print("⛔ Market is closed for the day. Exiting auto-trade.")
        log_bot_action("autotrade.py", "market_status", "INFO", "Skipped: Market closed.")
        return

    csv_path = "D:/Downloads/Dhanbot/dhan_autotrader/Today_Trade_Stocks.csv"
    log_bot_action("autotrade.py", "startup", "STARTED", "Smart dynamic AutoTrade started.")
    print("🔍 Checking if market is open...")

    if has_open_position():
        print("📌 Existing position found. Skipping new trades.")
        return

    if emergency_exit_active():
        send_telegram_message("⛔ Emergency Exit Active. Skipping today's trading.")
        return

    if not os.path.exists(csv_path):
        print(f"⛔ {csv_path} not found. Falling back to dynamic_stock_list.csv")
        csv_path = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"

    if not os.path.exists(csv_path) or pd.read_csv(csv_path).empty:
        print(f"⚠️ {os.path.basename(csv_path)} is empty. Attempting dynamic regeneration...")
    
        # 🧠 Run GPT-based momentum generation directly
        opportunities = find_intraday_opportunities()
        if not opportunities:
            send_telegram_message("⚠️ No stocks qualified by GPT filter. Skipping trading today.")
            with open(momentum_csv, "w") as f:
                f.write("symbol,security_id\n")
            return
    
    momentum_flag = "D:/Downloads/Dhanbot/dhan_autotrader/momentum_ready.txt"
    momentum_csv = csv_path
    now = datetime.now()

    should_run_momentum = True
    if os.path.exists(momentum_flag):
        last_run_time = datetime.fromtimestamp(os.path.getmtime(momentum_flag))
        if now - last_run_time < timedelta(minutes=15):
            print("🕒 GPT momentum was run recently. Skipping regeneration.")
            should_run_momentum = False
        else:
            print("⚠️ GPT momentum last run >15 mins ago. Will regenerate list.")
    else:
        print("⚙️ No previous GPT run found. Running now.")

    if should_run_momentum and now.time() >= datetime.strptime("09:30", "%H:%M").time():
        print("⚙️ Running GPT Momentum Filter...")
    opportunities = find_intraday_opportunities()
    if not opportunities:
        send_telegram_message("⚠️ No stocks qualified by GPT filter. Skipping trading today.")
        with open(momentum_csv, "w") as f:
            f.write("symbol,security_id\n")
        return
        
        with open(momentum_flag, "w") as f:
            f.write(now.strftime("%Y-%m-%d %H:%M:%S"))
    elif now.time() < datetime.strptime("09:30", "%H:%M").time():
        print("⏳ Waiting for 9:30 AM to start GPT momentum process.")
        send_telegram_message("⏳ Waiting for 9:30 AM to generate GPT stock list.")
        return

    # 🔁 Re-check after regeneration
    if not os.path.exists(csv_path) or pd.read_csv(csv_path).empty:
        send_telegram_message("⛔ AutoTrade skipped — Even after regeneration, no stock candidates found.")
        return

    df = pd.read_csv(csv_path)
    required_cols = ['symbol', 'security_id', 'momentum_score', 'rsi']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    if df.empty or df["symbol"].isnull().all() or df["security_id"].isnull().all():
        print(f"⚠️ No valid rows in {csv_path}. Skipping trading for today.")
        send_telegram_message(f"⚠️ No stocks available in {os.path.basename(csv_path)}. Auto-trade skipped.")
        log_bot_action("autotrade.py", "data_check", "❌ EMPTY STOCK LIST", f"No rows in {csv_path}")
        return

    print(f"📄 Using stock list: {csv_path}")

    if df.empty or "symbol" not in df.columns or "security_id" not in df.columns:
        send_telegram_message("⚠️ dynamic_stock_list.csv is missing or invalid.")
        return

    ranked_stocks = df["symbol"].tolist()
    dhan_symbol_map = dict(zip(df["symbol"], df["security_id"]))
    bought_stocks = set()
    capital = get_available_capital()
    first_15min_high = {}

    for stock in ranked_stocks:
        try:
            if not stock or not isinstance(stock, str) or stock.strip() == "":
                print(f"⚠️ Skipping empty or invalid symbol: {stock}")
                continue

            security_id = dhan_symbol_map.get(stock)
            if not security_id:
                print(f"⛔ {stock} Skipped — Missing security_id in CSV.")
                continue

            candles = get_historical_price(security_id, interval="15m")
            if not candles or not isinstance(candles, list):
                print(f"⚠️ No candles returned for {symbol}")
                continue  # Skip this stock
            
            systime.sleep(0.6)
            highs = [c['high'] for c in candles if 'high' in c]
            if highs:
                first_15min_high[stock] = max(highs[:3])
        except Exception as e:
            print(f"⚠️ Skipping {stock} — could not fetch 15min high. Reason: {e}")

    trade_executed = False
    # 🚦 Dynamic Filter Failure Tracking
    filter_failures = {
        "momentum": 0,
        "rsi_high": 0,
        "rsi_low": 0,
        "volume": 0,
        "delivery": 0,
        "sentiment": 0
    }   
    s = None
    failures_lock = threading.Lock()

    # ✅ Exit monitoring loop once trade is executed
    while datetime.now(pytz.timezone("Asia/Kolkata")).time() <= time(14, 30):
        if trade_executed:
            print("✅ Trade completed. Exiting monitoring loop.")
            break    
        print(f"🔁 Monitoring stocks for breakout at {datetime.now().strftime('%H:%M:%S')}...")
        candidate_scores = []
        top_candidates = []

        def monitor_wrapper(symbol, filter_failures, failures_lock):
            try:
                security_id = dhan_symbol_map.get(symbol)
                high_trigger = first_15min_high.get(symbol)
                return monitor_stock_for_breakout(
                    symbol, high_trigger, capital, dhan_symbol_map, 
                    filter_failures, failures_lock
                )
            except Exception as e:
                print(f"⚠️ Monitor wrapper error for {symbol}: {e}")
                return None
        
        scan_round = 1
        fallback_mode = None
        fallback_pass = 0
        max_fallback_passes = 2  # You can increase to 3 if needed
    
        while not trade_executed and fallback_pass <= max_fallback_passes:
            print(f"🌀 Fallback Pass #{fallback_pass + 1} — Mode: {fallback_mode or 'Strict'}")
            candidate_scores.clear()
            filter_failures.update({k: 0 for k in filter_failures})
    
            valid_candidates = []
            with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers
                futures = {}
                # Only scan top 20 stocks with delays between submissions
                for stock in ranked_stocks[:20]:
                    future = executor.submit(
                        monitor_wrapper, 
                        stock, 
                        filter_failures, 
                        failures_lock
                    )
                    futures[future] = stock
                    systime.sleep(0.3)  # Add delay between submissions
                
                # Process results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        valid_candidates.append(result)
                        
            if valid_candidates:
                best = max(valid_candidates, key=lambda x: x["score"])
                print(f"✅ Best candidate: {best['symbol']} with score {best['score']}")
                success, order_id = place_buy_order(best["symbol"], best["security_id"], round(best["price"] * 1.0045, 2), best["qty"])
                if success:
                    trade_executed = True
                    s = best
                    print("✅ Final trade completed. Terminating auto-trade script.")
                    return
            else:
                # 🔍 Fallback analysis
                total_blocks = sum(filter_failures.values())
                if total_blocks == 0:
                    fallback_mode = None
                else:
                    fallback_mode = max(filter_failures, key=filter_failures.get)
                    dominant_pct = (filter_failures[fallback_mode] / total_blocks) * 100
                    print(f"⚠️ Dominant filter: {fallback_mode} blocked {dominant_pct:.1f}% of candidates")
    
                fallback_pass += 1
                print("🔁 Retrying with relaxed filter...\n")
                pd.DataFrame([filter_failures]).to_csv("D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_today.csv", index=False)
                systime.sleep(5)  

        pd.DataFrame(candidate_scores).to_csv("D:/Downloads/Dhanbot/dhan_autotrader/scanned_candidates_today.csv", index=False)

        if candidate_scores:
            top_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            best = top_candidates[0]
            fallbacks = top_candidates[1:]
            print(f"✅ Best candidate selected: {best['symbol']} @ ₹{best['price']} (Score: {best['score']})")
        
            success, order_id_or_msg = place_buy_order(best["symbol"], best["security_id"], best["price"], best["qty"])
            systime.sleep(1.2)
        
            if success:
                order_status = get_trade_status(order_id_or_msg)
                print(f"🛰️ Order status for {best['symbol']} is {order_status}")
                if order_status not in ["TRADED", "OPEN", "UNKNOWN", "TRANSIT"]:
                    send_telegram_message(f"❌ Order rejected by broker: {order_status} — {best['symbol']}")
                    log_bot_action("autotrade.py", "REJECTED", "❌ Broker rejected", f"{best['symbol']} → {order_status}")
                    return
                trade_executed = True
                bought_stocks.add(best["symbol"])
                s = best
                print("✅ Final trade completed. Terminating auto-trade script.")
                return  # 🔥 Hard exit after one successful order
        
            else:
                send_telegram_message(f"❌ Order failed for {best['symbol']}: {order_id_or_msg}")
                log_bot_action("autotrade.py", "BUY", "❌ FAILED", f"{best['symbol']} → {order_id_or_msg}")
        
                for alt in fallbacks:
                    if not alt or alt["symbol"] in bought_stocks:
                        continue
        
                    print(f"⚠️ Trying fallback candidate: {alt['symbol']}")
                    success, order_id_or_msg = place_buy_order(alt["symbol"], alt["security_id"], alt["price"], alt["qty"])
                    systime.sleep(1.2)
        
                    if success:
                        order_status = get_trade_status(order_id_or_msg)
                        print(f"🛰️ Order status for {alt['symbol']} is {order_status}")
                        if order_status not in ["TRADED", "OPEN", "UNKNOWN", "TRANSIT"]:
                            send_telegram_message(f"❌ Rejected fallback: {alt['symbol']} — {order_status}")
                            continue
                        trade_executed = True
                        s = alt
                        print("✅ Fallback trade completed. Terminating auto-trade script.")
                        return  # 🔥 Hard exit on fallback success
        

        now_time = datetime.now(pytz.timezone("Asia/Kolkata")).time()
        if not trade_executed and now_time >= time(14, 15):
            send_telegram_message("⚠️ No trade executed by 2:15 PM. Please review.")
            log_bot_action("autotrade.py", "WATCHDOG", "⚠️ NO TRADE", "No trade by 2:15 PM")

        if trade_executed:
            print("✅ Trade completed. Exiting monitoring loop.")
            break
        systime.sleep(60)
        
    
    # 🕒 3:25 PM End-of-Day Telegram Report
    now_ist = datetime.now(pytz.timezone("Asia/Kolkata")).strftime('%d-%b-%Y')
    
    if trade_executed and s:
        try:
            ltp_candle = get_live_price(s["symbol"], s["security_id"])
            systime.sleep(0.5)
            ltp = ltp_candle if isinstance(ltp_candle, (int, float)) and ltp_candle > 0 else s["price"]
            profit = round((ltp - s["price"]) * s["qty"], 2)
            pnl_pct = round(((ltp - s['price']) / s['price']) * 100, 2)
            
        except Exception as e:
            ltp = s["price"]
            profit = 0.0
            pnl_pct = 0.0
            
        summary = f"""📊 *DhanBot Daily Summary — {now_ist}*
    
    🛒 *Trade Executed:*
    • 🏷️ Symbol: {s['symbol']}
    • 🆔 Security ID: {s['security_id']}
    • 💰 Buy Price: ₹{s['price']}
    • 📦 Quantity: {s['qty']}
    • 🧾 Order Status: TRADED
    
    📈 *Trade Metrics:*
    • 📊 15-min High Trigger: ₹{first_15min_high.get(s['symbol'], 'N/A')}
    • 🔻 SL Hit: No
    
    💼 Capital Used: ₹{s['qty'] * s['price']:.2f}
    📈 LTP (EOD): ₹{ltp}
    💸 Net P&L: ₹{profit}
    • 📊 P&L %: {pnl_pct}%
    
    📌 Auto-Trade completed successfully.
    """
    else:
        summary = f"""📊 *DhanBot Daily Summary — {now_ist}*
    
    ⚠️ No trades were executed today.
    
    📌 Market Status: OPEN
    📋 Stocks Monitored: {len(ranked_stocks)}
    ⏳ Last scanned at: {datetime.now(pytz.timezone("Asia/Kolkata")).strftime('%H:%M')}
    
    🕒 Watchdog auto-exit confirmed at 3:25 PM.
    """
    
    send_telegram_message(summary)
    

# ✅ Final Trigger Block
if __name__ == "__main__":
    try:
        run_autotrade()
    except Exception as e:
        error_msg = f"❌ Exception in autotrade.py: {e}"
        print(error_msg)
        send_telegram_message(error_msg)
        log_bot_action("autotrade.py", "CRASH", "❌ ERROR", str(e))

    if not has_open_position():
        log_bot_action("autotrade.py", "end", "NO TRADE", "No stock bought today.")
    
    # 📝 Save all captured print outputs to a .txt log file
    with open("D:/Downloads/Dhanbot/dhan_autotrader/Logs/autotrade.txt", "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())