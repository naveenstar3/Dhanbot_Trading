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
from deepseek_Dynamic_Gpt_Momentum import find_intraday_opportunities
from utils_logger import log_bot_action
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from utils_safety import safe_read_csv
import time as tm 
from db_logger import insert_portfolio_log_to_db, log_to_postgres
import math
import io
from io import StringIO
from decimal import Decimal, ROUND_UP, ROUND_DOWN
from pathlib import Path

# ===== STDOUT LOGGER CONFIG =====
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
AUTOTRADE_LOG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/autotrade_log.txt"


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
        
# ✅ Trade Verification via Trade_book API
def verify_order_status_with_tradebook(security_id):
    try:
        from dhanhq import dhanhq, DhanContext
        with open("D:/Downloads/Dhanbot/dhan_autotrader/config.json") as f:
            config_data = json.load(f)

        context = DhanContext(config_data["client_id"], config_data["access_token"])
        dhan = dhanhq(context)

        # Check trade book
        trade_data = dhan.get_trade_book()
        for t in trade_data.get("data", []):
            if str(t.get("security_id")).strip() == str(security_id).strip():
                return t.get("status", "").upper()

        # Fallback to full order list
        order_data = dhan.get_order_list()
        for o in order_data.get("data", []):
            if str(o.get("security_id")).strip() == str(security_id).strip():
                return o.get("orderStatus", "").upper()

    except Exception as e:
        print(f"⚠️ Tradebook API verification failed: {e}")
    return "UNKNOWN"


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
        # ✅ Simplified capital file reading - single value in first cell
        with open(CURRENT_CAPITAL_FILE, "r") as f:
            capital_value = f.read().strip()
            if not capital_value.replace('.', '', 1).isdigit():
                raise ValueError(f"Invalid number in capital file: {capital_value}")
            return float(capital_value)
    except Exception as e:
        print(f"⚠️ Failed to read capital file: {e}")
        base_capital = float(input("Enter your starting capital: "))
        with open(CURRENT_CAPITAL_FILE, "w") as f:
            f.write(str(base_capital))
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
        if not raw_lines or len(raw_lines) <= 1:
            print(f"ℹ️ No trades yet in {PORTFOLIO_LOG}. File has only header.")
            return False           
        reader = csv.DictReader(raw_lines)
        for row in reader:
            if row.get("status", "").upper() in ["HOLD"]:
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
    df = pd.read_csv(StringIO("".join(raw_lines)))
    return list(zip(df["symbol"], df["security_id"]))

def verify_buy_order_in_trade_book(security_id, symbol=None):
    try:
        response = requests.get(TRADE_BOOK_URL, headers=HEADERS)
        if response.status_code == 200:
            trades = response.json().get("data", [])
            print(f"📘 Trade book fetched: {len(trades)} entries")

            for trade in trades:
                t_sec_id = str(trade.get("securityId", "")).strip()
                t_type = trade.get("transactionType", "").upper()
                if t_sec_id == str(security_id).strip() and t_type == "BUY":
                    print(f"✅ Verified via security_id: {t_sec_id}")
                    return True

            # 🌀 Fallback: Check by tradingSymbol match
            if symbol:
                print(f"⚠️ No match by security_id. Trying fallback by symbol={symbol}...")
                for trade in trades:
                    t_symbol = trade.get("tradingSymbol", "").strip().upper()
                    t_type = trade.get("transactionType", "").upper()
                    if t_symbol == symbol.strip().upper() and t_type == "BUY":
                        print(f"✅ Verified via fallback tradingSymbol: {t_symbol}")
                        return True

            print(f"❌ No BUY match found in trade book for security_id={security_id}, symbol={symbol}")
            return False
        else:
            print(f"❌ Trade book fetch failed. Status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"⚠️ Error verifying trade book for security_id={security_id}: {e}")
        return False
    
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
    price = round(price, 2)
    # ✅ Permanent fix: Exact tick size rounding using Decimal only
    tick_size = Decimal("0.05")
    raw_price = Decimal(str(price)) * Decimal("1.002")  # Add buffer for execution
    rounded_ticks = (raw_price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN)
    buffer_price_decimal = (rounded_ticks * tick_size).quantize(tick_size, rounding=ROUND_DOWN)
    buffer_price = float(round(buffer_price_decimal, 2))
    print(f"🔧 Tick-adjusted price for {symbol}: ₹{buffer_price} (raw input: ₹{price})")
    
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
    
        if order_id:
            # ✅ Wait 3 seconds to allow broker to register the trade
            print(f"⏳ Waiting 3 seconds for order to reflect in trade book...")
            systime.sleep(3)
            # ✅ Trade verification using security_id + transaction_type from trade_book
            verified = False
            for attempt in range(3):
                verified = verify_buy_order_in_trade_book(security_id, symbol)
                print(f"🛰️ TradeBook Check Attempt {attempt+1} for {symbol}: {verified}")
                if verified:
                    break
                systime.sleep(1.0)           
    
            if not verified:
                reason = "❌ Trade book verification failed: BUY entry not found"
                print(f"❌ Order verification failed for {symbol}. Reason: {reason}")
                send_telegram_message(f"❌ Order rejected for {symbol}: {reason}")
                log_bot_action("autotrade.py", "BUY", "❌ FAILED", f"{symbol} → {reason}")
                return False, reason
    
            print(f"✅ Order accepted by broker — Security ID: {security_id}")
            send_telegram_message(f"✅ Order Placed: {symbol} | Qty: {qty} @ ₹{buffer_price}")
            print(f"🧾 Logging attempt: {symbol}, ID: {security_id}, Qty: {qty}, Price: {buffer_price}")
    
            try:
                # First log to CSV
                stop_pct, target_pct = log_trade(symbol, security_id, qty, buffer_price, order_id)
    
                # Then log to database
                insert_portfolio_log_to_db(
                    trade_date=datetime.now(pytz.timezone("Asia/Kolkata")),
                    symbol=symbol,
                    security_id=security_id,
                    quantity=qty,
                    buy_price=buffer_price,
                    stop_pct=stop_pct,
                    target_pct=target_pct,
                    stop_price=round(buffer_price * (1 - stop_pct / 100), 2),
                    target_price=round(buffer_price * (1 + target_pct / 100), 2),
                    status="HOLD",
                    order_id=order_id
                )
                print(f"✅ Trade logged to CSV and DB for {symbol}")
            except Exception as e:
                print(f"❌ log_trade() failed for {symbol}: {e}")
                send_telegram_message(f"⚠️ Order placed for {symbol}, but logging failed: {e}")
                log_bot_action("autotrade.py", "LOG_ERROR", "❌ Logging Failed", f"{symbol} → {e}")
                return True, order_id
    
            log_bot_action("autotrade.py", "BUY", "✅ EXECUTED", f"{symbol} @ ₹{buffer_price}")
            now_ts = datetime.now()
            log_to_postgres(now_ts, "autotrade.py", "✅ EXECUTED", f"{symbol} @ ₹{buffer_price}")
            return True, order_id
    
        else:
            reason = response.get("remarks") or "❌ Order ID not returned by broker"
            print(f"❌ Order failed for {symbol}. Reason: {reason}")
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
    import pandas as pd

    timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")

    atr = get_atr(security_id, period=14)
    if not atr:
        print(f"⚠️ ATR fetch failed for {symbol}. Proceeding with NULL stop/target for logging.")
        send_telegram_message(f"⚠️ ATR fetch failed for {symbol}. Logging with NULL target_pct.")
    
        target_pct = None
        stop_pct = None
        target_price = None
        stop_price = None
    else:
        if atr:
            target_pct = round((atr / price) * 100 * 1.2, 2)
            stop_pct = round((atr / price) * 100 * 0.8, 2)
            # Align target/stop price with SELL script logic using raw buffer price
            base_price = Decimal(str(price))
            target_decimal = (base_price * (Decimal(target_pct) / 100)).quantize(Decimal("0.0001"), rounding=ROUND_UP)
            target_price = float((base_price + target_decimal).quantize(Decimal("0.05"), rounding=ROUND_UP))            
            stop_decimal = (base_price * (Decimal(stop_pct) / 100)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
            stop_price = float((base_price - stop_decimal).quantize(Decimal("0.05"), rounding=ROUND_DOWN))

    print(f"🛠️ Attempting to append to portfolio log: {PORTFOLIO_LOG}")

    if not os.access(os.path.dirname(PORTFOLIO_LOG) or ".", os.W_OK):
        error_msg = f"🚫 Cannot write to directory for {PORTFOLIO_LOG}."
        print(error_msg)
        send_telegram_message(error_msg)
        raise PermissionError(error_msg)

    try:
        new_row = pd.DataFrame([{
            "timestamp": timestamp,
            "symbol": symbol,
            "security_id": security_id,
            "quantity": qty,
            "buy_price": price,
            "momentum_5min": 0,
            "target_pct": target_pct,
            "stop_pct": stop_pct,
            "target_price": target_price,
            "stop_price": stop_price,
            "live_price": '',
            "change_pct": '',
            "last_checked": '',
            "status": 'HOLD',
            "exit_price": '',
            "order_id": str(order_id)
        }])

        if os.path.exists(PORTFOLIO_LOG):
            existing = pd.read_csv(PORTFOLIO_LOG)
            df = pd.concat([existing, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(PORTFOLIO_LOG, index=False)
        print(f"✅ Portfolio log updated for {symbol} — Qty: {qty} @ ₹{price}")
        return stop_pct, target_pct

    except Exception as e:
        err_msg = f"❌ Failed to update portfolio_log.csv: {e}"
        print(err_msg)
        send_telegram_message(err_msg)
        raise
 
def breakout_confirmed(security_id, high_15min):
    try:
        candles = get_historical_price(security_id, interval="1m")
        if not candles or len(candles) < 3:
            return False
        return (
            candles[-1]['close'] > high_15min and
            candles[-2]['close'] > high_15min
        )
    except Exception as e:
        print(f"⚠️ Breakout confirm failed for security_id {security_id}: {e}")
        return False
 
def is_safe_to_buy(symbol, price, security_id, rsi):
    try:
        # ✅ 1. Check latest 3-min candle shape
        df = get_intraday_df(security_id, interval="3minute", lookback=5)
        if df is None or df.empty or len(df) < 3:
            print(f"⚠️ Skipping {symbol} — No 3-min data for safety check.")
            return False

        latest = df.iloc[-1]
        candle_range = latest["high"] - latest["low"]
        candle_body = abs(latest["close"] - latest["open"])
        candle_body_ratio = candle_body / (candle_range + 0.01)

        if candle_body_ratio < 0.65:
            print(f"⛔ {symbol} — Weak 3-min candle (body ratio {candle_body_ratio:.2f})")
            return False

        # ✅ 2. Volume confirmation
        avg_vol = df["volume"].tail(4).iloc[:-1].mean()
        if latest["volume"] < 1.8 * avg_vol:
            print(f"⛔ {symbol} — Volume not convincing. Current: {latest['volume']}, Avg: {avg_vol}")
            return False

        # ✅ 3. RSI already passed earlier — just re-check
        if rsi > 68:
            print(f"⛔ {symbol} — RSI too high: {rsi}")
            return False

        # ✅ 4. MACD histogram check
        macd_data = get_macd_data(security_id, interval="5minute")
        if not macd_data or macd_data.get("histogram", 0) <= 0:
            print(f"⛔ {symbol} — MACD histogram not positive.")
            return False

        # ✅ 5. VWAP distance
        vwap = get_live_vwap(security_id)
        if vwap and (price - vwap) / price > 0.01:
            print(f"⛔ {symbol} — Overextended above VWAP. Price: {price}, VWAP: {vwap}")
            return False

        # ✅ 6. Sector trend (optional)
        sector_trend = get_sector_momentum(symbol)
        if not sector_trend:
            print(f"⛔ {symbol} — Sector trend weak or unknown.")
            return False

        return True

    except Exception as e:
        print(f"⚠️ {symbol} — Safety check error: {e}")
        return False

def get_intraday_df(security_id, interval="5minute", lookback=10):
    """
    Returns a Pandas DataFrame with OHLCV data for the given security_id and interval.
    Uses get_historical_price() internally.
    """
    try:
        interval_map = {
            "1minute": "1m",
            "3minute": "3m",
            "5minute": "5m",
            "15minute": "15m"
        }

        interval_str = interval_map.get(interval.lower(), "5m")

        candles = get_historical_price(security_id, interval=interval_str)
        if not candles or len(candles) == 0:
            return None

        # Take only last `lookback` candles
        candles = candles[-lookback:]

        df = pd.DataFrame(candles)
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    except Exception as e:
        print(f"❌ Error in get_intraday_df for {security_id}: {e}")
        return None

def get_live_vwap(security_id):
    df = get_intraday_df(security_id, interval="5minute", lookback=20)
    if df is None or df.empty:
        return None

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_x_volume"] = df["typical_price"] * df["volume"]
    vwap = df["tp_x_volume"].sum() / df["volume"].sum()
    return round(vwap, 2)

def get_macd_data(security_id, interval="5minute"):
    df = get_intraday_df(security_id, interval=interval, lookback=40)
    if df is None or df.empty:
        return None

    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["histogram"] = df["macd"] - df["signal"]

    latest = df.iloc[-1]
    return {
        "macd": round(latest["macd"], 4),
        "signal": round(latest["signal"], 4),
        "histogram": round(latest["histogram"], 4)
    }

def detect_bullish_pattern(df):
    """Detect bullish candle patterns in 5-minute data"""
    if df is None or len(df) < 3:
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Bullish Engulfing
    if prev['close'] < prev['open'] and last['close'] > last['open']:
        if last['close'] > prev['open'] and last['open'] < prev['close']:
            return True

    # Hammer
    body = abs(last['close'] - last['open'])
    lower_wick = last['open'] - last['low'] if last['close'] > last['open'] else last['close'] - last['low']
    upper_wick = last['high'] - max(last['close'], last['open'])
    if body > 0 and lower_wick > 2 * body and upper_wick < body:
        return True

    # Morning Star
    if prev2['close'] < prev2['open'] and last['close'] > last['open']:
        if prev['close'] < prev2['close'] and last['close'] > (prev2['open'] + prev2['close']) / 2:
            return True

    return False

def get_sector_momentum(symbol):
    sector_map = {
        "RELIANCE": "NIFTY_ENERGY",
        "SBIN": "NIFTY_BANK",
        "HDFCBANK": "NIFTY_BANK",
        "TCS": "NIFTY_IT",
        "MARUTI": "NIFTY_AUTO",
        "SHRIRAMFIN": "NIFTY_FIN_SERVICE",
    }

    sector = sector_map.get(symbol.upper())
    if not sector:
        return True  # assume true if unknown

    try:
        # Fetch sector index live % change — replace with real API if available
        index_data = get_index_change(sector)  # Assume this exists
        if index_data["percent_change"] > 0:
            return True
        else:
            return False
    except:
        return True  # fallback to true if API fails


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
        if price > high_15min and breakout_confirmed(security_id, high_15min):
            # Add delay before volume check API call
            systime.sleep(0.8)
        
            try:
                candles = get_historical_price(security_id, interval="15m")
                if not candles or not isinstance(candles, list):
                    print(f"⚠️ No candle data to compute volume for {symbol}")
                    return None
                current_volume = sum(c.get("volume", 0) for c in candles if "volume" in c)
            except Exception as e:
                if "429" in str(e) or "Rate_Limit" in str(e):
                    print(f"⏳ Volume API rate limit hit for {symbol}. Sleeping 15s...")
                    systime.sleep(15)
                    try:
                        candles = get_historical_price(security_id, interval="15m")
                        if not candles or not isinstance(candles, list):
                            print(f"⚠️ No candle data to compute volume for {symbol} after retry")
                            return None
                        current_volume = sum(c.get("volume", 0) for c in candles if "volume" in c)
                    except:
                        print(f"⚠️ Failed volume check after retry for {symbol}")
                        return None
                else:
                    print(f"⚠️ Volume check error for {symbol}: {e}")
                    return None
        
            if current_volume is None or current_volume <= 0:
                print(f"⚠️ Invalid volume for {symbol}")
                return
        
            # ✅ Dynamic Anti-Top Rejection (Protect against late entry at top)
            candle_df = get_intraday_df(security_id, interval="5minute", lookback=5)
            if candle_df is not None and len(candle_df) >= 3:
                last_3 = candle_df.tail(3)
                gains = (last_3["close"] - last_3["open"]) / last_3["open"]
                recent_spike = gains.max() > 0.013  # >1.3% spike in last 3 candles
        
                wick_ratio = ((last_3["high"] - last_3["close"]) / (last_3["high"] - last_3["low"] + 0.01)).max()
                wicky = wick_ratio > 0.6  # Long upper wick
        
                v1, v2, v3 = last_3["volume"].iloc[-3:]
                volume_drop = v3 < 0.7 * ((v1 + v2) / 2)
        
                if recent_spike or wicky or volume_drop:
                    print(f"⛔ {symbol} breakout rejected dynamically — spike={recent_spike}, wick={wicky}, volume_drop={volume_drop}")
                    with failures_lock:
                        filter_failures["dynamic_top"] = filter_failures.get("dynamic_top", 0) + 1
                    return None       
                    
            capital = get_available_capital()

            # 🚀 Realistic volume threshold based on capital
            base_threshold = max(10000, capital * 0.05)  # 5% of capital, min 10k
            # Add time-based relaxation
            if datetime.now().time() > time(11, 0):
                volume_threshold = base_threshold * 0.5  # 50% relaxation after 11AM
            else:
                volume_threshold = base_threshold * 0.7  # 30% relaxation
            print(f"🧠 Adjusted volume threshold: {volume_threshold} (Base: {base_threshold})")
            
            if fallback_mode == "volume":
                volume_threshold = volume_threshold * 0.5
                
            if current_volume < volume_threshold:
                print(f"⛔ Breakout rejected: volume {current_volume} < threshold {volume_threshold}")
                print(f"🧪 VOLUME FAIL: {symbol} | CurVol={current_volume} | Thr={volume_threshold} | Cap={capital}")
                with failures_lock:
                    filter_failures["volume"] += 1
                return None
                            # 🧪 DEBUG: Print volume calculation
            print(f"🧮 Volume Calc: Cap={capital} → "
                  f"BaseThresh={max(50000, capital * 0.0002)} → "
                  f"AdjThresh={volume_threshold}")
        # ✅ Check Delivery Percentage (Minimum 30%)
        delivery_pct = get_estimated_delivery_percentage(security_id)
        if delivery_pct < 25:
            print(f"⛔ Skipping {symbol} — Low Delivery %: {delivery_pct}%")
            print(f"🧪 DELIVERY FAIL: {symbol} | Delivery%={delivery_pct}")
            with failures_lock:
                filter_failures["delivery"] += 1
            return           

        # RSI Check
        rsi = compute_rsi(security_id)
        systime.sleep(1.0)  # ✅ Rate-limit protection after RSI fetch
        if rsi is None:
            print(f"⚠️ Skipping {symbol} — Unable to compute RSI.")
            return
        rsi_limit = 70
        if fallback_mode == "rsi_high":
            rsi_limit = 75
        
        # Allow high RSI if price is near breakout level
        if rsi >= rsi_limit:
            if price > (high_15min * 0.995):  # Within 0.5% of high
                print(f"⚠️ High RSI but near breakout: {symbol} ({rsi})")
            else:
                print(f"⚠️ Skipping {symbol} — RSI too high: {rsi}")
                with failures_lock:
                    filter_failures["rsi_high"] += 1
                return
        elif rsi < 25:
            print(f"⚠️ Skipping {symbol} — RSI too low: {rsi}")
            with failures_lock:
                filter_failures["rsi_low"] += 1
            return

        # Ensure minimum 1 lot purchase
        qty = max(1, calculate_qty(price, capital))
        if qty <= 0:
            print(f"❌ Skipping {symbol} — Insufficient qty for price ₹{price}")
            return

        # ✅ Final Candidate with Weighted Score
        # Get current volume instead of historical sum
        try:
            # Get current volume from live data
            current_volume = get_stock_volume(security_id)
        except Exception as e:
            print(f"⚠️ Current volume fetch failed: {e}")
            current_volume = 0
        
        # Normalize volume - use log scale for better distribution
        log_val = math.log10(max(1, current_volume))
        volume_score = min(1.0, log_val / 6.0)   # 1M volume = 1.0
        
        # Use breakout strength instead of ATR distance
        breakout_strength = (price - high_15min) / price if price > 0 and high_15min else 0
        breakout_score = min(1.0, breakout_strength * 100)  # 1% breakout = 1.0
        
        # ✅ Robust momentum scoring with fallback
        try:
            # Use actual momentum score if available
            ml_score = df[df['symbol']==symbol]['momentum_score'].values[0]
        except:
            # Fallback to RSI-based estimation if momentum missing
            ml_score = max(0.4, min(0.7, (70 - rsi)/50))  # Map RSI 20→0.7, 70→0.4
            print(f"⚠️ Used RSI fallback score for {symbol}: {ml_score:.2f}")
        
        # Momentum scoring weights
        momentum_weight = 0.5
        volume_weight = 0.3
        breakout_weight = 0.2
        
        weighted_score = round(
            (ml_score * momentum_weight) + 
            (volume_score * volume_weight) + 
            (breakout_score * breakout_weight), 
            4
        )

        # 🧪 DEBUG: Print scoring details
        print(f"🧠 {symbol} Scoring: "
              f"Momentum={ml_score:.2f}*{momentum_weight}, "
              f"Volume={volume_score:.2f}*{volume_weight}, "
              f"Breakout={breakout_score:.2f}*{breakout_weight} → "
              f"Total={weighted_score:.4f}")
        

        # Time-based momentum relaxation
        momentum_cutoff = 0.4
        if fallback_mode == "momentum":
            momentum_cutoff = 0.3
            
        # After 11AM, relax momentum cutoff
        if datetime.now().time() > time(11, 0):
            momentum_cutoff *= 0.8  # 20% relaxation
            print(f"🕒 Relaxing momentum cutoff to {momentum_cutoff:.2f} after 11AM")
            
        if weighted_score < momentum_cutoff:
            print(f"❌ Skipping {symbol} — Score {weighted_score} < momentum cutoff {momentum_cutoff}")
            with failures_lock:
                filter_failures["momentum"] += 1
            return

               # ✅ Volatility Filter — Reject stocks with large noisy candle swings
        try:
            recent_candles = get_historical_price(security_id, interval="5m")
            if recent_candles and len(recent_candles) >= 3:
                last_candle = recent_candles[-1]
                body_size = abs(last_candle["close"] - last_candle["open"])
                range_size = last_candle["high"] - last_candle["low"]
                if range_size > 0:
                    wick_ratio = body_size / range_size
                    if wick_ratio < 0.3:  # Candle with huge wicks and small body
                        print(f"⛔ Volatility reject: {symbol} | Wick Ratio={wick_ratio:.2f} (Noisy candle)")
                        with failures_lock:
                            filter_failures["volatility"] += 1
                        return None
        except Exception as e:
            print(f"⚠️ Volatility check error for {symbol}: {e}")
        
        # ✅ Final breakout confirmation before approving candidate
        candle_df = get_intraday_df(security_id, interval="5minute", lookback=5)
        if not detect_bullish_pattern(candle_df):
            print(f"⛔ Pattern Rejected: No bullish confirmation for {symbol}")
            with failures_lock:
                filter_failures["pattern"] = filter_failures.get("pattern", 0) + 1
            return None
       
        if not is_safe_to_buy(symbol, price, security_id, rsi):
            print(f"⛔ Skipping {symbol} — Failed safe-to-buy filter.")
            with failures_lock:
                filter_failures["final_check_failed"] = filter_failures.get("final_check_failed", 0) + 1
            return None
        
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
        
def get_rolling_high(security_id, window=3):
    """Get highest price from last N candles"""
    try:
        candles = get_historical_price(security_id, interval="15m")
        if not candles or len(candles) < window:
            return None
        return max(c['high'] for c in candles[-window:])
    except Exception as e:
        print(f"⚠️ Rolling high error: {e}")
        return None
        

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
    fallback_csv = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"
    log_bot_action("autotrade.py", "startup", "STARTED", "Smart dynamic AutoTrade started.")
    print("🔍 Checking if market is open...")

    if has_open_position():
        print("📌 Existing position found. Skipping new trades.")
        trade_executed = True
        return

    if emergency_exit_active():
        send_telegram_message("⛔ Emergency Exit Active. Skipping today's trading.")
        return

    # ✅ Robust CSV file validation with empty file handling
    valid_csv = False
    selected_path = csv_path
    
    # Check primary file
    if os.path.exists(csv_path):
        try:
            if os.path.getsize(csv_path) > 10:  # Minimum file size check
                test_df = pd.read_csv(csv_path)
                if not test_df.empty and 'symbol' in test_df.columns:
                    valid_csv = True
                    print(f"✅ Using valid primary CSV: {csv_path}")
        except Exception as e:
            print(f"⚠️ Primary CSV read error: {str(e)}")
    
    # Check fallback file if primary fails
    if not valid_csv and os.path.exists(fallback_csv):
        try:
            if os.path.getsize(fallback_csv) > 10:
                test_df = pd.read_csv(fallback_csv)
                if not test_df.empty and 'symbol' in test_df.columns:
                    valid_csv = True
                    selected_path = fallback_csv
                    print(f"✅ Using valid fallback CSV: {fallback_csv}")
        except Exception as e:
            print(f"⚠️ Fallback CSV read error: {str(e)}")
    
        # Regenerate if both files are invalid
    if not valid_csv:
        print("⚠️ Both stock lists missing/invalid. Regenerating via GPT...")
        opportunities = find_intraday_opportunities()
        
        if not isinstance(opportunities, list):
            print(f"⚠️ GPT returned non-list type: {type(opportunities)}. Forcing fallback.")
            opportunities = []
    
        if not opportunities:
            # 🚨 CRITICAL FIX: Use fallback if GPT returns empty or invalid
            if os.path.exists(fallback_csv):
                print(f"⚠️ GPT returned 0 stocks. Using fallback: {fallback_csv}")
                selected_path = fallback_csv
                valid_csv = True
            else:
                send_telegram_message("⚠️ No stocks qualified by GPT filter. Skipping trading today.")
                return
        else:
            # Create new valid CSV only if opportunities exist
            try:
                opp_df = pd.DataFrame(opportunities)
                opp_df.to_csv(csv_path, index=False)
                selected_path = csv_path
                print(f"✅ Regenerated stock list with {len(opportunities)} entries")
                valid_csv = True
            except Exception as e:
                print(f"❌ Error saving regenerated stock list: {e}")
                if os.path.exists(fallback_csv):
                    print(f"⚠️ Fallback to: {fallback_csv}")
                    selected_path = fallback_csv
                    valid_csv = True
                else:
                    send_telegram_message(f"❌ GPT stock list save failed and no fallback. Skipping trade. Error: {e}")
                    return
    

    # Proceed with momentum flag logic using validated CSV
    momentum_flag = "D:/Downloads/Dhanbot/dhan_autotrader/momentum_ready.txt"
    momentum_csv = selected_path
    now = datetime.now()

    should_run_momentum = True
    momentum_csv = "D:/Downloads/Dhanbot/dhan_autotrader/Today_Trade_Stocks.csv"
    should_run_momentum = True
    
    try:
        if os.path.exists(momentum_csv):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(momentum_csv))
            now = datetime.now()
            file_age_minutes = (now - file_mtime).total_seconds() / 60
    
            with open(momentum_csv, 'r') as f:
                header_check = f.readline().strip()
                if not header_check or ',' not in header_check:
                    raise ValueError("CSV has no columns")
            
            df_check = pd.read_csv(momentum_csv)
            
            if not df_check.empty and file_age_minutes <= 15:
                print("🕒 GPT momentum CSV is fresh. Skipping regeneration.")
                should_run_momentum = False
            else:
                print("⚠️ GPT momentum CSV is outdated or empty. Will regenerate.")
        else:
            print("⚙️ No previous GPT momentum CSV found. Running now.")
    except Exception as e:
        print(f"⚠️ Error checking GPT momentum CSV age: {e}")
        should_run_momentum = True

    # ✅ Preserve original momentum generation logic
    if should_run_momentum and now.time() >= time(9, 30):
        print("⚙️ Running GPT Momentum Filter...")
        try:
            opportunities = find_intraday_opportunities()
            if not isinstance(opportunities, list):
                print(f"⚠️ GPT returned unexpected type: {type(opportunities)}. Forcing fallback.")
                opportunities = []           
            # Ensure minimum 15 stocks in watchlist
            if len(opportunities) < 5:
               print(f"⚠️ Only {len(opportunities)} stocks. Adding fallback candidates")
               fallback_stocks = load_dynamic_stocks()
               # Convert tuples to dictionaries to match opportunities structure
               fallback_dicts = [{'symbol': s, 'security_id': sid} for s, sid in fallback_stocks]
               # Add 10 random stocks from fallback list
               opportunities.extend(random.sample(fallback_dicts, min(10, len(fallback_dicts))))
            if not opportunities:
                print("⚠️ GPT returned 0 opportunities. Using fallback list")
                # Load fallback list
                df_fallback = pd.read_csv(fallback_csv)
                df_fallback.to_csv(csv_path, index=False)
                selected_path = csv_path
            else:
                pd.DataFrame(opportunities).to_csv(csv_path, index=False)
                selected_path = csv_path
                print(f"✅ Updated stock list with {len(opportunities)} entries")
            
            # Update momentum flag
            with open(momentum_flag, "w") as f:
                f.write(now.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            print(f"⚠️ GPT momentum failed: {e}. Using fallback CSV")
            log_bot_action("autotrade.py", "GPT_ERROR", str(e), "Using fallback list")
    elif now.time() < time(9, 30) and should_run_momentum:
        print("⏳ Waiting for 9:30 AM to start GPT momentum process.")
        send_telegram_message("⏳ Waiting for 9:30 AM to generate GPT stock list.")
        # Calculate wait time
        wait_until = now.replace(hour=9, minute=30, second=0, microsecond=0)
        seconds_to_wait = (wait_until - now).total_seconds()
        if seconds_to_wait > 0:
            systime.sleep(seconds_to_wait)
        # Run momentum after waiting
        print("⚙️ Running GPT Momentum Filter after wait...")
        opportunities = find_intraday_opportunities()
        if opportunities:
            pd.DataFrame(opportunities).to_csv(csv_path, index=False)
            selected_path = csv_path
            with open(momentum_flag, "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Final validation before proceeding
    try:
        df = pd.read_csv(selected_path)
        if df.empty:
            raise ValueError("DataFrame is empty after reading")
    except Exception as e:
        send_telegram_message(f"⛔ Critical error with stock list: {str(e)}")
        log_bot_action("autotrade.py", "data_error", "CRITICAL", f"Stock list invalid: {str(e)}")
        return
    
    # ✅ Robust column validation with dynamic calculation
    required_cols = ['symbol', 'security_id']
    missing_essential = [col for col in required_cols if col not in df.columns]
    if missing_essential:
        raise ValueError(f"Missing essential columns in CSV: {missing_essential}")
    
    # First ensure RSI exists (calculate dynamically if missing)
    if 'rsi' not in df.columns:
        print("⚠️ rsi column missing. Calculating dynamically...")
        # Calculate RSI for each stock in real-time
        df['rsi'] = df.apply(lambda row: compute_rsi(row['security_id']) or 50, axis=1)
    
    # Then calculate momentum_score if missing
    if 'momentum_score' not in df.columns or df['momentum_score'].isnull().all():
        print("🚫 No valid ML momentum score available. Aborting trade run.")
        return   
    
    if df.empty:
        print(f"⚠️ No valid rows in {selected_path}. Skipping trading for today.")
        send_telegram_message(f"⚠️ No stocks available in {os.path.basename(selected_path)}. Auto-trade skipped.")
        log_bot_action("autotrade.py", "data_check", "❌ EMPTY STOCK LIST", f"No rows in {selected_path}")
        return

    # Filter out invalid symbols
    df = df[df['symbol'].notna() & df['security_id'].notna()]
    
    print(f"📄 Using stock list: {selected_path} with {len(df)} valid entries")
    print(f"🔍 First 5 stocks: {df[['symbol', 'rsi', 'momentum_score']].head().to_dict('records')}")

    ranked_stocks = df["symbol"].tolist()
    dhan_symbol_map = dict(zip(df["symbol"], df["security_id"]))
    bought_stocks = set()
    capital = get_available_capital()
    first_15min_high = {}
    # ✅ Periodic refresh of 15-min highs (top 10 only)
    def refresh_highs():
        """Refresh 15-min highs for top 10 stocks in watchlist"""
        now_time = datetime.now(pytz.timezone("Asia/Kolkata")).time()
        if now_time >= time(14, 45):
            print("⏰ Skipping refresh - too close to market close")
        return
        top_stocks = ranked_stocks[:10]   # Only top 10
        n = len(top_stocks)
        print(f"■■ Refreshing 15-min highs for top {n} stocks...")
        for i, stock in enumerate(top_stocks):
            security_id = dhan_symbol_map.get(stock)
            if security_id:
                try:
                    candles = get_historical_price(security_id, interval="15m")
                    if candles and isinstance(candles, list):
                        # Get the last 3 candles instead of first 3
                        if len(candles) >= 3:
                            last_three_highs = [candle['high'] for candle in candles[-3:]]
                            new_high = max(last_three_highs)
                            first_15min_high[stock] = new_high
                            print(f"▲ Updated high for {stock}: ₹{new_high}")
                        else:
                            # If we don't have 3 candles, take the max of what we have
                            highs = [candle['high'] for candle in candles if 'high' in candle]
                            if highs:
                                new_high = max(highs)
                                first_15min_high[stock] = new_high
                                print(f"▲ Updated high (partial) for {stock}: ₹{new_high}")
                except Exception as e:
                    print(f"▲ Error refreshing high for {stock}: {e}")
            # Add a delay to avoid rate limiting, except for the last stock
            if i < n - 1:
                delay = random.uniform(0.5, 1.0)
                systime.sleep(delay)
        print("■■ Highs refreshed")
    
    # ✅ Add initial delay to prevent burst API requests
    systime.sleep(1.0)
    
    
    # 🔄 Process stocks with progressive delays
    for i, stock in enumerate(ranked_stocks):
        try:
            if not stock or not isinstance(stock, str) or stock.strip() == "":
                print(f"⚠️ Skipping empty or invalid symbol: {stock}")
                continue

            security_id = dhan_symbol_map.get(stock)
            if not security_id:
                print(f"⛔ {stock} Skipped — Missing security_id in CSV.")
                continue

            # 🕒 Add progressive delay - increases with each stock
            delay = 0.8 + (i * 0.1)
            systime.sleep(min(delay, 2.0))  # Cap at 2 seconds max
            
            candles = get_historical_price(security_id, interval="15m")
            if not candles or not isinstance(candles, list):
                print(f"⚠️ No candles returned for {stock}")
                continue  # Skip this stock
            
            rolling_high = get_rolling_high(security_id)
            if rolling_high:
                first_15min_high[stock] = rolling_high
                
        except Exception as e:
            if "429" in str(e) or "Rate_Limit" in str(e):
                print(f"⏳ Rate limit hit. Sleeping for 10 seconds before retrying {stock}...")
                systime.sleep(10)
                # Retry this stock after delay
                try:
                    candles = get_historical_price(security_id, interval="15m")
                    if candles and isinstance(candles, list):
                        rolling_high = get_rolling_high(security_id)
                        if rolling_high:
                            first_15min_high[stock] = rolling_high
                except:
                    print(f"⚠️ Skipping {stock} — failed after rate limit recovery")
            else:
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
    # Initialize last refresh time
    last_refresh_time = datetime.now(pytz.timezone("Asia/Kolkata"))

    # ✅ Exit monitoring loop once trade is executed
    while datetime.now(pytz.timezone("Asia/Kolkata")).time() <= time(14, 30):
        # ✅ Periodic high refresh every 15 minutes
        current_time = datetime.now(pytz.timezone("Asia/Kolkata"))
        if (current_time - last_refresh_time).total_seconds() >= 900:  # 15 minutes = 900 seconds
            refresh_highs()
            last_refresh_time = current_time
            
        if trade_executed:
            print("✅ Trade completed. Exiting monitoring loop.")
            break    
        print(f"🔁 Monitoring stocks for breakout at {datetime.now().strftime('%H:%M:%S')}...")
        candidate_scores = []
        top_candidates = []

        def monitor_wrapper(symbol, filter_failures, failures_lock, fallback_mode=None):
            try:
                security_id = dhan_symbol_map.get(symbol)
                high_trigger = first_15min_high.get(symbol)
                return monitor_stock_for_breakout(
                    symbol, high_trigger, capital, dhan_symbol_map, 
                    filter_failures, failures_lock, fallback_mode
                )
        
            except Exception as e:
                print(f"⚠️ Monitor wrapper error for {symbol}: {e}")
                return None
        
        scan_round = 1
        fallback_mode = None
        fallback_pass = 0
        max_fallback_passes = 3  # You can increase to 3 if needed
    
        while not trade_executed and fallback_pass <= max_fallback_passes:
            print(f"🌀 Fallback Pass #{fallback_pass + 1} — Mode: {fallback_mode or 'Strict'}")
            candidate_scores.clear()
            filter_failures.update({k: 0 for k in filter_failures})
    
            valid_candidates = []
            batch_size = 10
            total_stocks = len(ranked_stocks)
            num_batches = math.ceil(total_stocks / batch_size)
            current_batch_index = fallback_pass % num_batches
            batch_start = current_batch_index * batch_size
            batch_end = min(batch_start + batch_size, total_stocks)
            batch_symbols = ranked_stocks[batch_start:batch_end]
            
            print(f"📦 Batch Range: {batch_start} to {batch_end - 1} (Total Stocks: {total_stocks})")
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = {}
                for stock in batch_symbols:
                    delay = random.uniform(1, 1.5)
                    systime.sleep(delay)
            
                    future = executor.submit(monitor_wrapper, stock, filter_failures, failures_lock, fallback_mode)
                    futures[future] = stock
            
                
                # Process results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        valid_candidates.append(result)
                        
            if valid_candidates:
                best = max(valid_candidates, key=lambda x: x["score"])
                # ✅ Smart Market Time Check (skip trade if after 2:45 PM IST)
                now_time = datetime.now(pytz.timezone("Asia/Kolkata")).time()
                cutoff_time = time(14, 45)
                if now_time >= cutoff_time:
                    print(f"⛔ Market time is {now_time.strftime('%H:%M')}. Skipping trade (cutoff is 14:45 IST).")
                    send_telegram_message(f"⚠️ Trade skipped — Not enough market time left after {now_time.strftime('%H:%M')}")
                    break  # or use return if inside a function
                
                print(f"✅ Best candidate: {best['symbol']} with score {best['score']}")
                success, order_id = place_buy_order(best["symbol"], best["security_id"], best["price"], best["qty"])
                if success:
                    trade_executed = True
                    s = best
                    print("✅ Order placement attempted. Terminating auto-trade script.")
                    send_telegram_message(f"🚀 Order attempt for {best['symbol']} completed. Terminating script.")
                    sys.exit(0)
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
                
                # ➕ ADDED NEAR-BREAKOUT FALLBACK
                if fallback_pass == max_fallback_passes:  # Only on last pass
                    print("⚠️ Last fallback pass. Trying near-breakout stocks...")
            
                    try:
                        print("🔄 Reloading stock candidates from CSV due to inactivity...")
                        df = pd.read_csv(selected_path)
                        ranked_stocks = df["symbol"].tolist()
                        dhan_symbol_map = dict(zip(df["symbol"], df["security_id"]))
                        fallback_pass = 0  # reset fallback cycle
                        continue  # restart fallback cycle with reloaded list
                    except Exception as e:
                        print(f"❌ Failed to reload stocks from CSV: {e}")
                        send_telegram_message(f"❌ Stock reload failed: {e}")
                        log_bot_action("autotrade.py", "CRASH", "Stock reload failure", str(e))
                        break

                    near_breakout_stocks = []
                    
                    for stock in ranked_stocks[:10]:  # Only check top 10
                        security_id = dhan_symbol_map.get(stock)
                        if not security_id:
                            continue
                            
                        price = get_live_price(stock, security_id)
                        systime.sleep(1.0)  # ✅ Rate-limit protection after live price fetch
                        high = first_15min_high.get(stock)
                        
                        if (
                            price and high and price > (high * 0.998) and
                            compute_rsi(security_id) < 65 and
                            get_estimated_delivery_percentage(security_id) >= 35
                        ):
                            volume = get_stock_volume(security_id)
                            if volume < 100000:
                                print(f"❌ Skipping fallback {stock} — low volume: {volume}")
                                continue
                            
                        
                            print(f"⚠️ Near-breakout candidate: {stock} (Price: {price}, High: {high})")
                            qty = calculate_qty(price, capital)
                            if qty > 0:
                                near_breakout_stocks.append({
                                    "symbol": stock,
                                    "security_id": security_id,
                                    "price": price,
                                    "qty": qty,
                                    "score": 0.5  # Medium priority
                                })
                    
                    if near_breakout_stocks:
                        best_near = max(near_breakout_stocks, key=lambda x: x["price"]/x["qty"])  # Best value
                        print(f"✅ Best near-breakout: {best_near['symbol']}")
                        success, order_id = place_buy_order(
                            best_near["symbol"], 
                            best_near["security_id"], 
                            best_near["price"], 
                            best_near["qty"]
                        )
                        if success:
                            trade_executed = True
                            s = best_near
                            print("✅ Near-breakout trade executed. Terminating.")
                            sys.exit(0)
                
                print("🔁 Retrying with relaxed filter...\n")
                pd.DataFrame([filter_failures]).to_csv("D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_today.csv", index=False)
                systime.sleep(3) 

        pd.DataFrame(candidate_scores).to_csv("D:/Downloads/Dhanbot/dhan_autotrader/scanned_candidates_today.csv", index=False)

        if candidate_scores:
            top_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            best = top_candidates[0]
            fallbacks = top_candidates[1:]
            print(f"✅ Best candidate selected: {best['symbol']} @ ₹{best['price']} (Score: {best['score']})")
        
            # Prevent re-buy
            bought_stocks.add(best['symbol'])
        
            success, order_id_or_msg = place_buy_order(
                best["symbol"], best["security_id"], best["price"], best["qty"]
            )
            systime.sleep(1.2)
        
            if success:
                order_status = get_trade_status(order_id_or_msg)
                print(f"🛰️ Order status for {best['symbol']} is {order_status}")
                if order_status not in ["TRADED", "OPEN", "UNKNOWN", "TRANSIT"]:
                    send_telegram_message(f"❌ Order rejected by broker: {order_status} — {best['symbol']}")
                    log_bot_action("autotrade.py", "REJECTED", "❌ Broker rejected", f"{best['symbol']} → {order_status}")
                    return
                trade_executed = True
                s = best
                print("✅ Final trade completed. Terminating auto-trade script.")
                sys.exit(0)  # 🔥 Hard exit after one successful order
        
            else:
                send_telegram_message(f"❌ Order failed for {best['symbol']}: {order_id_or_msg}")
                log_bot_action("autotrade.py", "BUY", "❌ FAILED", f"{best['symbol']} → {order_id_or_msg}")
        
                for alt in fallbacks:
                    if not alt or alt["symbol"] in bought_stocks:
                        continue
        
                    print(f"⚠️ Trying fallback candidate: {alt['symbol']}")
                    bought_stocks.add(alt['symbol'])  # Prevent re-buy
        
                    success, order_id_or_msg = place_buy_order(
                        alt["symbol"], alt["security_id"], alt["price"], alt["qty"]
                    )
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
                        sys.exit(0)  # 🔥 Hard exit on fallback success
        
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
        log_to_postgres(datetime.now(), "autotrade.py", "CRASH", str(e))

    if not has_open_position():
        log_bot_action("autotrade.py", "end", "NO TRADE", "No stock bought today.")
        log_to_postgres(datetime.now(), "autotrade.py", "end", "No stock bought today.")
    
    # 📝 Save all captured print outputs to a .txt log file
    try:
        log_path = "D:/Downloads/Dhanbot/dhan_autotrader/Logs/autotrade.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_buffer.getvalue())
        print(f"📄 Autotrade log saved to {log_path}")
    except Exception as e:
        print(f"⚠️ Failed to write autotrade log: {e}")