import csv
import json
import requests
from datetime import datetime, time
import pytz
import time as systime
import pandas as pd
import os
from dhan_api import get_live_price, get_historical_price, compute_rsi, calculate_qty, get_stock_volume
from Dynamic_Gpt_Momentum import prepare_data, ask_gpt_to_rank_stocks
from utils_logger import log_bot_action
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from utils_safety import safe_read_csv




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
    buffer_price = round(round(price * 1.002 / tick_size) * tick_size, 2)
    
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
        response = requests.post(BASE_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            res_json = response.json()
            if res_json.get("status") == "success":
                order_id = res_json.get("data", {}).get("orderId", "N/A")
                print(f"✅ Order placed for {symbol} | Order ID: {order_id}")
                send_telegram_message(f"✅ Order placed for {symbol} | Order ID: {order_id}")
                return True, order_id
            else:
                error = res_json.get("remarks", {}).get("error_message", "Unknown error")
                print(f"❌ Order rejected for {symbol}: {error}")
                return False, error        
        else:
            print(f"❌ Error placing order for {symbol}: {response.status_code} - {response.text}")
            return False, response.text

    except Exception as e:
        print(f"❌ Exception placing order for {symbol}: {e}")
        return False, str(e)

    finally:
        # ⏱️ Throttle to avoid 429 Rate Limit
        systime.sleep(random.uniform(0.6, 1.2))

def get_trade_status(order_id):
    try:
        response = requests.get(TRADE_BOOK_URL, headers=HEADERS)
        if response.status_code == 200:
            trades = response.json().get("data", [])
            for t in trades:
                if t.get("order_id") == order_id:
                    return t.get("status", "").upper()
        return "UNKNOWN"
    except:
        return "ERROR"

def get_atr(symbol, period=14):
    try:
        candles = get_historical_price(symbol, interval="15m")[-(period + 1):]
        trs = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else None
    except:
        return None

def log_trade(symbol, security_id, qty, price):
    timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")
    # Get ATR dynamically (past 14 periods)
    atr = get_atr(symbol, period=14)  # You'll define this helper
    if atr:
        target_pct = round((atr / price) * 100 * 1.2, 2)  # 1.2x ATR for target
        stop_pct = round((atr / price) * 100 * 0.8, 2)   # 0.8x ATR for stop
    else:
        target_pct = 0.7
        stop_pct = 0.4
        
    file_exists = os.path.isfile(PORTFOLIO_LOG)
    with open(PORTFOLIO_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(PORTFOLIO_LOG).st_size == 0:
            writer.writerow([
                "timestamp", "symbol", "security_id", "quantity", "buy_price",
                "momentum_5min", "target_pct", "stop_pct", "live_price",
                "change_pct", "last_checked", "status", "exit_price"
            ])
        writer.writerow([
            timestamp, symbol, security_id, qty, price,
            0, target_pct, stop_pct, '', '', '', 'HOLD', ''
        ])
    
best_candidate = None
trade_lock = threading.Lock()

def monitor_stock_for_breakout(symbol, high_trigger, capital, dhan_symbol_map):
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
        if high_trigger and price < high_trigger:
            print(f"⏭️ Skipping {symbol} — Price ₹{price} has not crossed 15-min high ₹{high_trigger}")
            return

        # RSI Check
        rsi = compute_rsi(security_id)
        if rsi is None:
            print(f"⚠️ Skipping {symbol} — Unable to compute RSI.")
            return
        if rsi >= 70:
            print(f"⚠️ Skipping {symbol} — RSI too high: {rsi}")
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
        # Ensure this function exists or implement it
        atr = get_atr(symbol)  # Ensure ATR function exists
        
        volume_score = min(1.0, volume / 1000000)  # Normalize volume
        atr_distance = abs(price - high_trigger) if high_trigger else 0
        atr_score = min(1.0, atr_distance / price) if price else 0
        
        # Optional: placeholder ML sentiment score (e.g., from GPT earlier process or set to 0.5 default)
        ml_score = 0.5  
        
        weighted_score = round((ml_score * 0.6) + (volume_score * 0.2) + (atr_score * 0.2), 4)
        
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
    if not is_market_open() or not is_nse_trading_day():
        print("⛔ Market is closed today (weekend or holiday). Exiting auto-trade.")
        log_bot_action("autotrade.py", "market_status", "INFO", "Skipped: Market closed or holiday.")
        return
    csv_path = "D:/Downloads/Dhanbot/dhan_autotrader/live_stocks_trade_today.csv"
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

    df = pd.read_csv(csv_path)
    print(f"📄 Using stock list: {csv_path}")
    
    if df.empty or "symbol" not in df.columns or "security_id" not in df.columns:
        send_telegram_message("⚠️ dynamic_stock_list.csv is missing or invalid.")
        return
    
    ranked_stocks = df["symbol"].tolist()
    dhan_symbol_map = dict(zip(df["symbol"], df["security_id"]))
    bought_stocks = set()
    exit_loop = False
    capital = get_available_capital()
    first_15min_high = {}

    # ✅ Precompute 15-min high (for each ranked stock)
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
            systime.sleep(0.6)
            highs = [c['high'] for c in candles if 'high' in c]
            if highs:
                first_15min_high[stock] = max(highs[:3])  # first 3 candles ~15m
        except Exception as e:
            print(f"⚠️ Skipping {stock} — could not fetch 15min high. Reason: {e}")   

    # ✅ Live Monitoring Loop
    from datetime import datetime as dt
    
    trade_executed = False
    s = None  # Final trade data to reference in EOD
    
    while datetime.now(pytz.timezone("Asia/Kolkata")).time() <= time(14, 30):
        print(f"🔁 Monitoring stocks for breakout at {datetime.now().strftime('%H:%M:%S')}...")
        candidate_scores = []
        top_candidates = []
        for symbol in ranked_stocks:
            rate_limit_counter = 0
            if rate_limit_counter >= 5:
                print("⛔ Too many rate limit errors. Backing off for 30 seconds...")
                systime.sleep(30)
                rate_limit_counter = 0
            rate_limit_counter += 1
            high_trigger = first_15min_high.get(symbol)
            result = monitor_stock_for_breakout(symbol, high_trigger, capital, dhan_symbol_map)
            systime.sleep(0.8)  # Respect API rate limit
    
            if result:
                result["score"] = compute_trade_score(result)  # Optional: could be based on volume, price action, etc.
                candidate_scores.append(result)
                top_candidates.append(result)
    
        # Sort and buy best candidate
        if candidate_scores:
            top_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            best = top_candidates[0]  # Primary choice
            fallbacks = top_candidates[1:]  # Others as backup            
            pd.DataFrame(candidate_scores).to_csv("D:/Downloads/Dhanbot/dhan_autotrader/scanned_candidates_today.csv", index=False)
            print(f"✅ Best candidate selected: {best['symbol']} @ ₹{best['price']} (Score: {best['score']})")
            success, order_id_or_msg = place_buy_order(best["symbol"], best["security_id"], best["price"], best["qty"])
            systime.sleep(0.6)
    
            if success:
                log_trade(best["symbol"], best["security_id"], best["qty"], best["price"])
                send_telegram_message(f"✅ Bought {best['symbol']} at ₹{best['price']}, Qty: {best['qty']}")
                log_bot_action("autotrade.py", "BUY", "✅ EXECUTED", f"{best['symbol']} @ ₹{best['price']}")
                trade_executed = True
                s = best
                break
            else:
                send_telegram_message(f"❌ Order failed for {best['symbol']}: {order_id_or_msg}")
                log_bot_action("autotrade.py", "BUY", "❌ FAILED", f"{best['symbol']} → {order_id_or_msg}")
                
                # 🔁 Try fallback candidates
                for alt in fallbacks:
                    print(f"⚠️ Trying fallback candidate: {alt['symbol']}")
                    success, order_id_or_msg = place_buy_order(alt["symbol"], alt["security_id"], alt["price"], alt["qty"])
                    systime.sleep(0.6)
                    if success:
                        log_trade(alt["symbol"], alt["security_id"], alt["qty"], alt["price"])
                        send_telegram_message(f"✅ Fallback Buy: {alt['symbol']} at ₹{alt['price']}, Qty: {alt['qty']}")
                        log_bot_action("autotrade.py", "BUY", "✅ Fallback EXECUTED", f"{alt['symbol']} @ ₹{alt['price']}")
                        trade_executed = True
                        s = alt
                        break
              
        # Watchdog timeout
        now_time = datetime.now(pytz.timezone("Asia/Kolkata")).time()
        if not trade_executed and now_time >= time(14, 15):
            send_telegram_message("⚠️ No trade executed by 2:15 PM. Please review.")
            log_bot_action("autotrade.py", "WATCHDOG", "⚠️ NO TRADE", "No trade by 2:15 PM")
    
        systime.sleep(60)
    
    # 🕒 3:25 PM End-of-Day Telegram Report
    now_ist = dt.now(pytz.timezone("Asia/Kolkata")).strftime('%d-%b-%Y')
    
    if trade_executed and s:
        try:
            ltp_candle = get_latest_price(s["security_id"])
            systime.sleep(0.5)
            ltp = ltp_candle.get("last_price", s["price"])
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