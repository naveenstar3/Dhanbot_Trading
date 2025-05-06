import csv
import json
import requests
from datetime import datetime, time
import pytz
import time as systime
import os
from dhan_api import get_live_price, get_historical_price
from config import *
from Dynamic_Gpt_Momentum import prepare_data, ask_gpt_to_pick_stock  # 🔥 Import inside function if needed
from utils_logger import log_bot_action

# ✅ Constants
PORTFOLIO_LOG = "portfolio_log.csv"
LIVE_LOG = "live_prices_log.csv"
CURRENT_CAPITAL_FILE = "current_capital.csv"
GROWTH_LOG = "growth_log.csv"
BASE_URL = "https://api.dhan.co/orders"
TRADE_BOOK_URL = "https://api.dhan.co/trade-book"
NEWS_API_KEY = "c545f9478aab45bd9886110793d08bdb"
TELEGRAM_TOKEN = "7557430361:AAFZKf4KBL3fScf6C67quomwCrpVbZxQmdQ"
TELEGRAM_CHAT_ID = "5086097664"

# ✅ Load Dhan credentials
with open("dhan_config.json") as f:
    config = json.load(f)

HEADERS = {
    "access-token": config["access_token"],
    "client-id": config["client_id"],
    "Content-Type": "application/json"
}

# ✅ Bot Execution Logger
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


# ✅ Calculate Dynamic Minimum Net Profit
def get_dynamic_minimum_net_profit(capital):
    """
    Returns scaled minimum net profit:
    - Minimum ₹5
    - Scales as 0.1% of current capital
    """
    return max(5, round(capital * 0.001, 2))  # 0.1% of capital or ₹5, whichever is higher

# ✅ Load dynamic stock list
def load_dynamic_stocks():
    try:
        with open('D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.txt', 'r') as f:
            return [line.strip().upper() for line in f if line.strip()]
    except Exception as e:
        print(f"⚠️ Error loading dynamic stock list: {e}")
        return []

# ✅ Load Dhan Master CSV into memory
def load_dhan_master():
    dhan_map = {}
    try:
        with open("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get("SEM_TRADING_SYMBOL", "").strip().upper()
                if row.get("SEM_INSTRUMENT_NAME", "") == "EQUITY":
                    dhan_map[symbol] = row.get("SEM_SMST_SECURITY_ID")
        print(f"✅ Loaded {len(dhan_map)} securities from local dhan_master.csv")
    except Exception as e:
        print(f"⚠️ Error loading Dhan master CSV: {e}")
    return dhan_map

# ✅ Fetch security_id for given symbol instantly
def get_security_id(symbol):
    return dhan_symbol_map.get(symbol.upper())

# ✅ Part 2: Notifications, Utility Logic

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"⚠️ Telegram send error: {e}")

def emergency_exit_active():
    return os.path.exists("emergency_exit.txt")

def get_available_capital():
    try:
        with open(CURRENT_CAPITAL_FILE, "r") as f:
            base_capital = float(f.read().strip())
    except:
        base_capital = float(input("Enter your starting capital: "))
        with open(CURRENT_CAPITAL_FILE, "w") as f:
            f.write(str(base_capital))

    try:
        with open(GROWTH_LOG, newline="") as f:
            rows = list(csv.DictReader(f))
            if rows:
                last_growth = float(rows[-1].get("profits_realized", 0))
                if last_growth >= 5:
                    base_capital += last_growth
    except:
        pass

    return base_capital

def has_open_position():
    today = datetime.now().date()
    try:
        with open(PORTFOLIO_LOG, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status", "").upper() != "SOLD":
                    ts_str = row.get("timestamp", "")
                    try:
                        entry_date = datetime.strptime(ts_str, "%m/%d/%Y %H:%M").date()
                        if entry_date == today:
                            return True
                            log_bot_action("autotrade.py", "HOLD check", "INFO", "Open position already exists. Skipping buy.")
                    except:
                        continue
    except FileNotFoundError:
        return False
    return False

def is_market_open():
    now = datetime.now(pytz.timezone("Asia/Kolkata")).time()
    return time(9, 15) <= now <= time(15, 30)

def log_live_prices(price_data):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LIVE_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        for entry in price_data:
            writer.writerow([now, entry["symbol"], entry["price"]])

def place_buy_order_with_retry(payload, retries=1):
    for attempt in range(retries + 1):
        if TEST_MODE:
            return 200, {"order_id": "TEST_ORDER_ID_BUY"}
        response = requests.post(BASE_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return 200, response.json().get("data", {})
        else:
            if attempt < retries:
                systime.sleep(2)
                continue
            return response.status_code, {"error": response.text}

def get_trade_book():
    try:
        response = requests.get(TRADE_BOOK_URL, headers=HEADERS)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            return []
    except:
        return []

# ✅ Part 3: News Sentiment, Delivery, Volume Logic

def check_negative_news(stock_name):
    url = f"https://newsapi.org/v2/everything?q={stock_name}&apiKey={NEWS_API_KEY}&sortBy=publishedAt&pageSize=5"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()["articles"]
            negative_keywords = ["loss", "fraud", "scam", "penalty", "raid", "lawsuit", "problem", "downgrade", "default", "negative", "fire", "fine", "debt issue", "resignation"]
            for article in articles:
                title = article["title"].lower()
                description = article.get("description", "").lower()
                if any(word in title + description for word in negative_keywords):
                    return True
        return False
    except:
        return False

def get_delivery_percentage(symbol):
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers)
        data = response.json()
        delivery_qty = int(data['securityWiseDP']['deliveredQuantity'])
        total_qty = int(data['securityWiseDP']['tradedQuantity'])
        if total_qty > 0:
            return round((delivery_qty / total_qty) * 100, 2)
    except:
        return 0

def has_volume_surge(symbol):
    try:
        live_volume = get_live_price(symbol) * 1000
        avg_volume_5d = get_live_price(symbol) * 800
        if live_volume >= 1.2 * avg_volume_5d:
            return True
    except:
        return False
    return False

def psu_weightage_score(stock):
    try:
        market_cap_score = stock.get('market_cap', 0) / 1e10
        volume_score = stock.get('avg_volume_5d', 0) / 1000000
        return round((market_cap_score * 0.6) + (volume_score * 0.4), 2)
    except:
        return 0

def ml_momentum_predictor(momentum_5min, momentum_15min):
    strength_score = (momentum_5min * 0.6) + (momentum_15min * 0.4)
    return strength_score

# ✅ Part 4: AutoTrade Main Function

def run_autotrade():
    log_bot_action("autotrade.py", "startup", "STARTED", "AutoTrade script started.")
    print("🔍 Checking if market is open...")
    if not is_market_open():
        print("⏳ Market not open yet.")
        log_bot_action("autotrade.py", "market_status", "INFO", "Market closed today, skipping trading.")
        return

    if has_open_position():
        print("📌 Already have an open position. Skipping new trades.")
        return

    if emergency_exit_active():
        send_telegram_message("⛔ Emergency Exit Active. Skipping today's trading.")
        print("⛔ Emergency Exit Active. Skipping today's trading.")
        return

    capital = get_available_capital()
    MINIMUM_NET_PROFIT_REQUIRED = get_dynamic_minimum_net_profit(capital)

    # ✅ Load dynamic stocks
    candidates = load_dynamic_stocks()
    if not candidates:
        send_telegram_message("⚠️ No stocks in dynamic_stock_list.txt. Cannot proceed.")
        return

    # ✅ Fetch live 5-min and 15-min momentum
    df = prepare_data()
    if df.empty:
        send_telegram_message("⚠️ No live momentum data fetched. Skipping today's trade.")
        return

   # ✅ Ask GPT to pick safest stock
    gpt_pick = ask_gpt_to_pick_stock(df)
    
    # ✅ Check if GPT explicitly said SKIP
    if gpt_pick == "SKIP":
        send_telegram_message("⚠️ GPT advised to SKIP today. No safe stock to buy.")
        log_bot_action("autotrade.py", "gpt_decision", "SKIP", "GPT advised to skip trading.")
        return
    
    # ✅ NEW: Capital-based Affordability Check
    available_stocks = df["symbol"].tolist()
    
    def find_affordable_stock(available_stocks, capital):
        for stock in available_stocks:
            stock_price = get_live_price(stock)
            if not stock_price or stock_price <= 0:
                continue
            qty = int(capital // stock_price)
            if qty > 0:
                return stock  # First affordable stock
        return None
    
    # First check GPT pick affordability
    picked_price = get_live_price(gpt_pick)
    if picked_price and (capital // picked_price) >= 1:
        final_pick = gpt_pick
    else:
        print(f"⚠️ GPT pick {gpt_pick} too expensive. Searching alternative...")
        final_pick = find_affordable_stock(available_stocks, capital)
    
    if not final_pick:
        send_telegram_message("⚠️ No affordable stocks found even after fallback. Skipping today.")
        log_bot_action("autotrade.py", "stock_pick", "❌ SKIPPED", "No affordable stock found.")
        print("⚠️ No affordable stocks found. Skipping today's trade.")
        return
    
    print(f"✅ Final Stock Selected: {final_pick}")
    send_telegram_message(f"✅ Final Selected {final_pick} for today's buy.")
    
    # ✅ Find securityId for selected stock
    security_id = get_security_id(final_pick)
    if not security_id:
        send_telegram_message(f"⚠️ Security ID not found for {final_pick}. Cannot place order.")
        return
    
    # ✅ Fetch live price
    current_price = get_live_price(final_pick)
    if not current_price or current_price <= 0:
        send_telegram_message(f"⚠️ Live price unavailable for {final_pick}. Skipping.")
        return
    
    qty = int(capital // current_price)
    if qty <= 0:
        send_telegram_message(f"⚠️ Insufficient capital to buy {final_pick}. Needed more funds.")
        return
    
    approx_cost = current_price * qty
    buffer_required = approx_cost * 1.05
    if buffer_required > capital:
        send_telegram_message(f"⚠️ Skipping {final_pick}: Need ₹{round(buffer_required)} but have ₹{round(capital)}.")
        return
    
    # ✅ Place BUY order
    payload = {
        "transactionType": "BUY",
        "exchangeSegment": "NSE_EQ",
        "productType": "CNC",
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": security_id,
        "tradingSymbol": final_pick,
        "quantity": qty,
        "price": 0,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
        "amoTime": "OPEN",
        "triggerPrice": 0,
        "smartOrder": False
    }
    
    code, buy_response = place_buy_order_with_retry(payload, retries=1)
    
    matching_trades = []
    if code == 200:
        send_telegram_message(f"✅ Bought {final_pick} at approx ₹{current_price}, Qty: {qty}")
        log_bot_action("autotrade.py", "BUY attempt", "✅ Success", f"Bought {final_pick} @ ₹{round(current_price, 2)}")
        order_id = buy_response.get("orderId", "")
        systime.sleep(5)
        trade_book = get_trade_book()
        matching_trades = [t for t in trade_book if t.get("order_id") == order_id]
    if not matching_trades:
        print(f"⚠️ No matching trade found for order_id={order_id}. Trade Book: {trade_book}")
    else:
        trade_status = matching_trades[0].get("status", "").upper()
        print(f"🧾 Trade found. Status = {trade_status}")

    if matching_trades:
        trade_status = matching_trades[0].get("status", "").upper()
        if trade_status in ["TRADED", "PENDING", "OPEN"]:
            timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")
            target_pct = 0.7  # ✅ Fixed target 0.7%
            stop_pct = 0.4    # ✅ Fixed stoploss 0.4%

            with open(PORTFOLIO_LOG, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, final_pick, security_id, qty,
                    current_price, round(0, 2),  # dummy momentum
                    1, target_pct, stop_pct, '', 'HOLD', ''
                ])
            send_telegram_message(f"🗒️ Trade logged with Target {target_pct}% / Stop {stop_pct}%")

    else:
        send_telegram_message(f"❌ Buy order failed for {final_pick}: {buy_response}")

# ✅ Final Runner
if __name__ == "__main__":
    dhan_symbol_map = load_dhan_master()
    run_autotrade()
    
    if not has_open_position():  # means no trade was executed
        log_bot_action("autotrade.py", "end", "NO TRADE", "No stock bought today.")