import os
import sys
import json
import pandas as pd
import time
from datetime import datetime, timedelta, time as dtime
import pytz
from dhanhq import DhanContext, dhanhq
from decimal import Decimal, ROUND_DOWN
import requests
import numpy as np
from db_logger import insert_portfolio_log_to_db

import io
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

# ========== Config ==========
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
MASTER_CSV = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
CAPITAL_FILE = "D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv"
PORTFOLIO_LOG = "D:/Downloads/Dhanbot/dhan_autotrader/portfolio_log.csv"

# ========== Load Config ==========
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
context = DhanContext(config["client_id"], config["access_token"])
dhan = dhanhq(context)

# ========== Telegram from config ==========
TG_TOKEN = config["telegram_token"]
TG_CHAT_ID = config["telegram_chat_id"]

# ========== Pattern Confidence Weights ==========
PATTERN_WEIGHTS = {
    "Bullish Abandoned Baby": {"weight": 1.8, "vol_scale": 3.0},
    "Bullish Kicker": {"weight": 1.6, "vol_scale": 2.5},
    "Morning Star": {"weight": 1.5, "vol_scale": 3.0},
    "Bullish Pennant": {"weight": 1.4, "vol_scale": 2.0},
    "Bullish Three Methods": {"weight": 1.4, "vol_scale": 1.8},
    "Three White Soldiers": {"weight": 1.3, "vol_scale": 1.5},
    "Bullish Engulfing": {"weight": 1.3, "vol_scale": 1.5},
    "Piercing Line": {"weight": 1.2, "vol_scale": 1.3},
    "Bullish Separating Lines": {"weight": 1.2, "vol_scale": 1.2},
    "Inverse Hammer": {"weight": 1.1, "vol_scale": 1.2},
    "Hammer": {"weight": 1.1, "vol_scale": 1.2},
    "Bullish Harami": {"weight": 1.1, "vol_scale": 1.1}
}

# ========== ATR/ADR Calculation ==========
AVERAGE_TRUE_RANGE_PERIOD = 14
ADR_PERIOD = 10  # Days for Average Daily Range

def calculate_atr(candles):
    """Calculate Average True Range (ATR) for volatility measurement"""
    if len(candles) < AVERAGE_TRUE_RANGE_PERIOD + 1:
        return 0.0
        
    df = pd.DataFrame(candles)
    high, low, close = df['high'], df['low'], df['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(AVERAGE_TRUE_RANGE_PERIOD).mean().iloc[-1]
    return atr

def calculate_adr(security_id):
    """Calculate Average Daily Range (ADR) for realistic TP/SL capping"""
    try:
        from_dt = (datetime.now() - timedelta(days=ADR_PERIOD + 5)).strftime("%Y-%m-%dT%H:%M:%S")
        to_dt = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        daily_data = dhan.historical_minute_data(security_id, "NSE_EQ", "EQUITY", from_dt, to_dt, type='day')
        
        if not daily_data or len(daily_data) < ADR_PERIOD:
            return 0.0
            
        daily_ranges = [abs(d['high'] - d['low']) for d in daily_data[-ADR_PERIOD:]]
        return sum(daily_ranges) / len(daily_ranges)
    except:
        return 0.0

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": msg}
        requests.post(url, data=payload)
    except:
        pass

def get_capital():
    with open(CAPITAL_FILE, "r") as f:
        return float(f.read().strip())

def has_hold():
    if not os.path.exists(PORTFOLIO_LOG):
        return False
    df = pd.read_csv(PORTFOLIO_LOG)
    today = datetime.now().strftime("%m/%d/%Y")
    return any((df['status'].str.upper() == "HOLD") & df['timestamp'].str.contains(today))

# ========== Fetch Candles Function ==========
def fetch_candles(security_id, interval="5", count=20, cache={}):
    """Fetch candles with caching to prevent redundant calls"""
    cache_key = f"{security_id}_{interval}_{count}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        now = datetime.now()
        from_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
        to_dt = now.replace(second=0, microsecond=0)
    
        # Dhan API wants just the dates for intraday charts
        response = dhan.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment="NSE_EQ",
            instrument_type="EQUITY",
            from_date=from_dt.strftime("%Y-%m-%d"),
            to_date=to_dt.strftime("%Y-%m-%d")
        )
    
        print(f"🧪 Candle fetch raw for {security_id} ({from_dt} to {to_dt}): {response}")

        # Check if response is valid and extract data
        if not response or not isinstance(response, dict) or 'data' not in response:
            print(f"⚠️ Invalid response structure for {security_id}")
            return []
            
        raw_data = response['data']
        required_keys = ["open", "high", "low", "close", "volume", "timestamp"]
        if (
            not raw_data or
            any(k not in raw_data for k in required_keys) or
            any(not raw_data[k] for k in required_keys) or
            len(set(len(raw_data[k]) for k in required_keys)) > 1
        ):
            print(f"⚠️ Empty or malformed candle data for {security_id}")
            return []
        
        
        candles = []
        try:
            for i in range(len(raw_data["open"])):
                candles.append({
                    "open": raw_data["open"][i],
                    "high": raw_data["high"][i],
                    "low": raw_data["low"][i],
                    "close": raw_data["close"][i],
                    "volume": raw_data["volume"][i],
                    "timestamp": datetime.fromtimestamp(raw_data["timestamp"][i])
                })
        except Exception as e:
            print(f"❌ Error while parsing candle for {security_id}: {e}")
            return []
        
        if not candles:
            print(f"⚠️ No valid parsed candles for {security_id}")
            return []      
            
        cache[cache_key] = candles
        return candles

    except Exception as e:
        print(f"❌ Error fetching candles for {security_id}: {e}")
        return []


def detect_bullish_pattern(candles):
    """Enhanced pattern detection with body/wick scoring"""
    if not candles:
        return False, None, 0.0
    
    df = pd.DataFrame(candles)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    pattern_name = None
    pattern_score = 0.0  # Confidence score 0-1
    
    # Volume confirmation helper
    def volume_confirmed(index=-1, multiplier=1.2, lookback=5):
        if len(df) < lookback + 1:
            return True, 1.0  # Not enough data
        vol_avg = v.iloc[-lookback-1:-1].mean()
        vol_ratio = v.iloc[index] / vol_avg
        return vol_ratio >= multiplier, vol_ratio
    
    # Body/Range scoring
    def candle_score(index):
        body = abs(c.iloc[index] - o.iloc[index])
        candle_range = h.iloc[index] - l.iloc[index]
        body_ratio = body / candle_range if candle_range > 0 else 0
        upper_wick = h.iloc[index] - max(c.iloc[index], o.iloc[index])
        lower_wick = min(c.iloc[index], o.iloc[index]) - l.iloc[index]
        return min(1.0, body_ratio * 0.7 + (1 - (upper_wick + lower_wick)/candle_range) * 0.3)

    # 0. Bullish Pennant (continuation pattern)
    if len(c) >= 15:
        flagpole_high = h.iloc[-15:-10].max()
        flagpole_low = l.iloc[-15:-10].min()
        flagpole_height = flagpole_high - flagpole_low
        
        cons_highs = h.iloc[-10:-1]
        cons_lows = l.iloc[-10:-1]
        
        if flagpole_height > 0:
            high_range = cons_highs.max() - cons_highs.min()
            low_range = cons_lows.max() - cons_lows.min()
            
            if (high_range < flagpole_height * 0.5 and 
                low_range < flagpole_height * 0.5 and
                c.iloc[-1] > o.iloc[-1] and
                c.iloc[-1] > cons_highs.mean()):
                vol_ok, vol_ratio = volume_confirmed(index=-1, multiplier=1.5)
                if vol_ok:
                    pattern_name = "Bullish Pennant"
                    pattern_score = candle_score(-1) * 0.9

    # 1. Bullish Abandoned Baby (strong reversal pattern)
    if not pattern_name and len(c) >= 3:
        body2 = abs(c.iloc[-2] - o.iloc[-2])
        if (c.iloc[-3] < o.iloc[-3] and
            body2 < 0.1 * (h.iloc[-2] - l.iloc[-2]) and
            h.iloc[-2] < l.iloc[-3] and
            l.iloc[-1] > h.iloc[-2] and
            c.iloc[-1] > o.iloc[-1]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Bullish Abandoned Baby"
                gap_size = (l.iloc[-1] - h.iloc[-2]) / h.iloc[-2]  # Gap-up percentage
                pattern_score = min(1.0, 0.6 + gap_size*2)

    # 2. Bullish Kicker (strong reversal)
    if not pattern_name and len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and
            o.iloc[-1] > c.iloc[-2] and
            c.iloc[-1] > o.iloc[-1]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Bullish Kicker"
                pattern_score = candle_score(-1) * 0.85

    # 3. Bullish Three Methods (continuation pattern)
    if not pattern_name and len(c) >= 5:
        if (c.iloc[-5] > o.iloc[-5] and
            all(c.iloc[i] < o.iloc[i] for i in [-4, -3, -2]) and
            min(o.iloc[-4], o.iloc[-3], o.iloc[-2]) > o.iloc[-5] and
            max(c.iloc[-4], c.iloc[-3], c.iloc[-2]) < c.iloc[-5] and
            c.iloc[-1] > o.iloc[-1] and
            c.iloc[-1] > c.iloc[-5]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Bullish Three Methods"
                pattern_score = min(1.0, 0.8 * (vol_ratio / 2.0))

    # 4. Bullish Separating Lines (continuation)
    if not pattern_name and len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and
            abs(o.iloc[-1] - c.iloc[-2]) < 0.01 * c.iloc[-2] and
            c.iloc[-1] > o.iloc[-1] and
            c.iloc[-1] > o.iloc[-2]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Bullish Separating Lines"
                pattern_score = candle_score(-1) * 0.8

    # 5. Enhanced Morning Star (with proper gap checks)
    if not pattern_name and len(c) >= 3:
        body1 = o.iloc[-3] - c.iloc[-3]  # First candle body (bearish)
        body2 = abs(c.iloc[-2] - o.iloc[-2])
        if (c.iloc[-3] < o.iloc[-3] and
            o.iloc[-2] < c.iloc[-3] and
            body2 < 0.3 * body1 and
            c.iloc[-1] > o.iloc[-1] and
            o.iloc[-1] > c.iloc[-2] and
            c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Morning Star"
                pattern_score = min(1.0, 0.7 + (c.iloc[-1] - o.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]))

    # -- Existing patterns with volume confirmation --
    # Bullish Engulfing
    if not pattern_name and len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and 
            c.iloc[-1] > o.iloc[-1] and
            c.iloc[-1] > o.iloc[-2] and 
            o.iloc[-1] < c.iloc[-2]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Bullish Engulfing"
                pattern_score = candle_score(-1) * 0.85

    # Hammer (single candle pattern)
    if not pattern_name and len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
        uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        if lw > 2 * body and uw < body:
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Hammer"
                pattern_score = min(1.0, 0.75 * (lw / body))

    # Piercing Line
    if not pattern_name and len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and 
            c.iloc[-1] > o.iloc[-1] and
            o.iloc[-1] < c.iloc[-2] and 
            c.iloc[-1] > (o.iloc[-2] + c.iloc[-2]) / 2):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Piercing Line"
                pattern_score = min(1.0, 0.8 * (c.iloc[-1] - o.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]))

    # Bullish Harami
    if not pattern_name and len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and 
            c.iloc[-1] > o.iloc[-1] and
            o.iloc[-1] > c.iloc[-2] and 
            c.iloc[-1] < o.iloc[-2]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Bullish Harami"
                pattern_score = candle_score(-1) * 0.75

    # Inverse Hammer
    if not pattern_name and len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
        if uw > 2 * body and lw < body:
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Inverse Hammer"
                pattern_score = min(1.0, 0.7 * (uw / body))

    # Three White Soldiers
    if not pattern_name and len(c) >= 3:
        if (c.iloc[-3] > o.iloc[-3] and 
            c.iloc[-2] > o.iloc[-2] and 
            c.iloc[-1] > o.iloc[-1] and
            c.iloc[-2] > c.iloc[-3] and 
            c.iloc[-1] > c.iloc[-2]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_name = "Three White Soldiers"
                pattern_score = min(1.0, 0.9 * (vol_ratio / 2.0))
    
    # Return detection status, pattern name, and confidence score
    if pattern_name:
        return True, pattern_name, pattern_score
    return False, None, 0.0

def compute_rsi_macd(closes):
    delta = closes.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean().bfill()
    avg_loss = loss.rolling(14).mean().bfill().replace(0, 0.01)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return rsi.iloc[-1], histogram.iloc[-1]

def is_index_bullish(index_id):
    """Check bullishness for any index (NIFTY/Sector)"""
    candles = fetch_candles(index_id, count=20)
    if not candles:
        return False
    closes = pd.Series([c["close"] for c in candles])
    rsi, macd = compute_rsi_macd(closes)
    detected, _, _ = detect_bullish_pattern(candles)
    return detected and rsi > 50 and macd > 0

def check_breakout(candles, period=3):
    """Confirm 15-min high breakout"""
    if len(candles) < period:
        return False
    current_high = candles[-1]['high']
    prev_high = max(c['high'] for c in candles[-period-1:-1])
    return current_high > prev_high

def check_gap_up(security_id):
    """Prevent entries after significant gap-ups"""
    try:
        prev_close = dhan.get_quote(security_id)['previousClose']
        today_open = dhan.get_quote(security_id)['open']
        return (today_open - prev_close) / prev_close > 0.01  # 1% gap-up threshold
    except:
        return False

def detect_reversal_pattern(candles, pattern_type):
    """Detect bearish reversal patterns for position monitoring"""
    if not candles:
        return False
        
    df = pd.DataFrame(candles)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    
    # Shooting Star detection
    if pattern_type == "Shooting Star" and len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
        if uw > 2 * body and lw < body:
            return True
            
    # Bearish Engulfing detection
    elif pattern_type == "Bearish Engulfing" and len(c) >= 2:
        if (c.iloc[-2] > o.iloc[-2] and 
            c.iloc[-1] < o.iloc[-1] and
            o.iloc[-1] > c.iloc[-2] and 
            c.iloc[-1] < o.iloc[-2]):
            return True
            
    # Evening Star detection
    elif pattern_type == "Evening Star" and len(c) >= 3:
        body1 = c.iloc[-3] - o.iloc[-3]  # Bullish body
        body2 = abs(c.iloc[-2] - o.iloc[-2])
        if (c.iloc[-3] > o.iloc[-3] and
            o.iloc[-2] > c.iloc[-3] and
            body2 < 0.3 * body1 and
            c.iloc[-1] < o.iloc[-1] and
            o.iloc[-1] < c.iloc[-2] and
            c.iloc[-1] < (o.iloc[-3] + c.iloc[-3]) / 2):
            return True
            
    return False

def log_trade(symbol, security_id, action, price, qty, status):
    now = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    log_row = [now, symbol, security_id, action, price, qty, status]

    # CSV log
    with open(PORTFOLIO_LOG, "a") as f:
        f.write(",".join(map(str, log_row)) + "\n")

    # DB log
    try:
        insert_portfolio_log_to_db(now, symbol, security_id, action, price, qty, status)
    except Exception as e:
        print("❌ Failed to log to DB:", e)

def place_exit_order(symbol, security_id, qty):
    """Place exit order for position monitoring"""
    try:
        quote = dhan.get_quote(security_id)
        if not quote:
            return
            
        price = float(quote['ltp'])
        tick_size = Decimal("0.05")
        limit_price = Decimal(str(price)) * Decimal("0.998")
        limit_price = (limit_price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size
        limit_price = float(limit_price.quantize(tick_size, rounding=ROUND_DOWN))
        
        order = {
            "transactionType": "SELL",
            "exchangeSegment": "NSE_EQ",
            "productType": "CNC",
            "orderType": "LIMIT",
            "validity": "DAY",
            "securityId": str(security_id),
            "tradingSymbol": symbol,
            "quantity": qty,
            "price": limit_price,
            "disclosedQuantity": 0,
            "afterMarketOrder": False,
            "amoTime": "OPEN",
            "triggerPrice": 0,
            "smartOrder": False
        }
        
        res = dhan.place_order(**order)
        print(f"✅ Exit order placed for {symbol}: {res}")
        send_telegram(f"⚠️ Exiting {symbol} due to reversal signal @ ₹{limit_price:.2f}")
        log_trade(symbol, security_id, "SELL", limit_price, qty, "EXITED")
        
    except Exception as e:
        print(f"❌ Exit order failed for {symbol}: {e}")
        send_telegram(f"❌ Exit order failed for {symbol}: {e}")

def monitor_hold_position():
    """Check existing position for reversals and volume spikes"""
    if not os.path.exists(PORTFOLIO_LOG):
        return
        
    df = pd.read_csv(PORTFOLIO_LOG)
    today = datetime.now().strftime("%m/%d/%Y")
    hold_positions = df[(df['status'] == "HOLD") & (df['timestamp'].str.contains(today))]
    
    for _, pos in hold_positions.iterrows():
        candles = fetch_candles(pos['security_id'], count=5)
        if not candles:
            continue
            
        # Check for bearish reversal patterns
        bearish_patterns = ["Shooting Star", "Bearish Engulfing", "Evening Star"]
        for pattern in bearish_patterns:
            if detect_reversal_pattern(candles, pattern):
                place_exit_order(pos['symbol'], pos['security_id'], pos['qty'])
                break
                
        # Volume spike alert
        current_vol = candles[-1]['volume']
        avg_vol = sum(c['volume'] for c in candles[:-1]) / (len(candles) - 1)
        if current_vol > 2.5 * avg_vol and avg_vol > 0:
            send_telegram(f"⚠️ Volume spike {pos['symbol']}: {current_vol/avg_vol:.1f}x average")

def place_order(symbol, security_id, qty, price, pattern_name, candles):
    tick_size = Decimal("0.05")
    limit_price = Decimal(str(price)) * Decimal("1.002")
    limit_price = (limit_price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size
    limit_price = float(limit_price.quantize(tick_size, rounding=ROUND_DOWN))
    
    # Calculate ADR for realistic TP/SL capping
    adr = calculate_adr(security_id)
    
    order = {
        "transactionType": "BUY",
        "exchangeSegment": "NSE_EQ",
        "productType": "CNC",
        "orderType": "LIMIT",
        "validity": "DAY",
        "securityId": str(security_id),
        "tradingSymbol": symbol,
        "quantity": qty,
        "price": limit_price,
        "disclosedQuantity": 0,
        "afterMarketOrder": False,
        "amoTime": "OPEN",
        "triggerPrice": 0,
        "smartOrder": False
    }
    try:
        res = dhan.place_order(**order)
        print("✅ Order Placed:", res)
        msg = f"✅ BUY {symbol} Qty: {qty} @ ₹{limit_price}"
        if pattern_name:
            msg += f" | Pattern: {pattern_name}"
        send_telegram(msg)
        log_trade(symbol, security_id, "BUY", limit_price, qty, "HOLD")

        # Enhanced Stop Loss and Target via Forever Order
        try:
            # Base parameters
            base_sl_pct = 0.005
            base_tp_pct = 0.01
            
            # Pattern-specific adjustments
            if pattern_name and pattern_name in PATTERN_WEIGHTS:
                conf = PATTERN_WEIGHTS[pattern_name]
                base_tp_pct = 0.01 * conf["weight"]
                base_sl_pct = 0.005 * (2 - conf["weight"]/2)  # Inverse to weight
                
            # Volatility adjustment using ATR
            atr = calculate_atr(candles)
            entry_price = Decimal(str(price))
            atr_multiplier = conf["vol_scale"] * (atr / entry_price) if pattern_name and pattern_name in PATTERN_WEIGHTS else 1.0
            
            # Apply volatility scaling
            tp_pct = max(base_tp_pct, atr_multiplier)
            sl_pct = min(base_sl_pct, atr_multiplier * 0.7)  # Tighter stops
            
            # Time decay adjustment for late entries
            market_close = dtime(15, 30)
            remaining_time = datetime.combine(datetime.today(), market_close) - datetime.now()
            remaining_hours = remaining_time.total_seconds() / 3600
            time_decay = max(0.5, remaining_hours / 6.5)  # 6.5 trading hours
            tp_pct *= time_decay
            
            # Ensure minimum 1:2 risk-reward ratio
            if tp_pct / sl_pct < 2:
                tp_pct = sl_pct * 2.2  # Add small buffer
                
            # Calculate final SL and TP
            stop_loss = float(entry_price * (1 - sl_pct))
            target = float(entry_price * (1 + tp_pct))
            
            # Apply ADR capping
            max_move = adr * 0.3  # Allow up to 30% of ADR
            target = min(target, price + max_move)
            stop_loss = max(stop_loss, price - max_move * 0.7)
            
            # Round to nearest tick
            stop_loss = float((Decimal(stop_loss) / tick_size).quantize(1) * tick_size)
            target = float((Decimal(target) / tick_size).quantize(1) * tick_size)
            
            # Time feasibility check - don't set unrealistic targets
            required_move = (target - price) / price
            if required_move > 0.015 * (remaining_hours / 1.5):  # Max 1.5% per hour
                target = price * (1 + 0.015 * (remaining_hours / 1.5))
                target = float((Decimal(target) / tick_size).quantize(1) * tick_size)
                send_telegram(f"⚠️ Adjusted {symbol} target to ₹{target:.2f} for time constraints")

            # Special handling for Morning Star pattern - confirm BEFORE placing SL/TP
            if pattern_name == "Morning Star":
                # Add confirmation check
                next_candle = fetch_candles(security_id, count=1)
                if next_candle and next_candle[0]['close'] > candles[-1]['close']:
                    tp_pct *= 1.2  # Increase target if confirmation exists
                    target = float(entry_price * (1 + tp_pct))
                    target = float((Decimal(target) / tick_size).quantize(1) * tick_size)
                    print(f"🌟 Morning Star confirmation - Increased target to ₹{target:.2f}")

            # Small delay to avoid overlap
            time.sleep(1.5)

            dhan.place_forever(
                security_id=str(security_id),
                exchange_segment=dhan.NSE,
                transaction_type=dhan.SELL,
                product_type=dhan.CNC,
                quantity=qty,
                price=target,
                trigger_Price=stop_loss
            )
            print(f"🎯 SL/TP set for {symbol}: Target ₹{target:.2f}, Stop ₹{stop_loss:.2f}")
            send_telegram(
                f"🎯 {symbol} | {pattern_name}\n"
                f"ENTRY: ₹{limit_price:.2f} | QTY: {qty}\n"
                f"SL: ₹{stop_loss:.2f} ({sl_pct*100:.1f}%)\n"
                f"TARGET: ₹{target:.2f} ({tp_pct*100:.1f}%)"
            )

        except Exception as e:
            print("⚠️ Failed to place SL/TP:", e)
            send_telegram(f"⚠️ SL/TP setup failed for {symbol}: {e}")

    except Exception as e:
        print("❌ Order Failed:", e)
        send_telegram(f"❌ Order Failed for {symbol}: {e}")

def main():
    print('📌 Starting enhanced autotrade')
    
    # Precompute market close time with buffer (15:30 - 20 minutes = 15:10)
    market_close = dtime(15, 30)
    min_holding_window = timedelta(minutes=20)
    end_time = datetime.combine(datetime.today(), market_close) - min_holding_window
    
    # Load data once at start
    master_df = pd.read_csv(MASTER_CSV)
    print('📊 Loaded master CSV for index checks')
    
    # Sector mapping setup
    sector_indices = {
        "BANK": "NIFTY BANK", 
        "IT": "NIFTY IT",
        "AUTO": "NIFTY AUTO",
        "PHARMA": "NIFTY PHARMA",
        "FMCG": "NIFTY FMCG",
        "METAL": "NIFTY METAL"
    }
    
    # Find NIFTY index ID once
    nifty50_row = master_df[master_df["SM_SYMBOL_NAME"].str.upper().str.contains("NIFTY")].head(1)
    if nifty50_row.empty:
        print("❌ NIFTY index not found in master")
        send_telegram("❌ CRITICAL: NIFTY index not found in master data")
        return
    nifty_id = nifty50_row.iloc[0]["SEM_SMST_SECURITY_ID"]
    
    # Continuous monitoring loop until market close
    while datetime.now() < end_time:
        candle_cache = {}  # Reset cache for each iteration
        print(f"\n⏰ New scanning cycle at {datetime.now().strftime('%H:%M:%S')}")
        
        # 1. Monitor existing positions
        monitor_hold_position()
        if has_hold():
            print("⏩ Active hold exists - waiting for exit signal")
            time.sleep(300)  # Wait 5 minutes before next check
            continue  # Skip new entries but keep monitoring
        
        # 2. Load capital and stock list (reload for potential daily updates)
        capital = get_capital()
        print(f'💰 Capital loaded: ₹{capital:,.2f}')
        
        df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv")
        df["security_id"] = df["security_id"].astype(int).astype(str)
        print(f'📄 Loaded dynamic_stock_list.csv with {len(df)} entries')
        
        # 3. Check market sentiment
        nifty_bullish = is_index_bullish(nifty_id)
        sector_status = {}
        
        # Check all sectors once per cycle
        for sector, index_name in sector_indices.items():
            sector_row = master_df[master_df["SM_SYMBOL_NAME"] == index_name].head(1)
            if not sector_row.empty:
                sector_id = sector_row.iloc[0]["SEM_SMST_SECURITY_ID"]
                sector_status[sector] = is_index_bullish(sector_id)
                status = "bullish" if sector_status[sector] else "bearish"
                print(f'  📈 {index_name} sector: {status}')
        
        if not nifty_bullish:
            print("📉 Overall market bearish - focusing on bullish sectors")
            send_telegram("⚠️ NIFTY bearish. Focusing on bullish sectors")
        
        # 4. Stock evaluation loop
        order_placed = False
        for index, row in df.iterrows():
            try:
                if datetime.now() >= end_time:
                    print("⏰ Time window expired - stopping evaluation")
                    break
                    
                symbol = row["symbol"]
                secid = row["security_id"]
                sector = row.get("sector", "UNKNOWN")
                print(f'➡️ Evaluating {symbol} ({sector} sector)')
                
                # Skip if sector is known and bearish in bear market
                if not nifty_bullish and sector in sector_status and not sector_status[sector]:
                    print(f'  📉 Sector {sector} bearish - skipping in weak market')
                    continue
                    
                # Fetch candles
                candles = fetch_candles(secid, count=30, cache=candle_cache)
                time.sleep(0.5)  # API rate limit protection
                if not candles:
                    print('⚠️ No candle data available, skipping...')
                    continue
                    
                # Gap-up filter
                if check_gap_up(secid):
                    print(f'⏫ Gap-up detected: {symbol}')
                    continue
                    
                # Pattern detection
                detected, pattern_name, pattern_score = detect_bullish_pattern(candles)
                if not detected:
                    print('📉 No bullish pattern detected, skipping...')
                    continue
                
                # Breakout confirmation
                if not check_breakout(candles):
                    print(f'❌ No breakout confirmation: {symbol}')
                    continue
                    
                # Technical indicators
                closes = pd.Series([c["close"] for c in candles])
                rsi, macd = compute_rsi_macd(closes)
                print(f'📊 RSI: {rsi:.2f}, MACD: {macd:.4f}, Pattern Score: {pattern_score:.2f}')
                if not (45 < rsi < 70 and macd > 0):
                    print('❌ RSI/MACD filter failed, skipping...')
                    continue

                # Time check
                remaining_time = datetime.combine(datetime.today(), market_close) - datetime.now()
                if remaining_time < min_holding_window:
                    print(f"⏱️ Skipping {symbol} — too late to enter. Only {remaining_time} left.")
                    continue

                # Position sizing
                price = closes.iloc[-1]
                base_qty = int(capital // price)
                confidence = PATTERN_WEIGHTS.get(pattern_name, {"weight": 1.0})["weight"]
                adj_qty = max(1, int(base_qty * confidence * pattern_score))
                
                print(f'💸 Final Price: ₹{price:.2f}, Base Qty: {base_qty}, Confidence: {confidence:.1f}x, Adj Qty: {adj_qty}')
                if adj_qty <= 0:
                    print('⛔ Quantity is zero or negative, skipping...')
                    continue

                # Place order
                place_order(symbol, secid, adj_qty, price, pattern_name, candles)
                order_placed = True
                break  # Exit after successful order
                
            except Exception as e:
                print(f'⚠️ {symbol} evaluation failed: {str(e)}')
                continue
                
        if not order_placed:
            print("❌ No valid trades found this cycle")
            
        # Wait 5 minutes before next scan
        print("🔄 Waiting for next scan cycle in 2 minutes...")
        time.sleep(240)

    print("🛑 Trading window closed - stopping monitoring")

# Save full execution log to log file
try:
    log_path = "D:/Downloads/Dhanbot/dhan_autotrader/Logs/New_Autotrade.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(log_buffer.getvalue())
except Exception as e:
    print(f"⚠️ Failed to write log file: {e}")

if __name__ == "__main__":
    main()