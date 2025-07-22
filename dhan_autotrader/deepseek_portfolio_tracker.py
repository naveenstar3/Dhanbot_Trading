# ========== PART 2: AUTOTRADE MONITOR SCRIPT ==========
# This script handles position monitoring and trend reversal detection

import os
import sys
import json
import pandas as pd
import time
from datetime import datetime, timedelta, time as dtime
import pytz
from dhanhq import DhanContext, dhanhq
from decimal import Decimal, ROUND_HALF_UP
import requests
import numpy as np
from db_logger import update_portfolio_log_to_db, log_to_postgres
import math
import io
import traceback

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

# ========== Create DhanHQ global context and client ==========
context = DhanContext(config["client_id"], config["access_token"])
dhan = dhanhq(context)

# ========== Telegram from config ==========
TG_TOKEN = config["telegram_token"]
TG_CHAT_ID = config["telegram_chat_id"]

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": msg}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")

def get_capital():
    if not os.path.exists(CAPITAL_FILE):
        return 0.0
    with open(CAPITAL_FILE, "r") as f:
        try:
            return float(f.read().strip())
        except:
            return 0.0

def has_hold():
    if not os.path.exists(PORTFOLIO_LOG):
        return False
    try:
        df = pd.read_csv(PORTFOLIO_LOG)
        today = datetime.now().strftime("%m/%d/%Y")
        return any((df['status'].str.upper() == "HOLD") & df['timestamp'].str.contains(today))
    except:
        return False

def fetch_candles(security_id, count=20, cache={}, exchange_segment="NSE_EQ", instrument_type="EQUITY"):
    """Fetch candles with caching and retry mechanism to prevent redundant calls"""
    date_str = datetime.now().strftime("%Y%m%d")
    cache_key = f"{date_str}_{security_id}_{count}_{exchange_segment}"
    if cache_key in cache:
        return cache[cache_key]

    for attempt in range(3):  # Retry up to 3 times
        try:    
            now = datetime.now()
            from_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
            to_dt = now.replace(second=0, microsecond=0)
            
            response = dhan.intraday_minute_data(
                security_id=str(security_id),
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=from_dt.strftime("%Y-%m-%d"),
                to_date=to_dt.strftime("%Y-%m-%d")
            )

            # Check if response is valid and extract data
            if not response or not isinstance(response, dict) or 'data' not in response:
                print(f"⚠️ Invalid response structure for {security_id}")
                return []
                
            raw_data = response['data']
            required_keys = ["open", "high", "low", "close", "volume", "timestamp"]
            if (not raw_data or 
                any(k not in raw_data for k in required_keys) or 
                any(not raw_data[k] for k in required_keys) or 
                len(set(len(raw_data[k]) for k in required_keys if k in raw_data)) > 1):
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

            # ✅ Candle Timestamp Freshness Check
            last_candle_time = candles[-1]["timestamp"]
            if (datetime.now() - last_candle_time) > timedelta(minutes=5):
                print(f"⚠️ Stale data for {security_id}, ignoring cache")
                del cache[cache_key]  # Force refresh
                return fetch_candles(security_id, count, cache, exchange_segment, instrument_type)

            cache[cache_key] = candles
            return candles     
                
            cache[cache_key] = candles
            return candles

        except Exception as e:
            if "Rate_Limit" in str(e) and attempt < 2:
                wait_time = (attempt + 1) * 10
                print(f"⚠️ Rate limit hit for {security_id}, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            print(f"❌ Error fetching candles for {security_id}: {e}")
            return []
    return []

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
            
    # Bearish Harami detection
    elif pattern_type == "Bearish Harami" and len(c) >= 2:
        if (c.iloc[-2] > o.iloc[-2] and 
            c.iloc[-1] < o.iloc[-1] and
            o.iloc[-1] < c.iloc[-2] and 
            c.iloc[-1] > o.iloc[-2]):
            return True
            
    # Three Black Crows detection
    elif pattern_type == "Three Black Crows" and len(c) >= 3:
        if (c.iloc[-3] < o.iloc[-3] and 
            c.iloc[-2] < o.iloc[-2] and 
            c.iloc[-1极 < o.iloc[-1] and
            c.iloc[-3] > c.iloc[-2] and 
            c.iloc[-2] > c.iloc[-1] and
            o.iloc[-1] < o.iloc[-2] < o.iloc[-3]):
            return True
            
    # Bearish Kicker detection
    elif pattern_type == "Bearish Kicker" and len(c) >= 2:
        if (c.iloc[-2] > o.iloc[-2] and
            o.iloc[-1] < c.iloc[-2] and
            c.iloc[-1] < o.iloc[-1]):
            return True
            
    # Gravestone Doji detection
    elif pattern_type == "Gravestone Doji" and len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
        if uw > 3 * body and lw < body * 0.1:
            return True
            
    # Gap-Up Reversal detection
    elif pattern_type == "Gap-Up Reversal" and len(c) >= 2:
        gap_up = o.iloc[-1] > c.iloc[-2]
        strong_reversal = c.iloc[-1] < o.iloc[-1] and (o.iloc[-1] - c.iloc[-1]) > (h.iloc[-1] - l.iloc[-1]) * 0.7
        if gap_up and strong_reversal:
            return True
            
    # Hanging Man detection
    elif pattern_type == "Hanging Man" and len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
        uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        if lw > 2 * body and uw < body * 0.5 and c.iloc[-1] < o.iloc[-1]:
            return True
            
    # Breakdown Marubozu detection
    elif pattern_type == "Breakdown Marubozu" and len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        candle_range = h.iloc[-1] - l.iloc[-1]
        body_ratio = body / candle_range if candle_range > 0 else 0
        
        # Look for no wicks and closing at low
        if (body_ratio > 0.9 and 
            min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1] < candle_range * 0.05 and
            h.iloc[-1] - max(c.iloc[-1], o.iloc[-1]) < candle_range * 0.05 and
            c.iloc[-1] < o.iloc[-1]):
            return True
            
    return False

def detect_bearish_chart_pattern(candles):
    """Detect bearish chart patterns for position monitoring"""
    if not candles or len(candles) < 10:
        return False
        
    df = pd.DataFrame(candles)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    
    # Head and Shoulders pattern
    if len(c) >= 25:
        # Find left shoulder
        left_shoulder_idx = h.iloc[-25:-15].idxmax()
        left_shoulder = h.iloc[left_shoulder_idx]
        
        # Find head (highest point)
        head_idx = h.iloc[-20:-10].idxmax()
        head_high = h.iloc[head_idx]
        
        # Find right shoulder
        right_shoulder_idx = h.iloc[-10:-5].idxmax()
        right_shoulder = h.iloc[right_shoulder_idx]
        
        # Neckline (support)
        neckline = (l.iloc[left_shoulder_idx] + l.iloc[right_shoulder_idx]) / 2
        
        if (left_shoulder * 0.98 < right_shoulder < left_shoulder * 1.02 and
            head_high > left_shoulder * 1.03 and
            c.iloc[-1] < neckline):
            return True
    
    # Double/Triple Top
    if len(c) >= 20:
        # Find significant highs
        highs = h.rolling(5).max().dropna()
        max_idxs = highs.nlargest(3).index.tolist()
        max_idxs.sort()
        
        if len(max_idxs) >= 2:
            # Check if highs are approximately equal
            high1 = highs.iloc[max_idxs[0]]
            high2 = highs.iloc[max_idxs[1]]
            high_diff = abs(high1 - high2) / min(high1, high2)
            
            # Check breakdown below support
            support = l.iloc[max_idxs[0]:max_idxs[1]].min()
            if (high_diff < 0.02 and 
                c.iloc[-1] < support):
                return True
    
    # Descending Triangle
    if len(c) >= 15:
        # Horizontal support
        support = l.rolling(5).min().iloc[-15:].mean()
        support_range = l.iloc[-15:].max() - l.iloc[-15:].min()
        
        # Lower highs
        high_min = h.iloc[-15:].min()
        high_max = h.iloc[-15:].max()
        slope, _ = np.polyfit(range(15), h.iloc[-15:], 1)
        
        if (support_range / support < 0.02 and
            slope < 0 and
            (high_max - high_min) / high_min > 0.03 and
            c.iloc[-1] < support):
            return True
    
    # Bearish Rectangle
    if len(c) >= 10:
        # Horizontal support and resistance
        support = l.rolling(5).min().iloc[-10:].mean()
        resistance = h.rolling(5).max().iloc[-10:].mean()
        range_pct = (resistance - support) / support
        
        # Consolidation range
        if range_pct < 0.03 and c.iloc[-1] < support:
            return True
    
    # Bearish Wedge (Rising Wedge)
    if len(c) >= 20:
        # Converging trendlines both sloping upward
        highs = h.iloc[-20:]
        lows = l.iloc[-20:]
        
        high_slope, _ = np.polyfit(range(20), highs, 1)
        low_slope, _ = np.polyfit(range(20), lows, 1)
        
        # Both should be positive but lows slope less positive (converging)
        if high_slope > 0 and low_slope > 0 and high_slope > low_slope:
            # Breakdown below lower trendline
            if c.iloc[-1] < min(lows.iloc[:-1]):
                return True
    
    # Rounded Top
    if len(c) >= 30:
        # Fit polynomial curve to highs
        idx = np.array(range(30))
        highs = h.iloc[-30:].values
        coeffs = np.polyfit(idx, highs, 2)
        
        # Check inverted U-shape (negative quadratic coefficient)
        if coeffs[0] < 0:
            # Check if current price is below starting point
            start_price = highs[0]
            if c.iloc[-1] < start_price:
                return True
    
    # Bearish Channel (Downward Channel)
    if len(c) >= 20:
        # Get highs and lows of last 20 candles
        highs = h.iloc[-20:]
        lows = l.iloc[-20:]
        idx = np.array(range(20))
        
        # Fit trendlines to highs and lows
        high_slope, high_intercept = np.polyfit(idx, highs, 1)
        low_slope, low_intercept = np.polyfit(idx, lows, 1)
        
        # Check both trendlines are declining
        if high_slope < 0 and low_slope < 0:
            # Check channel is parallel (similar slopes)
            slope_diff = abs(high_slope - low_slope)
            if slope_diff / min(abs(high_slope), abs(low_slope)) < 0.25:
                # Calculate current channel boundaries
                current_high_bound = high_slope * 19 + high_intercept
                current_low_bound = low_slope * 19 + low_intercept
                
                # Confirm price is in channel and showing weakness
                if (h.iloc[-1] < current_high_bound and 
                    l.iloc[-1] > current_low_bound and
                    c.iloc[-1] < current_low_bound * 1.01):  # Closing near bottom
                    return True
    
    # Distribution Zone (Volume-based pattern)
    if len(c) >= 15:
        # Higher volume on down days than up days
        up_days_vol = 0
        down_days_vol = 0
        up_days = 0
        down_days = 0
        
        for i in range(1, 15):
            if c.iloc[-i] > c.iloc[-i-1]:
                up_days_vol += v.iloc[-i]
                up_days += 1
            else:
                down_days_vol += v.iloc[-i]
                down_days += 1
        
        if down_days > 0 and up_days > 0:
            avg_down_vol = down_days_vol / down_days
            avg_up_vol = up_days_vol / up_days
            if avg_down_vol > avg_up_vol * 1.3:
                return True
                    
    return False

def log_trade(symbol, security_id, action, price, qty, status, stop_pct=None, target_pct=None, stop_price=None, target_price=None, order_id="N/A", timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%m/%d/%Y %H:%M:%S")
    
    # CSV row log
    log_row = [
        symbol,
        timestamp_str,
        security_id,
        qty,
        price,
        0,  # momentum_5min placeholder
        target_pct if target_pct is not None else "",
        stop_pct if stop_pct is not None else "",
        "",  # live_price
        "",  # change_pct
        "",  # last_checked
        status,  # ✅ Correctly placed here
        "",  # exit_price
        order_id,
        target_price if target_price is not None else "",
        stop_price if stop_price is not None else ""
    ]

    # CSV logging
    with open(PORTFOLIO_LOG, "a") as f:
        f.write(",".join(map(str, log_row)) + "\n")

    # DB logging
    try:
        insert_portfolio_log_to_db(
            trade_date=timestamp,
            symbol=symbol,
            security_id=security_id,
            quantity=qty,
            buy_price=price,
            stop_pct=float(stop_pct) if stop_pct is not None else None,
            target_pct=float(target_pct) if target_pct is not None else None,
            stop_price=stop_price,
            target_price=target_price,
            status=status,
            order_id=order_id
        )
    except Exception as e:
        print("❌ DB log failed (portfolio_log):", e)

def place_exit_order(symbol, security_id, qty, tick_size_map):
    """Place exit order for position monitoring"""
    try:
        quote = dhan.get_quote(security_id)
        if not quote:
            return
            
        price = float(quote.get('ltp', 0))
        if price <= 0:
            print(f"⚠️ Invalid price for {symbol}: {price}")
            return
            
        tick_size_value = tick_size_map.get(str(security_id), 0.05)
        tick_size_dec = Decimal(str(tick_size_value))
        
        limit_price = Decimal(str(price)) * Decimal("0.998")
        limit_price = (limit_price / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec
        limit_price = float(limit_price)
        
        # Cancel existing SL/TP forever order if any
        try:
            dhan.cancel_forever_order(
                security_id=str(security_id),
                transaction_type="SELL",
                exchange_segment="NSE_EQ",
                product_type="CNC"
            )
            print(f"🧹 Cancelled existing SL/TP for {symbol}")
        except Exception as e:
            print(f"⚠️ Failed to cancel SL/TP for {symbol}: {e}")

        # Proceed with manual SELL
        order = {
            "security_id": str(security_id),
            "exchange_segment": "NSE_EQ",
            "transaction_type": "SELL",
            "order_type": "LIMIT",
            "product_type": "CNC",
            "quantity": qty,
            "price": limit_price,
            "validity": "DAY"
        }

        res = dhan.place_order(**order)
        if res.get('status') == 'REJECTED':
            print(f"❌ Exit order rejected for {symbol}: {res.get('message', 'No message')}")
            send_telegram(f"❌ Exit order rejected for {symbol}: {res.get('message', 'No message')}")
            return
            
        print(f"✅ Exit order placed for {symbol}: {res}")
        send_telegram(f"⚠️ Exiting {symbol} due to reversal signal @ ₹{limit_price:.2f}")
        log_trade(symbol, security_id, "SELL", limit_price, qty, "EXITED")
        exit_reason = f"{symbol} exited via manual SELL due to reversal pattern or stop/target logic"
        log_time = datetime.now()
        
        try:
            # CSV Logging
            with open("D:/Downloads/Dhanbot/dhan_autotrader/bot_execution_log.csv", "a") as flog:
                flog.write(f"{log_time.strftime('%Y-%m-%d %H:%M:%S')},autotrade.py,SUCCESS,\"{exit_reason}\"\n")
        
            # DB Logging
            log_to_postgres(
                timestamp=log_time,
                script="autotrade.py",
                status="SELL",
                message=exit_reason
            )
        except Exception as e:
            print(f"⚠️ Failed to log SELL reason: {e}")
            
    except Exception as e:
        print(f"❌ Exit order failed for {symbol}: {e}")
        send_telegram(f"❌ Exit order failed for {symbol}: {e}")

def monitor_hold_position(cache=None, tick_size_map=None):
    if cache is None:
        cache = {}
    if tick_size_map is None:
        tick_size_map = {}
    if not os.path.exists(PORTFOLIO_LOG):
        return
        
    try:
        df = pd.read_csv(PORTFOLIO_LOG)
        today = datetime.now().strftime("%m/%d/%Y")
        hold_positions = df[(df['status'] == "HOLD") & (df['timestamp'].str.contains(today))]
        
        # Check order status for executed SL/TP orders
        try:
            order_list = dhan.get_order_list()
            orders = []
            if isinstance(order_list, dict) and 'data' in order_list:
                orders = order_list['data']
            elif isinstance(order_list, list):
                orders = order_list
                
            for order in orders:
                if order.get('orderStatus') in ['FILLED', 'TRADED']:
                        security_id = str(order['securityId'])
                        if security_id in hold_positions['security_id'].values:
                            idx = df[
                                (df['security_id'] == security_id) &
                                (df['status'] == "HOLD")
                            ].index[-1]
            
                            exit_price = float(order.get('orderAverageTradedPrice', 0)) or float(order.get('orderPrice', 0))
                            entry_price = float(df.at[idx, 'price'])
                            status = "PROFIT" if exit_price > entry_price else "STOP LOSS"
            
                            df.at[idx, 'exit_price'] = exit_price
                            df.at[idx, 'status'] = status
                            df.at[idx, 'last_checked'] = datetime.now().strftime("%H:%M:%S")
                            
                            symbol = df.at[idx, 'symbol']
                            print(f"✅ SL/TP executed for {symbol} ➝ {status} @ ₹{exit_price:.2f} (Order ID: {order['orderId']})")
            
                            # ✅ Update DB as well
                            try:
                                update_portfolio_log_to_db(
                                    security_id=security_id,
                                    exit_price=exit_price,
                                    status=status
                                )
                            except Exception as e:
                                print(f"❌ DB update failed: {e}")
        except Exception as e:
            print(f"⚠️ Order status check failed: {e}")
        
        # Save updated status to CSV
        df.to_csv(PORTFOLIO_LOG, index=False)
        
        # Process positions still in HOLD status
        hold_positions = df[(df['status'] == "HOLD") & (df['timestamp'].str.contains(today))]
        for _, pos in hold_positions.iterrows():
            security_id = pos['security_id']
            symbol = pos['symbol']
            entry_price = float(pos['price'])
            qty = pos['qty']
            
            candles = fetch_candles(security_id, count=40, cache=cache)
            if not candles:
                continue
                
            # Get current price
            try:
                quote = dhan.get_quote(security_id)
                current_price = float(quote.get('ltp', 0))
            except:
                current_price = candles[-1]['close']
                
            # Time-based exit (after 3:15 PM)
            current_time = datetime.now().time()
            if current_time >= dtime(15, 15):
                status = "PROFIT" if current_price > entry_price else "STOP LOSS"
                print(f"⏰ Time-based exit for {symbol} ({status})")
                place_exit_order(symbol, security_id, qty, tick_size_map)
                continue
                
            # Trailing stop logic (move stop to breakeven at 1% profit)
            if current_price > entry_price * 1.01:
                # Check if we need to move stop to breakeven
                if 'stop_price' in pos and float(pos['stop_price']) < entry_price:
                    print(f"📈 Moving stop to breakeven for {symbol}")
                    try:
                        # Cancel existing stop
                        dhan.cancel_forever_order(
                            security_id=str(security_id),
                            transaction_type="SELL",
                            exchange_segment="NSE_EQ",
                            product_type="CNC"
                        )
                        # Place new stop at breakeven
                        tick_size_value = tick_size_map.get(str(security_id), 0.05)
                        tick_size_dec = Decimal(str(tick_size_value))
                        stop_price = entry_price * 0.999
                        stop_price = float((Decimal(str(stop_price)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
                        
                        dhan.place_forever(
                            security_id=str(security_id),
                            exchange_segment="NSE_EQ",
                            transaction_type="SELL",
                            product_type="CNC",
                            quantity=qty,
                            price=current_price,  # Target remains same
                            trigger_Price=stop_price
                        )
                        print(f"🔒 Trailing stop activated for {symbol} @ ₹{stop_price:.2f}")
                    except Exception as e:
                        print(f"⚠️ Failed to adjust trailing stop: {e}")
            
            # Check for bearish reversal candlestick patterns
            bearish_patterns = [
                "Shooting Star", "Bearish Engulfing", "Evening Star",
                "Bearish Harami", "Three Black Crows", "Bearish Kicker",
                "Gravestone Doji", "Gap-Up Reversal", "Hanging Man", "Breakdown Marubozu"
            ]
            for pattern in bearish_patterns:
                if detect_reversal_pattern(candles, pattern):
                    place_exit_order(symbol, security_id, qty, tick_size_map)
                    break
                    
            # Check for bearish chart patterns
            if detect_bearish_chart_pattern(candles):
                place_exit_order(symbol, security_id, qty, tick_size_map)
                continue
    except Exception as e:
        print(f"❌ Position monitoring failed: {e}")

def main():
    print('📌 Starting position monitoring')
    
    # Precompute market close time
    monitoring_end_time = datetime.combine(datetime.today(), dtime(15, 25))
    
    # Load master CSV for tick size map
    master_df = pd.read_csv(MASTER_CSV)
    tick_size_map = dict(zip(
        master_df['SEM_SMST_SECURITY_ID'].astype(str), 
        master_df['SEM_TICK_SIZE']
    ))
    
    # Continuous monitoring loop until market close
    while datetime.now() < monitoring_end_time:
        candle_cache = {}  # Reset cache for each iteration
        print(f"\n⏰ New monitoring cycle at {datetime.now().strftime('%H:%M:%S')}")
        
        # Monitor existing positions
        monitor_hold_position(cache=candle_cache, tick_size_map=tick_size_map)
        
        # Wait before next scan
        print("🔄 Waiting for next monitoring cycle in 60 seconds...")
        time.sleep(60)
    
    print("🛑 Monitoring window closed - stopping script")

# Save execution log
try:
    log_path = "D:/Downloads/Dhanbot/dhan_autotrader/Logs/New_Autotrade_Monitor.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(log_buffer.getvalue())
except Exception as e:
    print(f"⚠️ Failed to write log file: {e}")

if __name__ == "__main__":
    main()