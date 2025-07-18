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
from db_logger import insert_portfolio_log_to_db
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

# ========== Telegram from config ==========
TG_TOKEN = config["telegram_token"]
TG_CHAT_ID = config["telegram_chat_id"]

# ========== Pattern Confidence Weights ==========
PATTERN_WEIGHTS = {
    # Candlestick patterns
    "Bullish Hammer": {"weight": 1.1, "vol_scale": 1.2},
    "Bullish Engulfing": {"weight": 1.3, "vol_scale": 1.5},
    "Piercing Line": {"weight": 1.2, "vol_scale": 1.3},
    "Morning Star": {"weight": 1.5, "vol_scale": 3.0},
    "Inverted Hammer": {"weight": 1.1, "vol_scale": 1.2},
    "Bullish Harami": {"weight": 1.1, "vol_scale": 1.1},
    "Three White Soldiers": {"weight": 1.3, "vol_scale": 1.5},
    "Bullish Kicker": {"weight": 1.6, "vol_scale": 2.5},
    "Breakout Marubozu": {"weight": 1.4, "vol_scale": 1.8},
    "Volume Breakout Candle": {"weight": 1.4, "vol_scale": 2.0},
    "Gap-Down Reversal": {"weight": 1.5, "vol_scale": 2.0},
    # Chart patterns
    "Cup and Handle": {"weight": 1.7, "vol_scale": 2.0},
    "Double Bottom": {"weight": 1.5, "vol_scale": 1.8},
    "Triple Bottom": {"weight": 1.6, "vol_scale": 1.9},
    "Ascending Triangle": {"weight": 1.5, "vol_scale": 1.7},
    "Bullish Pennant": {"weight": 1.4, "vol_scale": 2.0},
    "Bullish Wedge (Falling Wedge)": {"weight": 1.4, "vol_scale": 1.8},
    "Rounding Bottom": {"weight": 1.3, "vol_scale": 1.6},
    "Inverse Head and Shoulders": {"weight": 1.8, "vol_scale": 2.2},
    "Rounded Consolidation (Roundboom)": {"weight": 1.3, "vol_scale": 1.5},
    "Bullish Rectangle": {"weight": 1.4, "vol_scale": 1.7},
    "Bullish Flag": {"weight": 1.5, "vol_scale": 1.8},  # New pattern
    "Symmetrical Triangle": {"weight": 1.4, "vol_scale": 1.6},  # New pattern
    # Bearish patterns
    "Head and Shoulders": {"weight": 1.8, "vol_scale": 2.0},
    "Double Top": {"weight": 1.6, "vol_scale": 1.8},
    "Triple Top": {"weight": 1.7, "vol_scale": 1.9},
    "Descending Triangle": {"weight": 1.5, "vol_scale": 1.7},
    "Bearish Rectangle": {"weight": 1.4, "vol_scale": 1.7},
    "Bearish Wedge (Rising Wedge)": {"weight": 1.4, "vol_scale": 1.8},
    "Rounded Top": {"weight": 1.3, "vol_scale": 1.6},
    "Distribution Zone": {"weight": 1.3, "vol_scale": 1.5}
}

def log_pattern_detection(symbol, pattern_name, detected, reason=""):
    """Log pattern detection result for debugging"""
    prefix = f"{symbol}: " if symbol else ""
    if detected:
        print(f"✅ {prefix}Pattern {pattern_name} detected")
    else:
        if reason:
            print(f"❌ {prefix}Pattern {pattern_name} not detected - {reason}")
        else:
            print(f"❌ {prefix}Pattern {pattern_name} not detected")

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
        from_dt = datetime.now() - timedelta(days=ADR_PERIOD + 5)
        to_dt = datetime.now()
        
        # Create new context for each request to prevent staleness
        with DhanContext(config["client_id"], config["access_token"]) as context:
            dhan = dhanhq(context)
            daily_data = dhan.historical_minute_data(
                security_id=str(security_id), 
                exchange_segment="NSE_EQ", 
                instrument_type="EQUITY", 
                from_date=from_dt.strftime("%Y-%m-%d"), 
                to_date=to_dt.strftime("%Y-%m-%d"),
                type='day'
            )
        
        if not daily_data or not isinstance(daily_data, dict) or 'data' not in daily_data:
            return 0.0
            
        data = daily_data.get('data', {})
        if not data or not all(key in data for key in ['high', 'low']) or len(data['high']) < ADR_PERIOD:
            return 0.0
            
        daily_ranges = [abs(h - l) for h, l in zip(data['high'], data['low'])][-ADR_PERIOD:]
        return sum(daily_ranges) / len(daily_ranges) if daily_ranges else 0.0
    except Exception as e:
        print(f"❌ ADR calculation failed: {e}")
        return 0.0

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
    cache_key = f"{security_id}_{count}_{exchange_segment}"
    if cache_key in cache:
        return cache[cache_key]

    for attempt in range(3):  # Retry up to 3 times
        try:
            # Create new context for each request to prevent staleness
            with DhanContext(config["client_id"], config["access_token"]) as context:
                dhan = dhanhq(context)
                
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

def detect_bullish_pattern(candles, symbol=None):
    """Enhanced pattern detection with multi-pattern scoring"""
    if not candles or len(candles) < 5:
        return False, None, 0.0
    
    df = pd.DataFrame(candles)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    detected_patterns = []  # Store all detected patterns with scores
    
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

    # =====================
    # CHART PATTERN DETECTION
    # =====================
    
    # 1. Cup and Handle pattern
    if len(c) >= 40:
        # Find left rim (high point before cup)
        left_rim_idx = -40
        left_rim_high = h.iloc[left_rim_idx]
        
        # Find cup bottom (lowest point in the cup)
        cup_bottom_idx = h.iloc[-35:-15].idxmin()
        cup_bottom = l.iloc[cup_bottom_idx]
        
        # Find right rim (high point after cup)
        right_rim_high = h.iloc[-15]
        
        # Handle formation (small downward drift)
        handle_lows = l.iloc[-15:-5]
        handle_highs = h.iloc[-15:-5]
        
        # Cup should be U-shaped, handle should be downward drift
        if (left_rim_high * 0.98 < right_rim_high < left_rim_high * 1.02 and
            cup_bottom < left_rim_high * 0.85 and
            max(handle_highs) < right_rim_high and
            min(handle_lows) > cup_bottom and
            c.iloc[-1] > right_rim_high):  # Close above resistance
            
            vol_ok, vol_ratio = volume_confirmed(index=-1, multiplier=1.8)
            if vol_ok:
                pattern_score = min(1.0, 0.8 + (vol_ratio / 5.0))
                detected_patterns.append(("Cup and Handle", pattern_score))
                log_pattern_detection(symbol, "Cup and Handle", True)
            else:
                log_pattern_detection(symbol, "Cup and Handle", False, f"Volume insufficient ({vol_ratio:.2f} < 1.8x)")
    
    # 2. Double/Triple Bottom
    if len(c) >= 20:
        # Find significant lows
        lows = l.rolling(5).min().dropna()
        min_idxs = lows.nsmallest(3).index.tolist()
        min_idxs.sort()
        
        if len(min_idxs) >= 2:
            # Check if lows are approximately equal
            low1 = lows.iloc[min_idxs[0]]
            low2 = lows.iloc[min_idxs[1]]
            low_diff = abs(low1 - low2) / min(low1, low2)
            
            # Check breakout above resistance (neckline)
            resistance = h.iloc[min_idxs[0]:min_idxs[1]].max()
            if (low_diff < 0.02 and 
                c.iloc[-1] > resistance):  # Close above resistance
                
                vol_ok, vol_ratio = volume_confirmed(multiplier=1.5)
                if vol_ok:
                    pattern_name = "Double Bottom" if len(min_idxs) == 2 else "Triple Bottom"
                    pattern_score = min(1.0, 0.7 + (c.iloc[-1] - resistance) / resistance * 5)
                    detected_patterns.append((pattern_name, pattern_score))
                    log_pattern_detection(symbol, pattern_name, True)
                else:
                    log_pattern_detection(symbol, pattern_name, False, f"Volume insufficient ({vol_ratio:.2f} < 1.5x)")
    
    # 3. Ascending Triangle
    if len(c) >= 15:
        # Horizontal resistance
        resistance = h.rolling(5).max().iloc[-15:].mean()
        resistance_range = h.iloc[-15:].max() - h.iloc[-15:].min()
        
        # Rising lows
        low_min = l.iloc[-15:].min()
        low_max = l.iloc[-15:].max()
        slope, _ = np.polyfit(range(15), l.iloc[-15:], 1)
        
        if (resistance_range / resistance < 0.02 and
            slope > 0 and
            (low_max - low_min) / low_min > 0.03 and
            c.iloc[-1] > resistance):  # Close above resistance
            
            vol_ok, vol_ratio = volume_confirmed(multiplier=1.4)
            if vol_ok:
                pattern_score = min(1.0, 0.75 + slope * 100)
                detected_patterns.append(("Ascending Triangle", pattern_score))
                log_pattern_detection(symbol, "Ascending Triangle", True)
            else:
                log_pattern_detection(symbol, "Ascending Triangle", False, f"Volume insufficient ({vol_ratio:.2f} < 1.4x)")
    
    # 4. Inverse Head and Shoulders
    if len(c) >= 25:
        # Find left shoulder
        left_shoulder_idx = h.iloc[-25:-15].idxmax()
        left_shoulder = h.iloc[left_shoulder_idx]
        
        # Find head (lowest point)
        head_idx = l.iloc[-20:-10].idxmin()
        head_low = l.iloc[head_idx]
        
        # Find right shoulder
        right_shoulder_idx = h.iloc[-10:-5].idxmax()
        right_shoulder = h.iloc[right_shoulder_idx]
        
        # Neckline (resistance)
        neckline = (left_shoulder + right_shoulder) / 2
        
        if (left_shoulder * 0.98 < right_shoulder < left_shoulder * 1.02 and
            head_low < left_shoulder * 0.95 and
            c.iloc[-1] > neckline):  # Close above neckline
            
            vol_ok, vol_ratio = volume_confirmed(multiplier=1.6)
            if vol_ok:
                pattern_score = min(1.0, 0.85 + (c.iloc[-1] - neckline) / neckline * 10)
                detected_patterns.append(("Inverse Head and Shoulders", pattern_score))
                log_pattern_detection(symbol, "Inverse Head and Shoulders", True)
    
    # 5. Bullish Rectangle
    if len(c) >= 10:
        # Horizontal support and resistance
        support = l.rolling(5).min().iloc[-10:].mean()
        resistance = h.rolling(5).max().iloc[-10:].mean()
        range_pct = (resistance - support) / support
        
        # Consolidation range
        if range_pct < 0.03 and c.iloc[-1] > resistance:  # Close above resistance
            vol_ok, vol_ratio = volume_confirmed(multiplier=1.5)
            if vol_ok:
                pattern_score = min(1.0, 0.7 + vol_ratio / 2.0)
                detected_patterns.append(("Bullish Rectangle", pattern_score))
                log_pattern_detection(symbol, "Bullish Rectangle", True)
    
    # 6. Rounding Bottom
    if len(c) >= 30:
        # Fit polynomial curve to lows
        idx = np.array(range(30))
        lows = l.iloc[-30:].values
        coeffs = np.polyfit(idx, lows, 2)
        
        # Check U-shape (positive quadratic coefficient)
        if coeffs[0] > 0:
            # Check if current price is above starting point
            start_price = lows[0]
            if c.iloc[-1] > start_price:
                pattern_score = min(1.0, 0.65 + (c.iloc[-1] - start_price) / start_price * 20)
                detected_patterns.append(("Rounding Bottom", pattern_score))
                log_pattern_detection(symbol, "Rounding Bottom", True)
    
    # 7. Bullish Pennant
    if len(c) >= 20:
        # Flagpole: sharp price movement
        flagpole_start = c.iloc[-20]
        flagpole_end = c.iloc[-15]
        flagpole_move = abs(flagpole_end - flagpole_start) / flagpole_start
        
        if flagpole_move > 0.05:  # At least 5% move
            # Pennant: converging trendlines with lower highs and higher lows
            highs = h.iloc[-15:]
            lows = l.iloc[-15:]
            
            # Fit trendlines
            high_slope, _ = np.polyfit(range(15), highs, 1)
            low_slope, _ = np.polyfit(range(15), lows, 1)
            
            # Pennant should have downward sloping highs and upward sloping lows
            if high_slope < 0 and low_slope > 0:
                # Volume should decrease during pennant formation
                vol_start = v.iloc[-15]
                vol_end = v.iloc[-1]
                if vol_end < vol_start * 0.7:
                    pattern_score = min(1.0, 0.7 + (flagpole_move * 10))
                    detected_patterns.append(("Bullish Pennant", pattern_score))
                    log_pattern_detection(symbol, "Bullish Pennant", True)
    
    # 8. Bullish Wedge (Falling Wedge)
    if len(c) >= 20:
        # Converging trendlines both sloping downward
        highs = h.iloc[-20:]
        lows = l.iloc[-20:]
        
        high_slope, _ = np.polyfit(range(20), highs, 1)
        low_slope, _ = np.polyfit(range(20), lows, 1)
        
        # Both should be negative but lows slope less negative (converging)
        if high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope):
            # Breakout above upper trendline
            if c.iloc[-1] > max(highs.iloc[:-1]):  # Close above resistance
                vol_ok, vol_ratio = volume_confirmed(multiplier=1.3)
                if vol_ok:
                    pattern_score = min(1.0, 0.75 + (abs(high_slope) * 100))
                    detected_patterns.append(("Bullish Wedge (Falling Wedge)", pattern_score))
                    log_pattern_detection(symbol, "Bullish Wedge (Falling Wedge)", True)
    
    # 9. Rounded Consolidation (Roundboom)
    if len(c) >= 25:
        # Fit polynomial curve to closes
        idx = np.array(range(25))
        closes = c.iloc[-25:].values
        coeffs = np.polyfit(idx, closes, 2)
        
        # Check U-shape in consolidation (positive quadratic coefficient)
        if coeffs[0] > 0:
            # Consolidation range
            high_point = max(h.iloc[-25:])
            low_point = min(l.iloc[-25:])
            consolidation_range = (high_point - low_point) / low_point
            
            if consolidation_range < 0.08:  # Tight consolidation
                # Volume should be higher at the edges
                edge_vol = (v.iloc[-25] + v.iloc[-1]) / 2
                center_vol = v.iloc[-12:-8].mean()
                if edge_vol > center_vol * 1.5:
                    pattern_score = min(1.0, 0.7 + consolidation_range * 10)
                    detected_patterns.append(("Rounded Consolidation (Roundboom)", pattern_score))
                    log_pattern_detection(symbol, "Rounded Consolidation (Roundboom)", True)
    
    # 10. Bullish Flag (New pattern)
    if len(c) >= 15:
        # Flagpole: sharp price movement (at least 5% in 3-5 candles)
        flagpole_rise = 0
        flagpole_end_idx = None
        for i in range(5, 10):
            start_idx = -i-5
            end_idx = -i
            if start_idx < -len(c) or end_idx >= 0:
                continue
            rise = (c.iloc[end_idx] - c.iloc[start_idx]) / c.iloc[start_idx]
            if rise > flagpole_rise and rise > 0.05:
                flagpole_rise = rise
                flagpole_end_idx = end_idx
        
        if flagpole_end_idx is not None:
            # Flag: consolidation with decreasing volume
            flag_highs = h.iloc[flagpole_end_idx+1:-1]
            flag_lows = l.iloc[flagpole_end_idx+1:-1]
            if len(flag_highs) < 4:  # At least 4 candles in flag
                pass
            else:
                resistance = flag_highs.max()
                # Breakout with volume confirmation
                if c.iloc[-1] > resistance and v.iloc[-1] > v.iloc[-2] * 1.5:
                    flag_vol = v.iloc[flagpole_end_idx+1:-1].mean()
                    flagpole_vol = v.iloc[flagpole_end_idx-4:flagpole_end_idx+1].mean()
                    if flag_vol < flagpole_vol * 0.7:
                        pattern_score = min(1.0, 0.7 + flagpole_rise)
                        detected_patterns.append(("Bullish Flag", pattern_score))
                        log_pattern_detection(symbol, "Bullish Flag", True)
    
    # 11. Symmetrical Triangle (New pattern)
    if len(c) >= 20:
        # Converging trendlines (at least 10 candles)
        triangle_highs = h.iloc[-15:-1]
        triangle_lows = l.iloc[-15:-1]
        
        if len(triangle_highs) < 10:
            pass
        else:
            # Fit trendlines
            idx = np.array(range(len(triangle_highs)))
            high_slope, high_intercept = np.polyfit(idx, triangle_highs, 1)
            low_slope, low_intercept = np.polyfit(idx, triangle_lows, 1)
            
            # Slopes should converge (high slope negative, low slope positive)
            if high_slope < 0 and low_slope > 0:
                # Breakout above upper trendline
                current_idx = len(triangle_highs)
                upper_trendline = high_slope * current_idx + high_intercept
                if c.iloc[-1] > upper_trendline and v.iloc[-1] > v.iloc[-2] * 1.2:
                    pattern_score = min(1.0, 0.8 + (abs(high_slope) + abs(low_slope)) / 2 * 100)
                    detected_patterns.append(("Symmetrical Triangle", pattern_score))
                    log_pattern_detection(symbol, "Symmetrical Triangle", True)
    
    # =====================
    # CANDLESTICK PATTERN DETECTION
    # =====================
    
    # 1. Breakout Marubozu
    if len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        candle_range = h.iloc[-1] - l.iloc[-1]
        body_ratio = body / candle_range if candle_range > 0 else 0
        
        # Look for no wicks and closing at high
        if (body_ratio > 0.9 and 
            min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1] < candle_range * 0.05 and
            h.iloc[-1] - max(c.iloc[-1], o.iloc[-1]) < candle_range * 0.05 and
            c.iloc[-1] > o.iloc[-1] and
            check_breakout([candles[-1]], period=1)):  # New high breakout
            
            vol_ok, vol_ratio = volume_confirmed(multiplier=1.8)
            if vol_ok:
                pattern_score = min(1.0, 0.85 + body_ratio * 0.5)
                detected_patterns.append(("Breakout Marubozu", pattern_score))
                log_pattern_detection(symbol, "Breakout Marubozu", True)
    
    # 2. Volume Breakout Candle
    if len(c) >= 10:
        current_vol = v.iloc[-1]
        avg_vol = v.iloc[-10:-1].mean()
        if current_vol > 2.5 * avg_vol and c.iloc[-1] > o.iloc[-1]:
            # Check if price breaks recent high
            recent_high = max(h.iloc[-10:-1])
            if c.iloc[-1] > recent_high:  # Close above resistance
                pattern_score = min(1.0, 0.8 + (current_vol / avg_vol) / 10)
                detected_patterns.append(("Volume Breakout Candle", pattern_score))
                log_pattern_detection(symbol, "Volume Breakout Candle", True)
    
    # 3. Gap-Down Reversal (with next candle confirmation)
    if len(c) >= 3:
        # Check gap down between candle -2 and candle -1
        gap_down = o.iloc[-2] < c.iloc[-3]
        gap_size = (c.iloc[-3] - o.iloc[-2]) / c.iloc[-3] if gap_down else 0
        
        # Strong reversal candle (candle -1)
        strong_reversal = (c.iloc[-2] > o.iloc[-2] and 
                          (c.iloc[-2] - o.iloc[-2]) > (h.iloc[-2] - l.iloc[-2]) * 0.7)
        
        # Next candle (current candle) confirmation
        confirmation = (c.iloc[-1] > o.iloc[-1] and  # Bullish
                       c.iloc[-1] > c.iloc[-2] and   # Closes above previous close
                       v.iloc[-1] > v.iloc[-2])      # Volume increases
        
        if gap_down and gap_size > 0.01 and strong_reversal and confirmation:
            vol_ok, vol_ratio = volume_confirmed(index=-1, multiplier=1.7)
            if vol_ok:
                pattern_score = min(1.0, 0.75 + gap_size * 50)
                detected_patterns.append(("Gap-Down Reversal", pattern_score))
                log_pattern_detection(symbol, "Gap-Down Reversal", True)
    
    # -- Existing patterns with volume confirmation --
    # Bullish Engulfing
    if len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and 
            c.iloc[-1] > o.iloc[-1] and
            c.iloc[-1] > o.iloc[-2] and 
            o.iloc[-1] < c.iloc[-2]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_score = candle_score(-1) * 0.85
                detected_patterns.append(("Bullish Engulfing", pattern_score))
                log_pattern_detection(symbol, "Bullish Engulfing", True)
    
    # Hammer (single candle pattern)
    if len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
        uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        if lw > 2 * body and uw < body:
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_score = min(1.0, 0.75 * (lw / body))
                detected_patterns.append(("Bullish Hammer", pattern_score))
                log_pattern_detection(symbol, "Bullish Hammer", True)
    
    # Piercing Line
    if len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and 
            c.iloc[-1] > o.iloc[-1] and
            o.iloc[-1] < c.iloc[-2] and 
            c.iloc[-1] > (o.iloc[-2] + c.iloc[-2]) / 2):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_score = min(1.0, 0.8 * (c.iloc[-1] - o.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]))
                detected_patterns.append(("Piercing Line", pattern_score))
                log_pattern_detection(symbol, "Piercing Line", True)
    
    # Bullish Harami
    if len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and 
            c.iloc[-1] > o.iloc[-1] and
            o.iloc[-1] > c.iloc[-2] and 
            c.iloc[-1] < o.iloc[-2]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_score = candle_score(-1) * 0.75
                detected_patterns.append(("Bullish Harami", pattern_score))
                log_pattern_detection(symbol, "Bullish Harami", True)
    
    # Inverse Hammer
    if len(c) >= 1:
        body = abs(c.iloc[-1] - o.iloc[-1])
        uw = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        lw = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
        if uw > 2 * body and lw < body:
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_score = min(1.0, 0.7 * (uw / body))
                detected_patterns.append(("Inverted Hammer", pattern_score))
                log_pattern_detection(symbol, "Inverted Hammer", True)
    
    # Three White Soldiers
    if len(c) >= 3:
        if (c.iloc[-3] > o.iloc[-3] and 
            c.iloc[-2] > o.iloc[-2] and 
            c.iloc[-1] > o.iloc[-1] and
            c.iloc[-2] > c.iloc[-3] and 
            c.iloc[-1] > c.iloc[-2]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_score = min(1.0, 0.9 * (vol_ratio / 2.0))
                detected_patterns.append(("Three White Soldiers", pattern_score))
                log_pattern_detection(symbol, "Three White Soldiers", True)
    
    # Bullish Kicker
    if len(c) >= 2:
        if (c.iloc[-2] < o.iloc[-2] and
            o.iloc[-1] > c.iloc[-2] and
            c.iloc[-1] > o.iloc[-1]):
            vol_ok, vol_ratio = volume_confirmed()
            if vol_ok:
                pattern_score = candle_score(-1) * 0.85
                detected_patterns.append(("Bullish Kicker", pattern_score))
                log_pattern_detection(symbol, "Bullish Kicker", True)
                
    # Morning Star with Volume Confirmation
    if len(c) >= 3:
        body1 = o.iloc[-3] - c.iloc[-3]  # First candle body (bearish)
        body2 = abs(c.iloc[-2] - o.iloc[-2])
        if (c.iloc[-3] < o.iloc[-3] and
            o.iloc[-2] < c.iloc[-3] and
            body2 < 0.3 * body1 and
            c.iloc[-1] > o.iloc[-1] and
            o.iloc[-1] > c.iloc[-2] and
            c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2 and
            v.iloc[-1] > v.iloc[-3]):  # Volume confirmation: third candle > first candle
            vol_ok, vol_ratio = volume_confirmed(multiplier=1.5)  # Stricter volume check
            if vol_ok:
                pattern_score = min(1.0, 0.8 + (c.iloc[-1] - o.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]))
                detected_patterns.append(("Morning Star", pattern_score))
                print(f"🌟 Volume Confirmed Morning Star detected (Vol Ratio: {vol_ratio:.2f}x)")
                log_pattern_detection(symbol, "Morning Star", True)
    
    # =====================
    # PATTERN SELECTION LOGIC
    # =====================
    if detected_patterns:
        # Calculate composite scores (pattern weight * confidence score)
        scored_patterns = []
        for name, score in detected_patterns:
            weight = PATTERN_WEIGHTS.get(name, {"weight": 1.0})["weight"]
            composite_score = weight * score
            scored_patterns.append((name, score, composite_score))
        
        # Select pattern with highest composite score
        best_pattern = max(scored_patterns, key=lambda x: x[2])
        print(f"🏆 Selected pattern: {best_pattern[0]} (Score: {best_pattern[1]:.2f}, Composite: {best_pattern[2]:.2f})")
        return True, best_pattern[0], best_pattern[1]
    
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
    macd_crossover = macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
    return rsi.iloc[-1], histogram.iloc[-1], macd_crossover

def is_index_bullish(index_id):
    """Check bullishness for any index (NIFTY/Sector)"""
    candles = fetch_candles(
        index_id, 
        count=20,
        exchange_segment="NSE_INDEX",
        instrument_type="INDEX"
    )
    if not candles:
        return False
    closes = pd.Series([c["close"] for c in candles])
    rsi, macd_hist, macd_cross = compute_rsi_macd(closes)
    detected, _, _ = detect_bullish_pattern(candles)
    return detected and rsi > 50 and macd_hist > 0 and macd_cross

def check_breakout(candles, period=3):
    """Confirm 15-min high breakout with closing confirmation"""
    if len(candles) < period:
        return False
    current_close = candles[-1]['close']
    prev_high = max(c['high'] for c in candles[-period-1:-1])
    return current_close > prev_high

def check_gap_up(security_id):
    """Prevent entries after significant gap-ups"""
    try:
        # Create new context for each request to prevent staleness
        with DhanContext(config["client_id"], config["access_token"]) as context:
            dhan = dhanhq(context)
            quote = dhan.get_quote(security_id)
            if not quote:
                return False
            prev_close = quote.get('previousClose', 0)
            today_open = quote.get('open', 0)
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
            c.iloc[-1] < o.iloc[-1] and
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
        "", "", "", "",  # live_price, change_pct, last_checked, exit_price
        status,
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
        # Create new context for each request to prevent staleness
        with DhanContext(config["client_id"], config["access_token"]) as context:
            dhan = dhanhq(context)
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
            with DhanContext(config["client_id"], config["access_token"]) as context:
                dhan = dhanhq(context)
                order_book = dhan.order_book()
                if order_book and 'data' in order_book:
                    for order in order_book['data']:
                        if order['orderStatus'] in ['FILLED', 'TRADED']:
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
                                    from db_logger import update_portfolio_log_to_db
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
                with DhanContext(config["client_id"], config["access_token"]) as context:
                    dhan = dhanhq(context)
                    quote = dhan.get_quote(security_id)
                    current_price = float(quote.get('ltp', 0))
            except:
                current_price = candles[-1]['close']
                
            # Time-based exit (after 3:15 PM)
            current_time = datetime.now().time()
            if current_time >= dtime(15, 15) and current_price > entry_price:
                print(f"⏰ Time-based exit for {symbol} (profit secured)")
                place_exit_order(symbol, security_id, qty, tick_size_map)
                continue
                
            # Trailing stop logic (move stop to breakeven at 1% profit)
            if current_price > entry_price * 1.01:
                # Check if we need to move stop to breakeven
                if 'stop_price' in pos and float(pos['stop_price']) < entry_price:
                    print(f"📈 Moving stop to breakeven for {symbol}")
                    try:
                        with DhanContext(config["client_id"], config["access_token"]) as context:
                            dhan = dhanhq(context)
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
                                security_id=str(security_id)),
                                exchange_segment="NSE_EQ",
                                transaction_type="SELL",
                                product_type="CNC",
                                quantity=qty,
                                price=current_price,  # Target remains same
                                trigger_price=stop_price
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

def place_order(symbol, security_id, qty, price, pattern_name, candles, tick_size_map):
    if price <= 0:
        print(f"⚠️ Invalid price for {symbol}: {price}")
        return
        
    tick_size_value = tick_size_map.get(str(security_id), 0.05)
    tick_size_dec = Decimal(str(tick_size_value))
    
    limit_price = Decimal(str(price)) * Decimal("1.002")
    limit_price = (limit_price / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec
    limit_price = float(limit_price)
    
    # Calculate ADR for realistic TP/SL capping
    adr = calculate_adr(security_id)
    
    order = {
        "security_id": str(security_id),
        "exchange_segment": "NSE_EQ",
        "transaction_type": "BUY",
        "order_type": "LIMIT",
        "product_type": "CNC",
        "quantity": qty,
        "price": limit_price,
        "validity": "DAY"
    }
    
    try:
        # Create new context for each request to prevent staleness
        with DhanContext(config["client_id"], config["access_token"]) as context:
            dhan = dhanhq(context)
            res = dhan.place_order(**order)
            
            if res.get('status') == 'REJECTED':
                print(f"❌ Order rejected for {symbol}: {res.get('message', 'No message')}")
                send_telegram(f"❌ Order rejected for {symbol}: {res.get('message', 'No message')}")
                return
                
            print("✅ Order Placed:", res)
            msg = f"✅ BUY {symbol} Qty: {qty} @ ₹{limit_price}"
            if pattern_name:
                msg += f" | Pattern: {pattern_name}"
            send_telegram(msg)
            now = datetime.now()
            log_trade(symbol, security_id, "BUY", limit_price, qty, "HOLD", timestamp=now)

            # Enhanced Stop Loss and Target via Forever Order
            try:
                # Base parameters
                base_sl_pct = 0.005
                base_tp_pct = 0.01
                
                # Pattern-specific adjustments
                pattern_conf = PATTERN_WEIGHTS.get(pattern_name, {"weight": 1.0, "vol_scale": 1.0})
                conf_weight = pattern_conf["weight"]
                vol_scale = pattern_conf["vol_scale"]
                    
                base_tp_pct = 0.01 * conf_weight
                base_sl_pct = 0.005 * (2 - conf_weight/2)  # Inverse to weight
                    
                # Volatility adjustment using ATR
                atr = calculate_atr(candles)
                entry_price = Decimal(str(price))
                atr_multiplier = vol_scale * (atr / float(entry_price)) if atr > 0 else 1.0
                
                # Apply volatility scaling
                tp_pct = max(base_tp_pct, atr_multiplier)
                sl_pct = min(base_sl_pct, atr_multiplier * 0.7)
                
                # Time decay adjustment for late entries
                market_close = dtime(15, 30)
                now_time = datetime.now().time()
                remaining_seconds = (datetime.combine(datetime.today(), market_close) - 
                                    datetime.combine(datetime.today(), now_time)).total_seconds()
                remaining_hours = max(0.1, remaining_seconds / 3600)
                time_decay = max(0.5, remaining_hours / 6.5)  # 6.5 trading hours
                tp_pct *= time_decay
                
                # Ensure minimum 1:2 risk-reward ratio
                if tp_pct / sl_pct < 2:
                    tp_pct = sl_pct * 2.2  # Add small buffer
                    
                # Calculate final SL and TP
                stop_loss = float(entry_price * (Decimal(1) - Decimal(sl_pct)))
                target = float(entry_price * (Decimal(1) + Decimal(tp_pct)))
                
                # Apply ADR capping
                max_move = adr * 0.3  # Allow up to 30% of ADR
                target = min(target, price + max_move)
                stop_loss = max(stop_loss, price - max_move * 0.7)
                
                # Round to nearest tick
                stop_loss = float((Decimal(str(stop_loss)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
                target = float((Decimal(str(target)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
                
                # Time feasibility check - don't set unrealistic targets
                required_move = (target - price) / price
                max_allowed_move = 0.015 * (remaining_hours / 1.5)  # Max 1.5% per hour
                if required_move > max_allowed_move:
                    target = price * (1 + max_allowed_move)
                    target = float((Decimal(str(target)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)
                    send_telegram(f"⚠️ Adjusted {symbol} target to ₹{target:.2f} for time constraints")

                # Special handling for Morning Star pattern - confirm BEFORE placing SL/TP
                if pattern_name == "Morning Star":
                    # Add confirmation check
                    time.sleep(2)  # Wait for next candle
                    next_candle = fetch_candles(security_id, count=1)
                    if next_candle and (
                        next_candle[0]['close'] > candles[-1]['close'] and 
                        next_candle[0]['volume'] > candles[-1]['volume']
                    ):
                        tp_pct *= 1.2
                        target = float(entry_price * (1 + tp_pct))
                        target = float((Decimal(str(target)) / tick_size_dec).quantize(0, rounding=ROUND_HALF_UP) * tick_size_dec)               
                        print(f"🌟 Morning Star confirmation - Increased target to ₹{target:.2f}")

                # Small delay to avoid overlap
                time.sleep(1.5)

                dhan.place_forever(
                    security_id=str(security_id),
                    exchange_segment="NSE_EQ",
                    transaction_type="SELL",
                    product_type="CNC",
                    quantity=qty,
                    price=target,
                    trigger_price=stop_loss
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
    
    # Load master CSV once at start
    master_df = pd.read_csv(MASTER_CSV)
    print('📊 Loaded master CSV for index checks')
    
    # Create tick size map per security ID
    tick_size_map = dict(zip(
        master_df['SEM_SMST_SECURITY_ID'].astype(str), 
        master_df['SEM_TICK_SIZE']
    ))
    
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
    nifty50_row = master_df[
        master_df["SM_SYMBOL_NAME"].str.upper().str.contains("NIFTY") & 
        ~master_df["SM_SYMBOL_NAME"].str.upper().str.contains("ETF")
    ].head(1)
    
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
        monitor_hold_position(cache=candle_cache, tick_size_map=tick_size_map)
        if has_hold():
            print("⏩ Active hold exists - waiting for exit signal")
            time.sleep(300)
            continue
        
        # ⏳ After profit exit, don't re-enter if it's too late
        current_time = datetime.now().time()
        cutoff_time = dtime(13, 40)
        if not has_hold() and current_time >= cutoff_time:
            print("🛑 Too late for safe new entries. Ending session.")
            send_telegram("✅ Profit secured. No new trades after 1:40 PM. Ending autotrade.")
            break
        
        try:
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
            candidates = []
        
            for index, row in df.iterrows():
                try:
                    if datetime.now() >= end_time:
                        print("⏰ Time window expired - stopping evaluation")
                        break
        
                    symbol = row["symbol"]
                    secid = row["security_id"]
                    sector = row.get("sector", "UNKNOWN")
                    print(f'➡️ Evaluating {symbol} ({sector} sector)')
        
                    # Skip bearish sector in weak market
                    if not nifty_bullish and not sector_status.get(sector, True):
                        print(f'  📉 Sector {sector} bearish - skipping in weak market')
                        continue
        
                    # Fetch candles with rate limit control
                    try:
                        candles = fetch_candles(secid, count=40, cache=candle_cache)  # Increased for chart patterns
                        time.sleep(1)
                        if not candles or len(candles) < 5:
                            print('⚠️ No candle data available, skipping...')
                            continue
                    except Exception as e:
                        if "Rate_Limit" in str(e):
                            print("⚠️ Rate limit hit, waiting 60 seconds...")
                            time.sleep(60)
                            continue
                        raise
        
                    # Gap-up filter
                    if check_gap_up(secid):
                        print(f'⏫ Gap-up detected: {symbol}')
                        continue
        
                    # Bullish pattern detection
                    detected, pattern_name, pattern_score = detect_bullish_pattern(candles, symbol)
                    if not detected:
                        print('📉 No bullish pattern detected, skipping...')
                        continue
        
                    # Breakout confirmation
                    if not check_breakout(candles):
                        print(f'❌ No breakout confirmation: {symbol}')
                        continue
        
                    # Technical indicators
                    closes = pd.Series([c["close"] for c in candles])
                    rsi, macd_hist, macd_cross = compute_rsi_macd(closes)
                    print(f'📊 RSI: {rsi:.2f}, MACD Hist: {macd_hist:.4f}, MACD Cross: {macd_cross}')
                    if not (45 < rsi < 70 and macd_hist > 0 and macd_cross):
                        print('❌ RSI/MACD filter failed, skipping...')
                        continue
        
                    # Position sizing
                    price = closes.iloc[-1]
                    if price <= 0:
                        print(f'⚠️ Invalid price for {symbol}: {price}, skipping...')
                        continue
                    base_qty = int(capital // price)
                    confidence = PATTERN_WEIGHTS.get(pattern_name, {"weight": 1.0})["weight"]
                    adj_qty = max(1, int(base_qty * confidence * pattern_score))
                    profit_potential = price * adj_qty * confidence * pattern_score
        
                    print(f'💸 Final Price: ₹{price:.2f}, Base Qty: {base_qty}, Adj Qty: {adj_qty}, Confidence: {confidence:.2f}, Potential: ₹{profit_potential:.2f}')
                    if adj_qty <= 0:
                        print('⛔ Quantity is zero or negative, skipping...')
                        continue
        
                    # 🔁 Early exit if strong confidence detected
                    if pattern_score > 0.9 and confidence >= 1.5:
                        print(f"🚨 High-confidence trade found early: {symbol}")
                        place_order(symbol, secid, adj_qty, price, pattern_name, candles, tick_size_map)
                        send_telegram(f"✅ High-confidence order placed for {symbol} with Qty: {adj_qty}. Exiting autotrade loop.")
                        order_placed = True
                        print("✅ Order placed. Continuing monitoring for reversals.")
                        break
        
                    # Add to ranked candidates with pattern weight
                    pattern_weight = PATTERN_WEIGHTS.get(pattern_name, {"weight": 1.0})["weight"]
                    priority_score = pattern_weight * pattern_score * confidence
                    
                    candidates.append({
                        "symbol": symbol,
                        "security_id": secid,
                        "qty": adj_qty,
                        "price": price,
                        "pattern": pattern_name,
                        "candles": candles,
                        "score": pattern_score,
                        "confidence": confidence,
                        "potential": profit_potential,
                        "priority": priority_score  # New priority score
                    })
        
                except Exception as e:
                    print(f'⚠️ {row.get("symbol", "UNKNOWN")} evaluation failed: {str(e)}')
                    continue
        
            # Pick best from ranked candidates if any
            if candidates:
                # Rank by potential profit then confidence
                best = sorted(candidates, key=lambda x: (x["priority"], x["potential"]), reverse=True)[0]
                print(f"🚀 Best pick: {best['symbol']} with Qty: {best['qty']} and Potential: ₹{best['potential']:.2f}")
                place_order(best["symbol"], best["security_id"], best["qty"], best["price"], best["pattern"], best["candles"], tick_size_map)
                send_telegram(f"✅ Order placed for {best['symbol']} with Qty: {best['qty']}. Exiting autotrade loop.")
                order_placed = True
                print("✅ Order placed. Continuing monitoring for reversals.")
                break
            else:
                print("❌ No valid trades found this cycle")
                send_telegram("❌ No valid trades found this cycle")
        
            # Just in case order_placed is still False
            if not order_placed:
                print("❌ No trade placed this round.")
        
            # Wait before next scan
            print("🔄 Waiting for next scan cycle in 10 seconds...")
            time.sleep(10)
        
        except Exception as e:
            print(f"⚠️ Main loop error: {e}")
            traceback.print_exc()
            time.sleep(60)  # Wait longer if something major failed
    
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