```python
# ── BASIC ORB WITH FVG STRATEGY TRADING SCRIPT (IST TIMEZONE) ─────────────────────────────────────
# This script implements the basic Opening Range Breakout (ORB) trading strategy with Fair Value Gap
# (FVG) confirmation as described in YouTube video https://www.youtube.com/watch?v=JeMafv2c16o&t=435s
# (before 4:53 minute mark), adapted for Indian Stock Market (IST timezone) for direct production.
# ── STANDARD LIBS ────────────────────────────────────────────────────────────────────────────────
import time
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Tuple, Optional, Deque
from collections import defaultdict, deque
# ── THIRD-PARTY ──────────────────────────────────────────────────────────────────────────────────
import pytz
import pandas as pd
# ── PROJECT MODULES (already present in your repo) ───────────────────────────────────────────────
import config                  # user-supplied credentials & parameters
import dhan_api as dh          # DHAN SDK wrapper you provided
# ── LOGGING ───────────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="basic_orb_fvg.log",
    level=logging.DEBUG,        # ← promote to DEBUG for full trace
    format="%(asctime)s %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)
# ── ADD CONSOLE OUTPUT ────────────────────────────────────────────────────────────────────────────
if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
    console = logging.StreamHandler()                    # stdout
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
    log.addHandler(console)
# ── DHAN REST-RATE LIMITER PATCH (fixed, non-recursive) ──────────────────────────────────────────
"""
DHAN HQ API Specs:
  · OHLC   → 10 requests / second
  · Quotes →  1 request / second
We rate-limit the two busiest SDK calls **without** causing recursion.
"""
class _RateGate:
    def __init__(self, min_interval_sec: float):
        self.min_interval = min_interval_sec
        self._last_ts = 0.0
        self._lock = Lock()
    def wait(self):
        with self._lock:
            delta = time.perf_counter() - self._last_ts
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last_ts = time.perf_counter()
# Gates: **one call per second** is the safest universal throttle
_OHLC_GATE  = _RateGate(1.05)   #  ≈ 0.95 req/s
_QUOTE_GATE = _RateGate(1.05)   #  ≈ 0.95 req/s
# Preserve ORIGINAL SDK functions **before** patching to avoid recursion
_orig_get_hist  = dh.get_historical_price
_orig_get_quote = dh.get_live_price
def _rl_get_historical_price(security_id: str, *, interval="1", **kw):
    """
    Rate-limited wrapper for intraday candles.
      • Sleeps ≥ 1 s between calls (avoids DH-904).
      • Accepts only numeric intervals ("1", "5", …) as required by Dhan API.
    """
    _OHLC_GATE.wait()
    # Normalise common aliases → numeric string
    if interval in {"1m", "1min", "1minute"}:
        interval = "1"
    elif isinstance(interval, int):
        interval = str(interval)
    return _orig_get_hist(security_id, interval=interval, **kw)
def _rl_get_live_price(symbol: str, security_id: str):
    """Rate-limited wrapper for quote feed (≤ 1 req/s)."""
    _QUOTE_GATE.wait()
    return _orig_get_quote(symbol, security_id)
# Monkey-patch the SDK so the rest of the code stays unchanged
dh.get_historical_price = _rl_get_historical_price
dh.get_live_price       = _rl_get_live_price
# ── CONSTANTS (override in config.py if needed) ───────────────────────────────────────────────────
IST               = pytz.timezone("Asia/Kolkata")
SESSION_START     = (9, 15)           # hh, mm
ORB_WINDOW_MIN    = 5                 # Opening Range is 5 minutes
TRADE_WINDOW_END  = (15, 20)          # Stop taking new trades before EOD
FLAT_ALL_TIME     = (15, 25)          # Flat all positions by this time
RISK_PCT          = 1.0               # Risk 1% of capital per trade
RR_RATIO          = 2.0               # 2:1 reward:risk for basic ORB
MAX_TRADES        = 15                # Max trades per day
MAX_ERRORS        = 5                 # Max errors before shutdown
POLL_SEC          = 55                # Poll interval in seconds
FVG_CANDLE_COUNT  = 3                 # Number of candles to track for FVG detection
# ── DATA STRUCTURES ──────────────────────────────────────────────────────────────────────────────
class ORBState:
    """
    Keeps ORB levels and trading state flags for one symbol with Fair Value Gap tracking.
    """
    __slots__ = (
        "high",
        "low",
        "locked",
        "breakout_side",
        "breakout_price",
        "breakout_timestamp",
        "fvg_high",
        "fvg_low",
        "fvg_confirmed",
        "fvg_fill_timestamp",
        "entry_taken",
        "orb_complete",
        "candle_history",  # Track recent candles for FVG detection
    )
    def __init__(self):
        # ORB boundaries
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.locked: bool = False
        # Breakout tracking
        self.breakout_side: Optional[str] = None  # "LONG" or "SHORT"
        self.breakout_price: Optional[float] = None
        self.breakout_timestamp: Optional[str] = None
        # FVG tracking
        self.fvg_high: Optional[float] = None
        self.fvg_low: Optional[float] = None
        self.fvg_confirmed: bool = False
        self.fvg_fill_timestamp: Optional[str] = None
        # Trading state
        self.entry_taken: bool = False
        self.orb_complete: bool = False
        # Candle history for FVG detection
        self.candle_history: Deque[Dict] = deque(maxlen=FVG_CANDLE_COUNT)
class Trade:
    __slots__ = ("order_id", "side", "entry", "stop", "target", "qty", "symbol", "sid")
    def __init__(self, **kw):  # pylint: disable=all
        for k, v in kw.items():
            setattr(self, k, v)
# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────────────────────────
def now_ist() -> datetime:
    return datetime.now(IST)
def mins_since(hm: Tuple[int, int]) -> int:
    h, m = hm
    return h * 60 + m
def in_window(start_hm, end_hm, cur: Optional[datetime] = None) -> bool:
    cur = cur or now_ist()
    cur_min = cur.hour * 60 + cur.minute
    return mins_since(start_hm) <= cur_min < mins_since(end_hm)
def load_watchlist(
    path: str = r"D:\Downloads\Dhanbot\dhan_autotrader\dynamic_stock_list.csv"
) -> List[Tuple[str, str]]:
    """
    Loads the dynamic_stock_list CSV and returns a list of
    (SYMBOL, SECURITY_ID) tuples.
    Fails loudly if the file is missing, empty, or lacks the required columns.
    """
    # ── 0 · Load CSV ────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"❌ Watch-list file not found at {path!r}. "
            "Please ensure the CSV is present."
        ) from e
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"❌ Watch-list CSV at {path!r} is empty.") from e
    # Normalise column names for robust matching
    col_map = {c.lower().strip(): c for c in df.columns}
    sym_aliases = {"symbol", "trading_symbol", "sem_trading_symbol"}
    sid_aliases = {"securityid", "security_id", "sem_smst_security_id"}
    sym_col = next((col_map[a] for a in sym_aliases if a in col_map), None)
    sid_col = next((col_map[a] for a in sid_aliases if a in col_map), None)
    if sym_col is None or sid_col is None:
        raise ValueError(
            "❌ Watch-list must contain columns for Symbol and SecurityID.\n"
            f"Accepted aliases → Symbol: {sorted(sym_aliases)} | "
            f"SecurityID: {sorted(sid_aliases)}\n"
            f"Columns found    → {list(df.columns)}"
        )
    watch = [
        (str(row[sym_col]).strip().upper(),
         str(row[sid_col]).strip())
        for _, row in df.iterrows()
        if pd.notna(row[sym_col]) and pd.notna(row[sid_col])
    ]
    if not watch:
        raise ValueError(
            "❌ Watch-list is empty after parsing valid Symbol / SecurityID rows."
        )
    return watch
def position_size(entry: float, stop: float, equity: float) -> int:
    risk_cash = equity * (RISK_PCT / 100)
    risk_per_sh = abs(entry - stop)
    if risk_per_sh < 0.01:  # Minimum price difference threshold
        log.error(f"❌ Invalid risk calculation: entry={entry} stop={stop}")
        return 0
    return max(int(risk_cash // risk_per_sh), 0)
def detect_fvg(candles: Deque[Dict]) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Detects Fair Value Gap (FVG) in the most recent candles.
    FVG exists when:
      - For LONG: low of current candle > high of candle two periods back
      - For SHORT: high of current candle < low of candle two periods back
    
    Returns:
      (is_fvg, fvg_high, fvg_low)
    """
    if len(candles) < 3:
        return False, None, None
    
    # Get the three most recent candles
    c0 = candles[-3]
    c1 = candles[-2]
    c2 = candles[-1]
    
    # LONG FVG: gap from c0.high up to c2.low
    if c2["low"] > c0["high"]:
        fvg_high = c2["low"]
        fvg_low  = c0["high"]
        return True, fvg_high, fvg_low
    
    # SHORT FVG: gap from c2.high down to c0.low
    if c2["high"] < c0["low"]:
        fvg_high = c0["low"]
        fvg_low  = c2["high"]
        return True, fvg_high, fvg_low
    
    return False, None, None

# ── CORE ENGINE ──────────────────────────────────────────────────────────────────────────────────
class BasicORBFVGEngine:
    def __init__(self):
        self.watch = load_watchlist()
        self.state: Dict[str, ORBState] = {s: ORBState() for s, _ in self.watch}
        self.trades: Dict[str, Trade] = {}
        self.error_count = 0
        # Capital from config (preferred) or broker helper
        try:
            self.equity_base = float(config.capital)
        except (AttributeError, ValueError, TypeError):
            self.equity_base = dh.get_current_capital()
    
    # ── ORB CAPTURE ───────────────────────────────────────────────────────────────────────────────
    def capture_orb(self):
        """
        Fills ORB range between 09:15-09:20 using 1-minute candles stacked.
        """
        for sym, sid in self.watch:
            # Fetch the five 1-minute bars spanning 09:15–09:20
            bars = dh.get_historical_price(sid, interval="1", limit=5)
            if not bars:
                log.error(f"❌ No ORB data for {sym} — could not fetch five 1-min candles")
                continue
            highs = [b["high"] for b in bars]
            lows  = [b["low"]  for b in bars]
            st = self.state[sym]
            # Update ORB bounds based on full set of 1-min highs/lows
            st.high = max(st.high or max(highs), max(highs))
            st.low  = min(st.low  or min(lows),  min(lows))
            log.debug(
                f"{sym}: ORB-capture highs={highs} lows={lows} "
                f"→ stored ORB low={st.low} high={st.high}"
            )
    
    
    # ── MAIN STATE MACHINE PER SYMBOL ─────────────────────────────────────────────────────────────
    def process_symbol(self, sym: str, sid: str):
        st = self.state[sym]
        if not st.locked:
            log.debug(f"{sym}: ORB not locked yet — skipping.")
            return
            
        # Get the latest 1-minute candle
        candles = dh.get_historical_price(sid, interval="1", limit=1)
        if not candles:
            log.debug(f"{sym}: no fresh 1-min candle — skipping.")
            return
            
        bar = candles[-1]
        close, high, low = bar["close"], bar["high"], bar["low"]
        
        # Add to candle history for FVG detection
        st.candle_history.append({
            "timestamp": bar.get("timestamp", ""),
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"]
        })
        
        # Format timestamp for logging
        bar_ts = bar.get("timestamp") or bar.get("time") or bar.get("datetime")
        if isinstance(bar_ts, (int, float)):
            ts = datetime.fromtimestamp(bar_ts).strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts = str(bar_ts)
        
        log.debug(
            f"{sym}: tick close={close} h={high} l={low} ts={ts} "
            f"| ORB h={st.high} l={st.low} | breakout={st.breakout_side} "
            f"fvg_confirmed={st.fvg_confirmed} entry_taken={st.entry_taken}"
        )
        
        # Check if we've already taken an entry for this symbol today
        if st.entry_taken:
            log.debug(f"{sym}: entry already taken today — skipping.")
            return
            
        # ------------------------------------------------------------------ #
        # STEP 1 – detect the initial breakout from ORB range
        # ------------------------------------------------------------------ #
        if st.breakout_side is None:
            # LONG side breakout
            if high > st.high:
                st.breakout_side      = "LONG"
                st.breakout_price     = high
                st.breakout_timestamp = ts
                log.info(f"✅ {sym}: breakout to the upside detected @ {high} at {ts}")
                return
            # SHORT side breakout
            if low < st.low:
                st.breakout_side      = "SHORT"
                st.breakout_price     = low
                st.breakout_timestamp = ts
                log.info(f"✅ {sym}: breakout to the downside detected @ {low} at {ts}")
                return
            # No breakout yet, continue monitoring
            return
            
        # ------------------------------------------------------------------ #
        # STEP 2 – detect Fair Value Gap (FVG) formation after breakout
        # ------------------------------------------------------------------ #
        if not st.fvg_confirmed and len(st.candle_history) >= 3:
            is_fvg, fvg_high, fvg_low = detect_fvg(st.candle_history)
            
            if is_fvg:
                st.fvg_confirmed = True
                st.fvg_high = fvg_high
                st.fvg_low = fvg_low
                log.info(
                    f"✅ {sym}: Fair Value Gap confirmed — "
                    f"gap from {fvg_low:.2f} to {fvg_high:.2f}"
                )
                return
                
        # ------------------------------------------------------------------ #
        # STEP 3 – detect FVG fill and enter trade
        # ------------------------------------------------------------------ #
        if st.fvg_confirmed and not st.entry_taken:
            # For LONG: price must close within the FVG area (between fvg_low and fvg_high)
            if st.breakout_side == "LONG" and st.fvg_low <= close <= st.fvg_high:
                log.info(
                    f"✅✅ {sym}: FVG filled for LONG trade @ {close}. "
                    f"Entering trade with FVG {st.fvg_low:.2f}-{st.fvg_high:.2f}"
                )
                # Use the FVG low as the stop for LONG
                self.open_trade(sym, sid, "LONG", close, st.fvg_low)
                st.entry_taken = True
                ...
            # For SHORT: price must close within the FVG area (between fvg_low and fvg_high)
            elif st.breakout_side == "SHORT" and st.fvg_low <= close <= st.fvg_high:
                log.info(
                    f"✅✅ {sym}: FVG filled for SHORT trade @ {close}. "
                    f"Entering trade with FVG {st.fvg_low:.2f}-{st.fvg_high:.2f}"
                )
                # Use the FVG high as the stop for SHORT
                self.open_trade(sym, sid, "SHORT", close, st.fvg_high)
                st.entry_taken = True
                st.orb_complete = True
                st.fvg_fill_timestamp = ts
    
    # ── OPEN TRADE ────────────────────────────────────────────────────────────────────────────────
    def open_trade(self, sym: str, sid: str, side: str, entry: float, stop: float):
        """
        Places the bracket order for the basic ORB strategy with FVG confirmation.
        """
        # Time validation - don't enter if too close to flat time
        now = now_ist()
        mins_to_flat = mins_since(FLAT_ALL_TIME) - (now.hour * 60 + now.minute)
        if mins_to_flat < 15:  # Need at least 15 minutes to potentially reach target
            log.info(f"{sym}: too close to flat time ({mins_to_flat} mins left); skipping entry.")
            return
            
        # Position sizing
        qty = position_size(entry, stop, self.equity_base)
        if qty == 0:
            log.info(f"{sym}: qty 0 — risk too small.")
            return
            
        # Calculate target based on RR_RATIO
        if side == "LONG":
            target = entry + RR_RATIO * (entry - stop)
        else:  # SHORT
            target = entry - RR_RATIO * (stop - entry)
            
        # Place order
        txn = "BUY" if side == "LONG" else "SELL"
        code, resp = dh.place_order(
            sid,
            qty,
            transaction_type=txn,
            super_order=True,         # Use bracket orders
            take_profit=target,
            stop_loss=stop,
        )
        
        if code != 200:
            log.error(f"{sym}: order rejected — {resp}")
            return
            
        oid = resp["data"]["order_id"]
        self.trades[oid] = Trade(
            order_id=oid,
            side=side,
            entry=entry,
            stop=stop,
            target=target,
            qty=qty,
            symbol=sym,
            sid=sid,
        )
        log.info(
            f"ORB {side} {sym} qty={qty} entry={entry} "
            f"SL={stop} TP={target} (RR={RR_RATIO}:1) | FVG Confirmed"
        )
    
    # ── MANAGE OPEN TRADES ────────────────────────────────────────────────────────────────────────
    def manage_trades(self):
        for oid, tr in list(self.trades.items()):
            ltp = dh.get_live_price(tr.symbol, tr.sid)
            if ltp is None:
                continue
                
            log.debug(
                f"{tr.symbol}: LTP={ltp} "
                f"entry={tr.entry} SL={tr.stop} TP={tr.target} side={tr.side}"
            )
            
            # Stop-loss
            if (tr.side == "LONG" and ltp <= tr.stop) or \
               (tr.side == "SHORT" and ltp >= tr.stop):
                dh.place_order(tr.sid, tr.qty,
                               transaction_type="SELL" if tr.side == "LONG" else "BUY")
                log.info(f"SL HIT {tr.symbol} @ {ltp}")
                self.trades.pop(oid, None)
                continue
                
            # Target
            if (tr.side == "LONG" and ltp >= tr.target) or \
               (tr.side == "SHORT" and ltp <= tr.target):
                dh.place_order(tr.sid, tr.qty,
                               transaction_type="SELL" if tr.side == "LONG" else "BUY")
                log.info(f"TP HIT {tr.symbol} @ {ltp}")
                self.trades.pop(oid, None)
    
    # ── RESET SYMBOL DAILY ────────────────────────────────────────────────────────────────────────
    def reset_symbol(self, sym: str):
        self.state[sym] = ORBState()
    
    # ── FLAT ALL ──────────────────────────────────────────────────────────────────────────────────
    def flat_all(self):
        for oid, tr in list(self.trades.items()):
            dh.place_order(tr.sid, tr.qty,
                           transaction_type="SELL" if tr.side == "LONG" else "BUY")
            log.info(f"EOD FLAT {tr.symbol}")
            self.trades.pop(oid, None)
    
    # ── HELPER : BACK-FILL ORB IF SCRIPT STARTED LATE ────────────────────────────────────────────
    def initialise_orb_if_missed(self):
        """
        If the engine starts *after* the ORB window (post-09:20 IST),
        back-fill the ORB high/low for every watched symbol using historical
        5-minute candles from 09:15 to 09:20 and immediately lock the levels.
        """
        now = now_ist()
        orb_end = mins_since(SESSION_START) + ORB_WINDOW_MIN
        if now.hour * 60 + now.minute < orb_end:
            return  # We are still inside the ORB window, nothing to do.
            
        if all(st.locked for st in self.state.values()):
            return  # Already initialised.
            
        for sym, sid in self.watch:
            # Historical candles for 09:15–09:20 IST
            trade_date = now.strftime("%Y-%m-%d")
            from_dt = f"{trade_date} {SESSION_START[0]:02d}:{SESSION_START[1]:02d}:00"
            to_dt   = (datetime.strptime(from_dt, "%Y-%m-%d %H:%M:%S")
                       + timedelta(minutes=ORB_WINDOW_MIN)).strftime("%Y-%m-%d %H:%M:%S")
            candles = dh.get_historical_price(
                sid,
                interval="5",
                from_date=from_dt,
                to_date=to_dt
            )
            if not candles:
                log.error(f"❌ Historical data unavailable for {sym} during ORB back-fill")
                continue
                
            highs = [c["high"] for c in candles]
            lows  = [c["low"]  for c in candles]
            st = self.state[sym]
            st.high = max(highs)
            st.low  = min(lows)
            st.locked = True
        log.info("ORB back-filled from historical data – late start handled.")
    
    # ── MAIN LOOP ─────────────────────────────────────────────────────────────────────────────────
    def run(self):
        log.info("Basic ORB with FVG engine started.")
        loop = 0                     # 🔢 heartbeat counter
        current_session_date = now_ist().date()     # ⏰ track the trading day
        
        while True:
            loop += 1
            try:
                now = now_ist()
                log.debug(
                    f"🔄 Loop #{loop} started at {now.strftime('%H:%M:%S')} "
                    f"(watch-size={len(self.watch)})"
                )
                
                # ── Reset all pattern / trade flags at the dawn of a new trading day ──
                if now.date() != current_session_date and now.hour >= 9:
                    log.info("🔄 New trading day detected — resetting ORB states.")
                    current_session_date = now.date()
                    self.state = {s: ORBState() for s, _ in self.watch}
                    self.trades.clear()
                
                # Handle late starts (after 09:20) by back-filling ORB levels
                self.initialise_orb_if_missed()
                
                # ORB capture window (live filling when script starts on time)
                if in_window(SESSION_START,
                             (SESSION_START[0], SESSION_START[1] + ORB_WINDOW_MIN),
                             now):
                    self.capture_orb()
                    time.sleep(POLL_SEC)
                    continue
                
                # Lock ORB levels once window closes (for on-time starts)
                current_minutes = now.hour * 60 + now.minute
                orb_end_minutes = SESSION_START[0] * 60 + SESSION_START[1] + ORB_WINDOW_MIN
                if current_minutes >= orb_end_minutes and \
                   not all(s.locked for s in self.state.values()):
                    for s in self.state.values():
                        s.locked = True
                    log.info("ORB locked for all symbols (live capture).")
                
                # Run detailed stock-wise iteration once ORB levels are locked (handles late starts)
                if all(st.locked for st in self.state.values()):
                    # ── Detailed stock-wise iteration with per-symbol debug ──
                    original_total = len(self.watch)
                    for idx, (sym, sid) in enumerate(self.watch, 1):
                        if len(self.trades) >= MAX_TRADES:
                            break
                            
                        st = self.state[sym]
                        def fmt(x):
                            return f"{x:.2f}" if isinstance(x, (int, float)) else "NA"
                        
                        # ── Permanently skip for today once entry is taken ──
                        if st.entry_taken:
                            log.info(
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"Entry already taken today – ORB {fmt(st.low)}-{fmt(st.high)}. Skipping."
                            )
                            continue
                        
                        # ── Build a clear status headline for *every* symbol ──
                        if not st.locked:
                            headline = (
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"ORB window active – waiting for range definition."
                            )
                        elif st.entry_taken:
                            headline = (
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"Trade executed – ORB {fmt(st.low)}-{fmt(st.high)}. Skipping."
                            )
                        elif st.breakout_side is None:
                            headline = (
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"ORB defined {fmt(st.low)}-{fmt(st.high)} – "
                                f"waiting for breakout."
                            )
                        elif not st.fvg_confirmed:
                            headline = (
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"Breakout {st.breakout_side} detected – "
                                f"waiting for FVG confirmation."
                            )
                        else:
                            headline = (
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"FVG confirmed – waiting for fill "
                                f"({fmt(st.fvg_low)}-{fmt(st.fvg_high)})"
                            )
                        log.info(headline)
                        
                        # ── Detailed tick-level processing ──
                        self.process_symbol(sym, sid)
                
                self.manage_trades()
                
                # Forced flat near close
                if in_window(FLAT_ALL_TIME, (15, 30), now):
                    self.flat_all()
                    log.info("Session finished; shutting down.")
                    break
                    
                time.sleep(POLL_SEC)
            except KeyboardInterrupt:
                log.warning("Manual interrupt — flattening positions.")
                self.flat_all()
                break
            except Exception as e:
                self.error_count += 1
                log.exception(f"Runtime error: {e}")
                if self.error_count >= MAX_ERRORS:
                    log.critical("MAX_ERRORS reached — shutting down.")
                    self.flat_all()
                    break

# ── ENTRY POINT ─────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BasicORBFVGEngine().run()
```