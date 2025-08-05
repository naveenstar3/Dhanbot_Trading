# ── STANDARD LIBS ─────────────────────────────────────────────────────────────
import time
import logging
from datetime import datetime
from threading import Lock
from typing import Dict, List, Tuple, Optional

# ── THIRD-PARTY ───────────────────────────────────────────────────────────────
import pytz
import pandas as pd

# ── PROJECT MODULES (already present in your repo) ───────────────────────────
import config                  # user-supplied credentials & parameters
import dhan_api as dh          # DHAN SDK wrapper you provided

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="double_break.log",
    level=logging.DEBUG,          # ← promote to DEBUG for full trace
    format="%(asctime)s %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── ADD CONSOLE OUTPUT ────────────────────────────────────────────────────
if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
    console = logging.StreamHandler()                    # stdout
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
    log.addHandler(console)

# ── DHAN REST-RATE LIMITER PATCH (fixed, non-recursive) ──────────────────────
"""
DHAN HQ API Specs (per image):
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

# ── CONSTANTS (override in config.py if needed) ──────────────────────────────
IST               = pytz.timezone("Asia/Kolkata")
SESSION_START     = (9, 15)           # hh, mm
ORB_WINDOW_MIN    = 5
SIGNAL_WINDOW_MIN = 90               # 09:20→10:50
FLAT_ALL_TIME     = (15, 25)
RISK_PCT          = 1.0
RR_RATIO          = 3.0              # 3 : 1 for Double Break
MAX_TRADES        = 10
MAX_ERRORS        = 5
POLL_SEC          = 55

# ── DATA STRUCTURES ──────────────────────────────────────────────────────────
class ORBState:
    """
    Keeps ORB levels and Double-Break state flags for one symbol.
    """
    __slots__ = (
        "high",
        "low",
        "locked",
        "first_break_side",
        "first_break_candle_high",
        "first_break_candle_low",
        "retracement_extreme",
        "pulled_back",
        "entry_taken",
        "double_break_complete",          # NEW → permanently blacklist symbol
    )

    def __init__(self):
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.locked: bool = False

        # Double-Break bookkeeping
        self.first_break_side: Optional[str] = None       # "LONG" or "SHORT"
        self.first_break_candle_high: Optional[float] = None
        self.first_break_candle_low: Optional[float] = None
        self.retracement_extreme: Optional[float] = None  # swing low (long) / swing high (short)
        self.pulled_back: bool = False                    # has price closed back in range?
        self.entry_taken: bool = False                    # one trade per day


class Trade:
    __slots__ = ("order_id", "side", "entry", "stop", "target", "qty", "symbol", "sid")
    def __init__(self, **kw):  # pylint: disable=all
        for k, v in kw.items():
            setattr(self, k, v)

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
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


# ── CORE ENGINE ──────────────────────────────────────────────────────────────
class DoubleBreakEngine:
    def __init__(self):
        self.watch = load_watchlist()
        self.state: Dict[str, ORBState] = {s: ORBState() for s, _ in self.watch}
        self.trades: Dict[str, Trade] = {}
        self.error_count = 0
        self.trade_placed = False  

        # Capital from config (preferred) or broker helper
        try:
            self.equity_base = float(config.capital)
        except (AttributeError, ValueError, TypeError):
            self.equity_base = dh.get_current_capital()

    # ── ORB CAPTURE ───────────────────────────────────────────────────────────
    def capture_orb(self):
        """
        Fills ORB range between 09:15-09:20.
        Adds DEBUG lines so we know exactly which candle was processed.
        """
        for sym, sid in self.watch:
            bars = dh.get_historical_price(sid, interval="1", limit=1)
            if not bars:
                log.error(f"❌ No historical data for {sym} ({sid}) during ORB capture")
                continue
    
            bar = bars[-1]
            st = self.state[sym]
            prev_high, prev_low = st.high, st.low
            st.high = max(st.high or bar["high"], bar["high"])
            st.low  = min(st.low  or bar["low"],  bar["low"])
    
            log.debug(
                f"{sym}: ORB-capture candle h={bar['high']} l={bar['low']} "
                f"→ stored high {prev_high}->{st.high} | low {prev_low}->{st.low}"
            )
    

    # ── MAIN STATE MACHINE PER SYMBOL ─────────────────────────────────────────
    def process_symbol(self, sym: str, sid: str):
        st = self.state[sym]
    
        if not st.locked:
            log.debug(f"{sym}: ORB not locked yet — skipping.")
            return
    
        candles = dh.get_historical_price(sid, interval="1", limit=1)
        if not candles:
            log.debug(f"{sym}: no fresh 1-min candle — skipping.")
            return
    
        bar = candles[-1]
        close, high, low = bar["close"], bar["high"], bar["low"]
        log.debug(
            f"{sym}: tick close={close} h={high} l={low} "
            f"| ORB h={st.high} l={st.low} | first_break={st.first_break_side} "
            f"pulled_back={st.pulled_back} entry_taken={st.entry_taken}"
        )
    
        orb_range = st.high - st.low

        # ------------------------------------------------------------------ #
        # STEP 1 – detect the FIRST break, store its candle extremes
        # ------------------------------------------------------------------ #
        if st.first_break_side is None:
            # LONG side first break
            if close > st.high:
                st.first_break_side = "LONG"
                st.first_break_candle_low = low
                st.first_break_candle_high = high
                st.retracement_extreme = low  # start with its low
                log.info(f"{sym}: first LONG break recorded @ {close}")
                return
            # SHORT side first break
            if close < st.low:
                st.first_break_side = "SHORT"
                st.first_break_candle_low = low
                st.first_break_candle_high = high
                st.retracement_extreme = high  # start with its high
                log.info(f"{sym}: first SHORT break recorded @ {close}")
                return
            # nothing else to do yet
            return

        # ------------------------------------------------------------------ #
        # STEP 2 – track retracement inside ORB range
        # ------------------------------------------------------------------ #
        if not st.pulled_back:
            if st.first_break_side == "LONG":
                # Has close come back INTO range?
                if close < st.high:
                    st.pulled_back = True
                    st.retracement_extreme = low
                    log.info(f"{sym}: pull-back detected (LONG).")
                return  # wait for pull-back
            else:  # SHORT
                if close > st.low:
                    st.pulled_back = True
                    st.retracement_extreme = high
                    log.info(f"{sym}: pull-back detected (SHORT).")
                return

        # update retracement extremes while inside the range
        if st.pulled_back and not st.entry_taken:
            if st.first_break_side == "LONG":
                # update swing low
                if st.retracement_extreme is None:
                    st.retracement_extreme = low
                else:
                    st.retracement_extreme = min(st.retracement_extreme, low)
                # abort if pull-back travels > one ORB range (close below ORB-low)
                if close < st.low:
                    log.info(f"{sym}: pull-back too deep; resetting LONG setup.")
                    self.reset_symbol(sym)
            else:  # SHORT
                if st.retracement_extreme is None:
                    st.retracement_extreme = high
                else:
                    st.retracement_extreme = max(st.retracement_extreme, high)
                if close > st.high:
                    log.info(f"{sym}: pull-back too deep; resetting SHORT setup.")
                    self.reset_symbol(sym)

        # ------------------------------------------------------------------ #
        # STEP 3 – detect SECOND break and enter trade
        # ------------------------------------------------------------------ #
        if st.pulled_back and not st.double_break_complete:
            if st.first_break_side == "LONG" and close > st.high:
                self.open_trade(sym, sid, "LONG", close, st.retracement_extreme)
                st.entry_taken = True          # may remain True/False depending on order result
                st.double_break_complete = True
            elif st.first_break_side == "SHORT" and close < st.low:
                self.open_trade(sym, sid, "SHORT", close, st.retracement_extreme)
                st.entry_taken = True
                st.double_break_complete = True

    # ── HELPER · historical Double-Break replay ──────────────────────────────
    def estimate_minutes_needed(self, sym: str, sid: str,
                                lookback_days: int = 60,
                                min_samples: int = 3) -> float:
        """
        Scan backwards day-by-day (max `lookback_days`) until `min_samples`
        Double-Breaks are replayed.  Returns median minutes-to-target.
        Caches result in `_hist_cache` so the heavy replay runs only once
        per symbol per day.
        """
        if not hasattr(self, "_hist_cache"):
            self._hist_cache = {}

        if sym in self._hist_cache:
            return self._hist_cache[sym]

        from datetime import timedelta, date as _date
        samples, days_checked = [], 0
        cur_d = now_ist().date() - timedelta(days=1)   # start with prev day

        while len(samples) < min_samples and days_checked < lookback_days:
            if cur_d.weekday() >= 5:                   # skip weekends
                cur_d -= timedelta(days=1); days_checked += 1; continue

            from_dt = f"{cur_d} 09:15:00"
            to_dt   = f"{cur_d} 15:25:00"
            candles = dh.get_historical_price(sid, interval="1",
                                              from_date=from_dt, to_date=to_dt)
            days_checked += 1
            cur_d -= timedelta(days=1)
            if len(candles) < 30:                      # not enough data
                continue

            # --- 1 · ORB levels
            orb_high = max(c["high"] for c in candles[:5])
            orb_low  = min(c["low"]  for c in candles[:5])

            first_side = None
            pulled_back = False
            entry_idx   = None
            entry_price = None
            stop_price  = None

            for idx, c in enumerate(candles[5:], start=5):
                close = c["close"]

                # detect first break
                if first_side is None:
                    if close > orb_high:
                        first_side, retrace_ext = "LONG", c["low"]
                    elif close < orb_low:
                        first_side, retrace_ext = "SHORT", c["high"]
                    continue

                # track pull-back
                if not pulled_back:
                    if first_side == "LONG" and close < orb_high:
                        pulled_back = True
                    elif first_side == "SHORT" and close > orb_low:
                        pulled_back = True
                    continue

                # detect second break → entry
                if pulled_back and entry_idx is None:
                    if first_side == "LONG" and close > orb_high:
                        entry_idx, entry_price = idx, close
                        stop_price = retrace_ext
                    elif first_side == "SHORT" and close < orb_low:
                        entry_idx, entry_price = idx, close
                        stop_price = retrace_ext
                    continue

                # after entry: look for TP
                if entry_idx is not None:
                    rr = RR_RATIO * abs(entry_price - stop_price)
                    target = entry_price + rr if first_side == "LONG" \
                             else entry_price - rr
                    hit = (first_side == "LONG" and close >= target) or \
                          (first_side == "SHORT" and close <= target)
                    if hit:
                        minutes = (idx - entry_idx)
                        samples.append(minutes)
                        break   # done with this day

        # Median or fallback 60 min
        median = sorted(samples)[len(samples)//2] if samples else 60
        self._hist_cache[sym] = median
        return median

    # ── OPEN TRADE ────────────────────────────────────────────────────────────
    def open_trade(self, sym: str, sid: str, side: str, entry: float, stop: float):
        """
        Places the Super(beta) bracket order only if there is **sufficient time
        left** before the 15:25 EOD-flat to reach the profit target.
        """
        # ── 1 · Time-remaining gate (historic + intraday blend) ───────────────
        now       = now_ist()
        mins_left = mins_since(FLAT_ALL_TIME) - (now.hour * 60 + now.minute)

        # (a) Percent move required
        target_pct = abs(entry - stop) * RR_RATIO / entry * 100

        # (b) Intraday speed from last 24×5-min candles
        candles_5m = dh.get_historical_price(sid, interval="5", limit=24)
        avg_5m_pct = (sum(abs(c["high"] - c["low"]) / c["close"] * 100
                          for c in candles_5m) / len(candles_5m)) if candles_5m else 0.2
        intraday_needed = (target_pct / (avg_5m_pct / 5)) if avg_5m_pct else 60

        # (c) Historical average from replay
        hist_needed = self.estimate_minutes_needed(sym, sid)

        minutes_needed = max(hist_needed, intraday_needed)
        buffer = 5   # safety margin

        if mins_left < minutes_needed + buffer:
            log.info(
                f"{sym}: needs ≈{minutes_needed:.1f} min but only {mins_left} min left; "
                "skipping new entry."
            )
            return
    
        # ── 2 · Position sizing & capital checks (unchanged) ────────────────
        locked = sum(t.entry * t.qty for t in self.trades.values() if t.side == "LONG")
        available_cash = max(0, self.equity_base - locked)
        qty = position_size(entry, stop, self.equity_base)
    
        if side == "LONG" and qty * entry > available_cash:
            log.warning(f"{sym}: insufficient capital for LONG ({qty} @ {entry}).")
            return
        if qty == 0:
            log.info(f"{sym}: qty 0 — risk too small.")
            return
    
        # ── 3 · Send Super(beta) order (unchanged) ───────────────────────────
        target = entry + RR_RATIO * (entry - stop) if side == "LONG" \
                else entry - RR_RATIO * (stop - entry)
        txn = "BUY" if side == "LONG" else "SELL"
    
        code, resp = dh.place_order(
            sid,
            qty,
            transaction_type=txn,
            super_order=True,         # ← activates “Super (Beta)” bracket order
            take_profit=target,
            stop_loss=stop,
        )
        
        # ── Flag the session as complete on the first successful order ────────────
        if code == 200 and resp.get("status") == "success":
            self.trade_placed = True
            
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
            f"DOUBLE-BREAK {side} {sym} qty={qty} entry={entry} "
            f"SL={stop} TP={target}"
        )

    # ── MANAGE OPEN TRADES ───────────────────────────────────────────────────
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

    # ── RESET SYMBOL DAILY ───────────────────────────────────────────────────
    def reset_symbol(self, sym: str):
        self.state[sym] = ORBState()

    # ── FLAT ALL ──────────────────────────────────────────────────────────────
    def flat_all(self):
        for oid, tr in list(self.trades.items()):
            dh.place_order(tr.sid, tr.qty,
                           transaction_type="SELL" if tr.side == "LONG" else "BUY")
            log.info(f"EOD FLAT {tr.symbol}")
            self.trades.pop(oid, None)

    # ── HELPER : BACK-FILL ORB IF SCRIPT STARTED LATE ───────────────────────
    def initialise_orb_if_missed(self):
        """
        If the engine starts *after* the ORB window (post-09:20 IST),
        back-fill the ORB high/low for every watched symbol using historical
        1-minute candles from 09:15 to 09:20 and immediately lock the levels.
        """
        now = now_ist()
        orb_end = mins_since(SESSION_START) + ORB_WINDOW_MIN
        if now.hour * 60 + now.minute < orb_end:
            return  # We are still inside the ORB window, nothing to do.

        if all(st.locked for st in self.state.values()):
            return  # Already initialised.

        from datetime import timedelta
        for sym, sid in self.watch:
            # Historical candles for 09:15–09:20 IST
            trade_date = now.strftime("%Y-%m-%d")
            from_dt = f"{trade_date} {SESSION_START[0]:02d}:{SESSION_START[1]:02d}:00"
            to_dt   = (datetime.strptime(from_dt, "%Y-%m-%d %H:%M:%S")
                       + timedelta(minutes=ORB_WINDOW_MIN)).strftime("%Y-%m-%d %H:%M:%S")

            candles = dh.get_historical_price(
                sid,
                interval="1",
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

    # ── MAIN LOOP ────────────────────────────────────────────────────────────
    def run(self):
        log.info("Double-Break engine started.")
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
                    self.state = defaultdict(ORBState)   # fresh flags for every symbol
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

                # Double-Break logic runs until 10:50 IST
                if in_window((SESSION_START[0], SESSION_START[1] + ORB_WINDOW_MIN),
                             (SESSION_START[0], SESSION_START[1] + ORB_WINDOW_MIN + SIGNAL_WINDOW_MIN),
                             now):
                    # ── Detailed stock-wise iteration with per-symbol debug ──
                    original_total = len(self.watch)
                    pending_symbols = [(s, i) for i, (s, _) in enumerate(self.watch, 1)
                                    if not self.state[s].double_break_complete]
                    pending_count = len(pending_symbols)
                    
                    for idx, (sym, sid) in enumerate(self.watch, 1):
                        if len(self.trades) >= MAX_TRADES:
                            break
                    
                        st = self.state[sym]
                    
                        def fmt(x):
                            return f"{x:.2f}" if isinstance(x, (int, float)) else "NA"
                    
                        # ── New: per-stock DEBUG before any skipping or headlines ──
                        if st.double_break_complete:
                            # 2/50 Checking Stock-B: Double break already formed.
                            log.debug(
                                f"{idx}/{original_total} Checking {sym}: "
                                f"Double break already formed – "
                                f"ORB {fmt(st.low)}-{fmt(st.high)}; "
                                f"first break side={st.first_break_side}"
                            )
                            # remove from future consideration automatically by not including in pending_symbols
                            continue
                    
                        # compute position among pending
                        # find its 1-based index in pending_symbols
                        pending_idx = next((i for i, (s, _) in enumerate(pending_symbols, 1) if s == sym), None)
                    
                        # 1/Out of 50 Checking Stock-A: Double Break not yet formed: Waiting…
                        log.debug(
                            f"{pending_idx}/{original_total} Checking {sym}: "
                            f"Double Break not yet formed – "
                            f"ORB {fmt(st.low)}-{fmt(st.high)}, "
                            f"expect price > {fmt(st.high)}"
                        )
                    
                        # ── 1 · Permanently skip for today once pattern is done ──
                        if st.double_break_complete:
                            log.info(
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"Double-Break complete – ORB {fmt(st.low)}-{fmt(st.high)}; "
                                f"side={st.first_break_side}. Skipping."
                            )
                            continue
                    
                        # ── 2 · Build a clear status headline for *every* symbol ──
                        if st.first_break_side is None:
                            headline = (
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"Awaiting 1st break – ORB {fmt(st.low)}-{fmt(st.high)}."
                            )
                        elif not st.pulled_back:
                            headline = (
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"1st break {st.first_break_side} seen – waiting pull-back "
                                f"inside ORB {fmt(st.low)}-{fmt(st.high)}."
                            )
                        else:
                            trg = st.high if st.first_break_side == "LONG" else st.low
                            headline = (
                                f"{idx:02d}/{original_total:02d} {sym}: "
                                f"Pull-back done – waiting 2nd break "
                                f"{'above' if st.first_break_side == 'LONG' else 'below'} "
                                f"{fmt(trg)} (ORB {fmt(st.low)}-{fmt(st.high)})."
                            )
                    
                        log.info(headline)
                    
                        # ── 3 · Detailed tick-level processing (unchanged) ──
                        self.process_symbol(sym, sid)
                    


                self.manage_trades()
                
                # Stop scanning once a trade is on
                if self.trade_placed:
                    log.info("First trade placed — halting further symbol evaluation.")
                    break

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

# ── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DoubleBreakEngine().run()
