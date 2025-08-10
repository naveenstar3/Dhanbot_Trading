# ── STANDARD LIBS ─────────────────────────────────────────────────────────────
import sys
import time
import math
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ── THIRD-PARTY ───────────────────────────────────────────────────────────────
import pytz
import pandas as pd

# ── PROJECT MODULES (must exist in your environment) ──────────────────────────
# Keep these imports identical to your existing project layout
import config                  # user-supplied credentials & parameters
import dhan_api as dh          # DHAN SDK wrapper you provided
from pathlib import Path  # ★ needed for file existence checks

# ── TIMEZONE ──────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")
ENGINE_START_TS: Optional[datetime] = None

# ── LOGGING ───────────────────────────────────────────────────────────────────
log = logging.getLogger("fvg_bounce_3r")
log.setLevel(logging.DEBUG)
if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s"))
    log.addHandler(sh)

# ── CONFIG HELPERS (do NOT change user config; just read it robustly) ─────────
def CFG(name: str, default):
    return getattr(config, name, default)

# Core timings (NSE default 09:15–09:20 for the first 5-minute window)
ORB_START_HM: Tuple[int, int] = CFG("ORB_START_HM", (9, 15))
ORB_END_HM:   Tuple[int, int] = CFG("ORB_END_HM",   (9, 20))

# Strategy behavior toggles
USE_BOUNCE_RULE: bool = CFG("USE_BOUNCE_RULE", True)    # keep True for this script
RR_RATIO: float        = CFG("RR_RATIO", 3.0)           # 3R fixed
POLL_SEC: int          = CFG("POLL_SEC", 15)            # polling cadence (seconds)
MAX_TRADES: int        = CFG("MAX_TRADES", 3)           # safety limit

# Risk & position sizing caps
CAPITAL: float         = CFG("CAPITAL", 100000.0)
RISK_PCT: float        = CFG("RISK_PCT", 0.01)          # 1% risk per trade by default
MAX_BUDGET: float      = CFG("MAX_BUDGET", CAPITAL)     # cap not to exceed this
MAX_TOTAL_RISK: Optional[float] = CFG("MAX_TOTAL_RISK", None)  # rupees committed across day

# Execution precision/safety
TICK_SIZE: float       = CFG("TICK_SIZE", 0.05)         # NSE equities default
LOT_SIZE: int          = CFG("LOT_SIZE", 1)             # global fallback for equities
LOT_SIZES: dict        = CFG("LOT_SIZES", {})           # optional map: { "SBIN":1, "2885":25, ... }
VERIFY_OCO: bool       = CFG("VERIFY_OCO", True)
ORDER_RETRY_ATTEMPTS, ORDER_RETRY_BACKOFF = CFG("ORDER_RETRY", (3, 1.2))
STRICT_BOUNCE_STOP: bool = CFG("STRICT_BOUNCE_STOP", False)

# ⬇️ New, opt-in safety toggles (default preserves current behavior)
AUTO_CANCEL_ON_OCO_FAIL: bool = CFG("AUTO_CANCEL_ON_OCO_FAIL", True)  # cancel parent if OCO verify fails
OCO_FAIL_ABORTS: bool         = CFG("OCO_FAIL_ABORTS", False)          # return non-200 on failed OCO
RECONCILE_FILLED_QTY: bool    = CFG("RECONCILE_FILLED_QTY", False)     # read filled qty for ledger/risk
FILL_RETRY_ATTEMPTS, FILL_RETRY_DELAY = CFG("FILL_RETRY", (3, 1.0))    # attempts/delay for filled-qty read
ABORT_REQUESTED: bool = False  # ⬅️ new

# ✅ NEW: stale-plan safety (opt-in; keeps current behavior by default)
USE_STALE_PLAN_GUARD: bool    = CFG("USE_STALE_PLAN_GUARD", False)      # re-confirm candle recency & LTP before placing
MAX_CANDLE_STALENESS_SEC: int = CFG("MAX_CANDLE_STALENESS_SEC", 75)     # last 1m candle must be within this age
MAX_ENTRY_SLIPPAGE_R: float   = CFG("MAX_ENTRY_SLIPPAGE_R", 0.30)       # max allowed |LTP-entry| in R units

# ✅ NEW: per-plan “consume latch” (opt-in; unchanged default)
USE_CONSUME_PLAN_LATCH: bool  = CFG("USE_CONSUME_PLAN_LATCH", False)    # don’t retry the same (symbol, plan.time)
CONSUMED_PLANS: set = set()  # {(symbol, datetime)}

def lot_size_for(symbol: str, security_id: str) -> int:
    """
    Resolve correct lot size for this instrument.
    Priority:
      1) config.LOT_SIZES by security_id (string) or SYMBOL (upper)
      2) dhan_master.csv LOT_SIZE column (if present)
      3) global LOT_SIZE fallback (>=1)  # default behaviour preserved
    """
    # ⚙️ New (no behaviour change by default): allow strict refusal instead of generic fallback
    #      Set config.STRICT_LOT_SOURCE=True to raise if no specific lot is found.
    strict = CFG("STRICT_LOT_SOURCE", False)  # ← default keeps current behaviour

    # 1) config map (unchanged)
    try:
        if str(security_id) in LOT_SIZES:
            return max(int(LOT_SIZES[str(security_id)]), 1)
        if symbol.upper() in LOT_SIZES:
            return max(int(LOT_SIZES[symbol.upper()]), 1)
    except Exception:
        pass

    # 2) dhan_master.csv (optional) (unchanged logic)
    try:
        from pathlib import Path as _Path
        import pandas as _pd
        mpath = _Path("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv")
        if mpath.exists():
            mdf = _pd.read_csv(mpath)
            sid_col = next((c for c in mdf.columns if str(c).lower().replace("_","") in
                            ("semsmstsecurityid","securityid","sid")), None)
            lot_col = next((c for c in mdf.columns if str(c).lower().replace("_","") in
                            ("lotsize","lot_size","derivativelotsize")), None)
            if sid_col and lot_col:
                row = mdf[mdf[sid_col].astype(str) == str(security_id)]
                if not row.empty:
                    return max(int(row.iloc[0][lot_col]), 1)
    except Exception:
        pass  # keep existing resilience to malformed files

    # 3) fallback (unchanged by default) — but allow strict guard when operator opts in
    if strict:
        raise ValueError(f"STRICT_LOT_SOURCE=True and no lot-size found for {symbol}/{security_id}")
    log.warning(f"lot_size_for: using global LOT_SIZE={LOT_SIZE} for {symbol}/{security_id}")  # visibility only; no behavior change
    return max(int(LOT_SIZE), 1)

# Optional signal filters
USE_VWAP_FILTER: bool  = CFG("USE_VWAP_FILTER", False)  # requires 'volume' in 1m candles
USE_HTF_TREND: bool    = CFG("USE_HTF_TREND", False)    # 5m resample + SMA
USE_SECTOR_CONFIRM: bool = CFG("USE_SECTOR_CONFIRM", False)  # needs sector feed
STRICT_VWAP_VOLUME: bool = CFG("STRICT_VWAP_VOLUME", False)  # if True, skip symbols lacking 'volume'

# EOD hygiene
AUTO_SQUARE_OFF_IST: Tuple[int, int] = CFG("AUTO_SQUARE_OFF_IST", (15, 20))

# Audit / ledger
TRADE_LOG_CSV: str     = CFG("TRADE_LOG_CSV", "trade_log.csv")

# ★ Fixed dynamic list path (do NOT read from config)
DYNAMIC_STOCK_PATH: str = r"D:\Downloads\Dhanbot\dhan_autotrader\dynamic_stock_list.csv"

# ── DATA STRUCTURES ───────────────────────────────────────────────────────────
@dataclass
class TradePlan:
    side: str           # "LONG" or "SHORT"
    entry: float
    stop: float
    target: float
    time: datetime      # timestamp of entry signal candle (used for logs)
    fvg_top: float      # upper bound of FVG zone
    fvg_bottom: float   # lower bound of FVG zone
    note: str           # debug info

# ── TIME HELPERS ──────────────────────────────────────────────────────────────
def now_ist() -> datetime:
    return datetime.now(IST)

def within(hm: Tuple[int, int], cur: Optional[datetime] = None) -> datetime:
    """Return today's datetime at hm (IST)."""
    cur = cur or now_ist()
    return cur.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)

def in_session(cur: Optional[datetime] = None) -> bool:
    cur = cur or now_ist()
    # NSE: clamp to exact cash session (09:15–15:30 IST).
    start = cur.replace(hour=9, minute=15, second=0, microsecond=0)  # ⬅️ was 09:14
    end   = cur.replace(hour=15, minute=30, second=0, microsecond=0)  # ⬅️ was 15:31
    return start <= cur <= end

# ── MARKET DATA ADAPTERS (compatible with your dhan_api wrapper) ──────────────
def get_ohlc_1m(security_id: str, lookback: int = 400) -> pd.DataFrame:
    """
    Return a DataFrame with columns: ['ts','open','high','low','close',('volume'?)] in IST timezone.
    NOTE: We now PRESERVE 'volume' if your feed provides it so VWAP can work.
    """
    if hasattr(dh, "get_ohlc_1m"):
        df = dh.get_ohlc_1m(security_id, lookback=lookback)
    elif hasattr(dh, "get_ohlc"):
        df = dh.get_ohlc(security_id, interval="1m", lookback=lookback)
    elif hasattr(dh, "fetch_candles"):
        df = dh.fetch_candles(security_id, interval="1m", lookback=lookback)
    elif hasattr(dh, "get_historical_price"):
        # ✅ NEW: fallback to your existing dhan_api.get_historical_price(..., interval="1")
        # This path adapts the list-of-dicts response into the normalized OHLCV DataFrame expected by the engine.
        try:
            recs = dh.get_historical_price(security_id, interval="1", limit=lookback)  # ← 1-minute candles
            if not recs:
                raise ValueError("Empty candle payload from get_historical_price()")
            tmp = pd.DataFrame(recs)
            # normalize field names emitted by dhan_api.get_historical_price()
            # expected keys: 'timestamp','open','high','low','close','volume'
            if "timestamp" not in tmp.columns:
                raise ValueError("Candle DF missing 'timestamp' key from get_historical_price()")
            tmp = tmp.rename(columns={"timestamp": "ts"})
            # ensure tz-aware IST
            ts = pd.to_datetime(tmp["ts"], errors="coerce")
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize(IST)
            else:
                ts = ts.dt.tz_convert(IST)
            tmp["ts"] = ts
            # preserve volume when present
            for col in ("open", "high", "low", "close"):
                if col not in tmp.columns:
                    raise ValueError(f"Missing '{col}' in 1m candles from get_historical_price()")
            core_cols = ["ts", "open", "high", "low", "close"] + (["volume"] if "volume" in tmp.columns else [])
            df = tmp[core_cols].sort_values("ts").reset_index(drop=True)
        except Exception as e:
            raise RuntimeError(f"Failed to adapt get_historical_price() to 1m OHLC: {e}")  # explicit, no silent fallback
    else:
        raise NotImplementedError("dhan_api must expose get_ohlc_1m / get_ohlc / fetch_candles / get_historical_price")  # ← extended message

    # Normalize columns (now includes 'volume' if present)
    cols_lower = {c.lower(): c for c in df.columns}
    rename: dict = {}

    # 🔧 CHANGE: robust mapping for timestamp column (common aliases)
    if "ts" not in df.columns:
        for alias in ("timestamp", "time", "datetime", "date_time", "date", "candle_time"):
            if alias in cols_lower:
                rename[cols_lower[alias]] = "ts"
                break
        else:
            # as a last resort, accept names starting with these aliases
            for c in df.columns:
                lc = c.lower()
                if lc.startswith("timestamp") or lc == "t":
                    rename[c] = "ts"
                    break

    # Existing heuristic for OHLCV (kept intact), but re-uses cols_lower
    for need in ["open", "high", "low", "close", "volume"]:
        if need in cols_lower:
            # ensure case-normalization even when a lower-case alias exists in cols_lower
            rename[cols_lower[need]] = need
        else:
            for c in df.columns:
                lc = c.lower()
                if lc == need or lc.startswith(need):
                    rename[c] = need
                    break

    if rename:
        df = df.rename(columns=rename)

    # Ensure 'ts' tz-aware IST
    if "ts" not in df.columns:
        # Build from index if datetime-like
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("Candle DF missing 'ts' column and datetime index.")

    if getattr(df["ts"].dtype, "tz", None) is None:
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(IST)
    else:
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(IST)

    # Preserve volume if present
    core_cols = ["ts", "open", "high", "low", "close"]
    if "volume" in df.columns:
        core_cols.append("volume")
    return df[core_cols].sort_values("ts").reset_index(drop=True)
    
    
def round_to_tick(x: float, *, up: Optional[bool] = None) -> float:
    step = TICK_SIZE
    q = x / step
    if up is True:
        q = math.ceil(q)
    elif up is False:
        q = math.floor(q)
    else:
        q = round(q)
    return round(q * step, 2)

def normalize_stops_targets(side: str, entry: float, stop: float, target: float) -> Tuple[float, float]:
    # round conservatively
    if side.upper() == "LONG":
        stop   = round_to_tick(stop,   up=False)
        target = round_to_tick(target, up=True)
    else:
        stop   = round_to_tick(stop,   up=True)
        target = round_to_tick(target, up=False)
    return stop, target

def verify_bracket_children(order_id: str | int, *, attempts: int = 5, delay: float = 1.2) -> bool:
    """
    Confirms Super(Beta) created both SL & TP legs. Uses dhan_api.get_order_details.
    Accepts multiple broker payload shapes: top-level, data-wrapped, children/legDetails, or bo* keys.
    """
    if not hasattr(dh, "get_order_details"):
        return False if VERIFY_OCO else True

    def _f(x) -> float:
        try: return float(x)
        except Exception: return 0.0

    for _ in range(attempts):
        try:
            raw = dh.get_order_details(order_id)
        except Exception:
            raw = None

        if raw:
            od = (raw.get("data") if isinstance(raw, dict) else None) or raw or {}
            # 1) explicit child arrays
            for key in ("childOrderIds", "childOrders", "children"):
                kids = od.get(key) or []
                if isinstance(kids, (list, tuple)) and len(kids) >= 2:
                    return True

            # 2) leg objects with types
            legs = od.get("legDetails") or od.get("legs") or []
            if isinstance(legs, (list, tuple)) and legs:
                kinds = [str(d.get("type") or d.get("legType") or "").upper() for d in legs if isinstance(d, dict)]
                has_tp = any(("PROFIT" in k) or ("TARGET" in k) or ("TAKE" in k) for k in kinds)
                has_sl = any(("STOP" in k) or ("SL" in k) for k in kinds)
                if has_tp and has_sl:
                    return True

            # 3) alternate scalar keys
            tp_ok = _f(od.get("squareOff") or od.get("takeProfit") or od.get("boProfitValue")) > 0
            sl_ok = _f(od.get("stopLoss")  or od.get("sl")         or od.get("boStopLossValue")) > 0
            if tp_ok and sl_ok:
                return True

        time.sleep(delay)

    return False

# ── ORDER ADAPTERS (uses Super (Beta) if available) ───────────────────────────
def place_bracket_order(symbol: str, security_id: str, side: str, qty: int,
                        entry: float, stop: float, target: float) -> Tuple[int, dict]:
    """
    One canonical implementation:
      1) Try Super(Beta) with retries.
      2) Fallback to generic bracket.
      3) Final fallback: MARKET + manual SL/TP.
    Always uses INTRADAY/DAY on all paths. No unreachable code.
    """
    txn = "BUY" if side.upper() == "LONG" else "SELL"
    stop, target = normalize_stops_targets(side, entry, stop, target)

    # ✅ round by the instrument's lot size; never bump up (respect risk/budget)
    lot = lot_size_for(symbol, security_id)
    qty_used = int((qty // lot) * lot)
    if qty_used < lot:
        return 422, {"error": "quantity below one lot after rounding",
                     "requested_qty": int(qty), "lot_size": int(lot)}

    # NEW: sanitize product type to avoid accidental CNC on intraday scalps
    pt = str(CFG("PRODUCT_TYPE", "INTRADAY")).upper()
    if pt not in ("INTRADAY", "DAY"):
        log.warning(f"{symbol}: unsupported PRODUCT_TYPE '{pt}' – clamping to INTRADAY")
        pt = "INTRADAY"

    # 1) Super (Beta)
    if hasattr(dh, "place_super_order"):
        for attempt in range(1, ORDER_RETRY_ATTEMPTS + 1):
            try:
                # modern signature (MARKET) — 🚫 do NOT send entry_price for MARKET
                code, resp = dh.place_super_order(
                    security_id=security_id,
                    quantity=qty_used,
                    transaction_type=txn,
                    order_type="MARKET",
                    product_type=pt,                           # ← sanitized
                    validity=CFG("VALIDITY", "DAY"),
                    take_profit=float(target),
                    stop_loss=float(stop),
                )
            except TypeError:
                # Robust legacy fallbacks — try WITHOUT entry_price first, then WITH if required
                code, resp = 500, {"error": "legacy_signature_mismatch"}
                try:
                    # A) legacy with product_type + validity, no order_type, no entry_price
                    code, resp = dh.place_super_order(
                        security_id=security_id,
                        quantity=qty_used,
                        transaction_type=txn,
                        bo_profit_value=float(target),
                        bo_stop_loss_value=float(stop),
                        product_type=pt,
                        validity=CFG("VALIDITY", "DAY"),
                    )
                except TypeError:
                    try:
                        # B) legacy with product_type only, no entry_price
                        code, resp = dh.place_super_order(
                            security_id=security_id,
                            quantity=qty_used,
                            transaction_type=txn,
                            bo_profit_value=float(target),
                            bo_stop_loss_value=float(stop),
                            product_type=pt,
                        )
                    except TypeError:
                        try:
                            # C) minimal legacy (some wrappers require entry_price even for MARKET)
                            code, resp = dh.place_super_order(
                                security_id=security_id,
                                quantity=qty_used,
                                transaction_type=txn,
                                entry_price=float(entry),           # ← last-resort
                                bo_profit_value=float(target),
                                bo_stop_loss_value=float(stop),
                            )
                        except Exception as e:
                            code, resp = 500, {"error": str(e)}
                except Exception as e:
                    code, resp = 500, {"error": str(e)}
            except Exception as e:
                code, resp = 500, {"error": str(e)}
    
            if code == 200:
                oid = (resp.get("data", {}) or resp).get("order_id")
                if VERIFY_OCO and oid and not verify_bracket_children(oid):
                    log.error(f"{symbol}: Super order placed but OCO legs not confirmed; order_id={oid}")
                    # 🔒 NEW: auto-cancel parent on OCO verify failure (naked-risk guard)
                    if AUTO_CANCEL_ON_OCO_FAIL and hasattr(dh, "cancel_order"):
                        try:
                            try:
                                c_code, c_resp = dh.cancel_order(oid)
                            except TypeError:
                                c_code, c_resp = dh.cancel_order(order_id=oid)
                            if c_code == 200:
                                log.warning(f"{symbol}: parent order {oid} auto-cancelled due to OCO verify failure.")
                            else:
                                log.warning(f"{symbol}: auto-cancel request failed for {oid}: {c_code} {c_resp}")
                        except Exception as e:
                            log.warning(f"{symbol}: auto-cancel raised: {e}")
                    return 409, {"error": "oco_verification_failed", "order_id": oid}  # ← same behaviour (409)
                return code, resp
    
            time.sleep(ORDER_RETRY_BACKOFF * attempt)
    
    # 2) Generic bracket
    if hasattr(dh, "place_bracket_order"):
        try:
            code, resp = dh.place_bracket_order(
                security_id=security_id,
                quantity=qty_used,
                transaction_type=txn,
                entry_type="MARKET",
                take_profit=target,
                stop_loss=stop,
                product_type=pt,                               # ← sanitized
                validity=CFG("VALIDITY", "DAY"),
            )
        except Exception as e:
            return 500, {"error": str(e)}

        # verify legs just like Super(Beta)
        if code == 200:
            oid = (resp.get("data", {}) or resp).get("order_id")
            if VERIFY_OCO and oid and not verify_bracket_children(oid):
                log.error(f"{symbol}: Bracket order placed but OCO legs not confirmed; order_id={oid}")
                if AUTO_CANCEL_ON_OCO_FAIL and hasattr(dh, "cancel_order"):
                    try:
                        try:
                            c_code, c_resp = dh.cancel_order(oid)
                        except TypeError:
                            c_code, c_resp = dh.cancel_order(order_id=oid)
                        if c_code == 200:
                            log.warning(f"{symbol}: parent order {oid} auto-cancelled due to OCO verify failure.")
                        else:
                            log.warning(f"{symbol}: auto-cancel request failed for {oid}: {c_code} {c_resp}")
                    except Exception as e:
                        log.warning(f"{symbol}: auto-cancel raised: {e}")
                return 409, {"error": "oco_verification_failed", "order_id": oid}  # ← always 409
        return code, resp

    # 3) MARKET + manual SL/TP helpers
    if hasattr(dh, "place_order"):
        try:
            code, resp = dh.place_order(
                security_id,
                qty_used,
                transaction_type=txn,
                order_type="MARKET",
                product_type=pt,                               # ← sanitized
                validity=CFG("VALIDITY", "DAY"),
            )
        except TypeError:
            # minimal legacy wrapper
            code, resp = dh.place_order(security_id, qty_used, transaction_type=txn, order_type="MARKET")

        if code != 200:
            return code, resp

        # best-effort manual child legs
        oid = None
        try:
            oid = (resp.get("data", {}) or resp).get("order_id")
        except Exception:
            oid = None

        tp_ok = True
        sl_ok = True

        if hasattr(dh, "place_take_profit"):
            try:
                dh.place_take_profit(security_id, qty_used, side, target)
            except Exception:
                tp_ok = False
        else:
            tp_ok = False

        if hasattr(dh, "place_stop_loss"):
            try:
                dh.place_stop_loss(security_id, qty_used, side, stop)
            except Exception:
                sl_ok = False
        else:
            sl_ok = False

        # avoid leaving a naked parent in fallback path
        if VERIFY_OCO and (not tp_ok or not sl_ok):
            log.error(f"{symbol}: manual SL/TP fallback failed to attach both legs.")
            if AUTO_CANCEL_ON_OCO_FAIL and oid and hasattr(dh, "cancel_order"):
                try:
                    try:
                        c_code, c_resp = dh.cancel_order(oid)
                    except TypeError:
                        c_code, c_resp = dh.cancel_order(order_id=oid)
                    if c_code == 200:
                        log.warning(f"{symbol}: parent order {oid} auto-cancelled (manual legs failed).")
                    else:
                        log.warning(f"{symbol}: auto-cancel request failed for {oid}: {c_code} {c_resp}")
                except Exception as e:
                    log.warning(f"{symbol}: auto-cancel raised: {e}")
            return 409, {"error": "manual_legs_failed", "order_id": oid or ""}  # ← always 409

        return code, resp

    # No known placement function
    raise NotImplementedError("No suitable order placement function found in dhan_api.")

# ── DYNAMIC STOCK LIST LOADER (fixed absolute path) ───────────────────────────
def load_dynamic_list() -> List[Tuple[str, str]]:
    """
    Load (Symbol, Security_Id) pairs from the fixed CSV.
    Fails loudly on missing/empty/malformed data.
    Canonical keys after normalization (lowercase, no underscores):
      - symbol: ["symbol","tradingsymbol","semtradingsymbol","name","ticker"]
      - security_id: ["securityid","semsmstsecurityid","secid","sid"]
    """
    p = Path(DYNAMIC_STOCK_PATH)
    if not p.exists():
        raise FileNotFoundError(f"dynamic_stock_list.csv not found at: {p}")

    df = pd.read_csv(p)
    if df.empty:
        raise ValueError("dynamic_stock_list.csv is empty.")

    def _norm(s: str) -> str:
        return str(s).lower().replace("_", "").strip()

    norm_map = { _norm(c): c for c in df.columns }

    sym_key = next((k for k in ("symbol","tradingsymbol","semtradingsymbol","name","ticker") if k in norm_map), None)
    sid_key = next((k for k in ("securityid","semsmstsecurityid","secid","sid") if k in norm_map), None)

    sym_col = norm_map.get(sym_key)
    sid_col = norm_map.get(sid_key)

    if sym_col is None or sid_col is None:
        raise KeyError("dynamic_stock_list.csv must include Symbol and Security_Id (common aliases supported).")

    out: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        sym = str(row[sym_col]).strip()
        sid = str(row[sid_col]).strip()
        if sym and sid and sid.lower() != "nan":
            out.append((sym, sid))

    if not out:
        raise ValueError("No valid (Symbol, Security_Id) pairs found in dynamic_stock_list.csv.")
    return out

# ── ORB (first 5-min after open) ──────────────────────────────────────────────
def compute_orb(df1m: pd.DataFrame) -> Tuple[float, float]:
    """Compute ORB high/low using 1m candles within ORB_START_HM..ORB_END_HM."""
    now = now_ist()
    start_dt = within(ORB_START_HM, now)
    end_dt   = within(ORB_END_HM,   now)
    sub = df1m[(df1m["ts"] >= start_dt) & (df1m["ts"] < end_dt)]
    if sub.empty:
        raise ValueError("Insufficient candles to compute ORB yet.")
    return float(sub["high"].max()), float(sub["low"].min())

# ── FVG DETECTION (outside the ORB) ───────────────────────────────────────────
def detect_fvg_signals(df1m: pd.DataFrame, orb_high: float, orb_low: float) -> List[TradePlan]:
    """
    Scan all 3-candle windows AFTER ORB_END and yield standard-entry plans (no bounce rule).
    - Require middle candle to CLOSE beyond the ORB (wicks alone don't count).
    - Bullish FVG: mid.low > prev.high
      Entry at close of 3rd candle, stop at mid.low, target = entry + 3R*(entry-stop).
    - Bearish FVG: mid.high < prev.low
    """
    signals: List[TradePlan] = []
    end_dt = within(ORB_END_HM, now_ist())
    df = df1m[df1m["ts"] >= end_dt].reset_index(drop=True)
    if len(df) < 3:
        return signals

    for i in range(2, len(df)):
        prev = df.iloc[i-2]
        mid  = df.iloc[i-1]
        last = df.iloc[i]

        # LONG candidate
        if mid["close"] > orb_high and mid["low"] > prev["high"]:
            entry = float(last["close"])
            stop  = float(mid["low"])
            risk  = entry - stop
            if risk > 0:
                target = entry + RR_RATIO * risk
                gap_top = float(mid["low"])
                gap_bottom = float(prev["high"])
                note = f"STD-LONG FVG @ {last['ts']} | mid.low={mid['low']} prev.high={prev['high']}"
                signals.append(TradePlan("LONG", entry, stop, target, pd.Timestamp(last["ts"]).to_pydatetime(), gap_top, gap_bottom, note))

        # SHORT candidate
        if mid["close"] < orb_low and mid["high"] < prev["low"]:
            entry = float(last["close"])
            stop  = float(mid["high"])
            risk  = stop - entry
            if risk > 0:
                target = entry - RR_RATIO * risk
                gap_top = float(prev["low"])
                gap_bottom = float(mid["high"])
                note = f"STD-SHORT FVG @ {last['ts']} | mid.high={mid['high']} prev.low={prev['low']}"
                signals.append(TradePlan("SHORT", entry, stop, target, pd.Timestamp(last["ts"]).to_pydatetime(), gap_top, gap_bottom, note))

    return signals

# ── BOUNCE RULE (tap the FVG + engulfing confirmation) ────────────────────────
def apply_bounce_rule(df1m: pd.DataFrame, orb_high: float, orb_low: float) -> List[TradePlan]:
    """
    For each valid FVG outside ORB:
      1) Wait for a RETRACE that taps the FVG zone.
      2) Then require an ENGULFING confirmation in trade direction.
      3) Entry on engulfing candle close; stop at engulfing candle's opposite extreme.
      4) Target = 3R.
    """
    plans: List[TradePlan] = []
    end_dt = within(ORB_END_HM, now_ist())
    df = df1m[df1m["ts"] >= end_dt].reset_index(drop=True)
    if len(df) < 5:
        return plans

    # Build list of base FVGs first (standard)
    base = []
    for i in range(2, len(df)):
        prev = df.iloc[i-2]; mid = df.iloc[i-1]; last = df.iloc[i]
        # Bullish base FVG outside ORB
        if mid["close"] > orb_high and mid["low"] > prev["high"]:
            base.append(("LONG", i, float(prev["high"]), float(mid["low"])))   # zone: [prev.high, mid.low]
        # Bearish base FVG outside ORB
        if mid["close"] < orb_low and mid["high"] < prev["low"]:
            base.append(("SHORT", i, float(mid["high"]), float(prev["low"])))  # zone: [mid.high, prev.low]

    # For each base FVG, search forward for tap + engulfing
    for side, idx, z_low, z_high in base:
        # The 3-candle FVG ends at index `idx` (the third candle)
        for k in range(idx+1, len(df)):
            c_prev = df.iloc[k-1]
            c_k = df.iloc[k]

            # Check tap of zone at or before k
            tapped = False
            for j in range(idx+1, k+1):
                c = df.iloc[j]
                if side == "LONG":
                    # FVG zone [z_low(prev.high), z_high(mid.low)]
                    if (c["low"] <= z_high) and (c["high"] >= z_low):
                        tapped = True; break
                else:
                    # SHORT zone [z_low(mid.high), z_high(prev.low)]
                    if (c["high"] >= z_low) and (c["low"] <= z_high):
                        tapped = True; break
            if not tapped:
                continue

            # Engulfing confirmation
            if side == "LONG":
                # bullish body close beyond previous body high
                prev_body_high = max(c_prev["open"], c_prev["close"])
                if c_k["close"] > prev_body_high:
                    entry = float(c_k["close"])
                    # STRICT → SL at engulfing low; CONSERVATIVE → min(engulfing low, prior low)
                    sl_mode = "STRICT" if STRICT_BOUNCE_STOP else "CONSERVATIVE"
                    stop_raw = float(c_k["low"]) if STRICT_BOUNCE_STOP else float(min(c_k["low"], c_prev["low"]))
                    risk = entry - stop_raw
                    if risk <= 0:
                        continue
                    target_raw = entry + RR_RATIO * risk
                    plans.append(TradePlan(
                        "LONG", entry, stop_raw, target_raw, pd.Timestamp(c_k["ts"]).to_pydatetime(),
                        z_high, z_low,
                        f"BOUNCE-LONG [{sl_mode}] | tap@<= {df.iloc[k]['ts']} eng@{c_k['ts']}"
                    ))
                    break
            else:
                # bearish body close beyond previous body low
                prev_body_low = min(c_prev["open"], c_prev["close"])
                if c_k["close"] < prev_body_low:
                    entry = float(c_k["close"])
                    # STRICT → SL at engulfing high; CONSERVATIVE → max(engulfing high, prior high)
                    sl_mode = "STRICT" if STRICT_BOUNCE_STOP else "CONSERVATIVE"
                    stop_raw = float(c_k["high"]) if STRICT_BOUNCE_STOP else float(max(c_k["high"], c_prev["high"]))
                    risk = stop_raw - entry
                    if risk <= 0:
                        continue
                    target_raw = entry - RR_RATIO * risk
                    plans.append(TradePlan(
                        "SHORT", entry, stop_raw, target_raw, pd.Timestamp(c_k["ts"]).to_pydatetime(),
                        z_high, z_low,
                        f"BOUNCE-SHORT [{sl_mode}] | tap@<= {df.iloc[k]['ts']} eng@{c_k['ts']}"
                    ))
                    break

    return plans

# ── SIZING ────────────────────────────────────────────────────────────────────
def calc_quantity(entry: float, stop: float) -> int:
    """Risk-based position sizing with budget cap."""
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 0
    risk_cash = CAPITAL * RISK_PCT
    qty = int(risk_cash // risk_per_share)
    if qty < 1:
        return 0
    # Cap by MAX_BUDGET
    if entry * qty > MAX_BUDGET:
        qty = int(MAX_BUDGET // entry)
    return max(qty, 0)
    
    
LEDGER_HEADERS = ["ts", "symbol", "side", "qty", "entry", "stop", "target", "risk_rs", "order_id", "note"]
def append_trade_ledger(row: List):
    new_file = not Path(TRADE_LOG_CSV).exists()
    with open(TRADE_LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(LEDGER_HEADERS)
        w.writerow(row)

try:
    from db_logger import insert_portfolio_log_to_db  # optional
    _HAVE_DB = True
except Exception:
    _HAVE_DB = False
    
    
def square_off_all(open_orders: List[dict]) -> None:
    """Failsafe: try to close any parents still open at 15:20 IST."""
    for od in list(open_orders):
        oid = od.get("order_id")

        # If we can inspect the order, skip ones already closed/filled (unchanged)
        if hasattr(dh, "get_order_details") and oid:
            det = dh.get_order_details(oid) or {}
            status = (det.get("orderStatus") or det.get("status") or "").upper()
            if "CANCEL" in status or "COMPLETE" in status or "FILLED" in status:
                continue  # assume child legs will handle exit

        # belt/suspenders: place reverse MARKET order with explicit product/validity
        side = "SELL" if od.get("side") == "LONG" else "BUY"

        # ✅ NEW: compute a safer square-off quantity
        #    If order-details expose a remaining/pending or filled qty, prefer that to avoid
        #    over/under-hedging on partial fills (Issue #3). Falls back to od["qty"] unchanged.
        qty_req = int(od.get("qty", 0) or 0)
        qty_to_close = qty_req
        if hasattr(dh, "get_order_details") and oid:
            try:
                det = dh.get_order_details(oid) or {}
                # common keys seen across brokers/wrappers
                filled = det.get("filledQty") or det.get("tradedQuantity") or det.get("quantityTraded") or det.get("filled_quantity")
                pending = det.get("pendingQuantity") or det.get("remainingQuantity") or det.get("unfilledQuantity")
                total = det.get("quantity") or det.get("orderQuantity") or det.get("orderQty") or det.get("qty")
                # normalize numbers
                def _to_int(x):
                    try: return int(float(x))
                    except Exception: return None
                f, p, t = _to_int(filled), _to_int(pending), _to_int(total)

                if p is not None and p > 0:
                    qty_to_close = p                                       # ← most accurate (remaining)
                elif (t is not None) and (f is not None) and (t - f > 0):
                    qty_to_close = t - f                                   # ← compute remaining if keys exist
                elif f is not None and f > 0:
                    qty_to_close = f                                       # ← last resort: hedge filled part
            except Exception:
                pass  # if anything goes wrong, keep original qty_to_close

        # NEW: sanitize product type to avoid CNC during square-off (unchanged)
        pt = str(CFG("PRODUCT_TYPE", "INTRADAY")).upper()
        if pt not in ("INTRADAY", "DAY"):
            log.warning(f"{od['symbol']}: unsupported PRODUCT_TYPE '{pt}' – clamping to INTRADAY for square-off")
            pt = "INTRADAY"

        # ✅ If qty_to_close resolves to 0 or negative (e.g., already netted), skip cleanly
        if not isinstance(qty_to_close, int) or qty_to_close <= 0:
            log.info(f"🧹 Skipping square-off for {od.get('symbol')} — no remaining quantity to close.")
            continue

        try:
            try:
                dh.place_order(
                    od["security_id"],
                    qty_to_close,                                 # ← use reconciled qty (fix)
                    transaction_type=side,
                    order_type="MARKET",
                    product_type=pt,                              # ← sanitized
                    validity=CFG("VALIDITY", "DAY"),
                    super_order=False,
                )
            except TypeError:
                # fallback for simpler wrappers
                dh.place_order(od["security_id"], qty_to_close, transaction_type=side, order_type="MARKET")
            log.info(f"🧹 Square-off sent for {od['symbol']} qty={qty_to_close} (reverse {side}).")  # ← include qty for audit
        except Exception as e:
            log.error(f"Square-off failed for {od['symbol']}: {e}")

# ── MAIN ENGINE ───────────────────────────────────────────────────────────────
def process_symbol(
    symbol: str,
    security_id: str,
    *,
    open_orders: List[dict],
    risk_state: Dict[str, float],
    done_symbols: Optional[set] = None,   # NEW: lets us latch a symbol as “missed” on late start
) -> bool:
    """Returns True if a trade was placed (and counted against MAX_TRADES)."""
    global ABORT_REQUESTED  # ⬅️ allow this function to request an engine abort when configured
    _now = now_ist()
    if (not in_session(_now)) or (_now < within(ORB_END_HM, _now)):
        return False

    try:
        df = get_ohlc_1m(security_id, lookback=450)
    except Exception as e:
        log.error(f"{symbol}: failed to fetch candles: {e}")
        return False
    # Wait until we can compute ORB (handles on-time start with feed delay)
    try:
        orb_high, orb_low = compute_orb(df)
    except Exception as e:
        log.debug(f"{symbol}: ORB not ready yet: {e}")
        return False

    # Optional filters (soft-opt-in for VWAP when volume exists)
    last_close, last_vwap = None, None  # ← compute only if feasible
    if USE_VWAP_FILTER:
        if "volume" not in df.columns:
            if STRICT_VWAP_VOLUME:
                log.info(f"{symbol}: VWAP filter on but feed lacks 'volume'; skipping this symbol (STRICT_VWAP_VOLUME).")
                return False
            log.info(f"{symbol}: proceeding without VWAP – feed lacks 'volume'.")
        else:
            # ✅ restrict VWAP to TODAY's session only (09:15 IST → now)
            session_start = within((9, 15), now_ist())
            df_sess = df[df["ts"] >= session_start].copy()
            if df_sess.empty:
                log.info(f"{symbol}: VWAP filter session empty (>=09:15 IST); skipping this symbol for now.")
                return False

            # NEW: build cumulative sums ONCE and reuse (local "cache" for this pass)
            _tp  = (df_sess["high"] + df_sess["low"] + df_sess["close"]) / 3.0  # NEW
            _vol = df_sess["volume"].astype(float)                                # NEW
            _cum_vol = _vol.cumsum()                                              # NEW

            if _cum_vol.iloc[-1] == 0:
                log.info(f"{symbol}: proceeding without VWAP – cumulative volume is 0.")
            else:
                _safe_cum  = _cum_vol.replace(0, pd.NA)                           # NEW
                _cum_typv  = (_tp * _vol).cumsum()                                # NEW
                last_close = float(df_sess["close"].iloc[-1])                     # NEW
                last_vwap  = float(_cum_typv.iloc[-1] / _safe_cum.iloc[-1])       # NEW
                if not math.isfinite(last_vwap):
                    log.info(f"{symbol}: proceeding without VWAP – last VWAP is NaN/inf.")
                    last_vwap = None

                # NEW: fast lookup for VWAP/close at any timestamp ≤ t (used later)
                _vwap_index = df_sess["ts"]                                       # NEW
                def _vwap_at(ts):                                                 # NEW
                    try:
                        ts = pd.Timestamp(ts)
                        ts = ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)
                        j = _vwap_index.searchsorted(ts, side="right") - 1
                        if j >= 0:
                            return float(_cum_typv.iloc[j] / _safe_cum.iloc[j]), float(df_sess["close"].iloc[j])
                    except Exception:
                        return None
                    return None
    
    
    if USE_HTF_TREND:
        r = df.set_index("ts").resample("5T").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        if len(r) < 30:
            log.info(f"{symbol}: HTF trend filter skipped — insufficient 5m history (<30 bars).")  # CHANGED: log instead of exception
            return False  # CHANGED: soft-skip this symbol for this pass
        r["sma20"] = r["close"].rolling(20).mean()
        last_close_5m, last_sma20 = float(r["close"].iloc[-1]), float(r["sma20"].iloc[-1])

    # Generate trade plans per strategy (unchanged)
    plans = apply_bounce_rule(df, orb_high, orb_low) if USE_BOUNCE_RULE else detect_fvg_signals(df, orb_high, orb_low)
    if not plans:
        return False

    # 🔒 Fixed late-start behavior:
    # If the engine started after ORB_END and this symbol already had a valid plan BEFORE we started,
    # mark it “missed” and never revisit it today. Otherwise only consider plans at/after start.
    if ENGINE_START_TS is not None:
        orb_end_dt = within(ORB_END_HM, now_ist())
        if ENGINE_START_TS >= orb_end_dt:
            prior = [p for p in plans if p.time < ENGINE_START_TS]
            if prior:
                if done_symbols is not None and CFG("LATCH_MISSED_SIGNALS", True):  # NEW: make “missed latch” opt-in; default True keeps behavior
                    done_symbols.add(symbol)
                ts_first = min(p.time for p in prior).strftime("%H:%M:%S")
                log.info(f"{symbol}: late start — valid entry existed at {ts_first}; skipping this symbol for today.")
                if CFG("LATCH_MISSED_SIGNALS", True):  # NEW: only hard-skip when latching is enabled
                    return False
            plans = [p for p in plans if p.time >= ENGINE_START_TS]
            if not plans:
                return False     

    # ✅ NEW: skip already-consumed plans (same signal retried) — opt-in; default keeps current behavior
    if USE_CONSUME_PLAN_LATCH and CONSUMED_PLANS:
        plans = [p for p in plans if (symbol, p.time) not in CONSUMED_PLANS]
        if not plans:
            return False

    # First valid plan
    plan = plans[0]

    # Apply filters to the chosen plan (VWAP only if computed)
    if USE_VWAP_FILTER and (last_vwap is not None):
        vwap_for_check = last_vwap
        close_for_check = last_close

        # NEW (opt-in): evaluate VWAP at the plan’s candle time without recomputing the series
        if CFG("USE_VWAP_AT_SIGNAL_TIME", True) and ("volume" in df.columns):
            try:
                if "_vwap_at" in locals() and callable(_vwap_at):                 # NEW: use cached cumulative sums
                    _pair = _vwap_at(plan.time)
                    if _pair:
                        vwap_for_check, close_for_check = _pair
                else:
                    # Fallback to existing recompute path (unchanged behavior)
                    session_start = within((9, 15), now_ist())
                    _until = pd.Timestamp(plan.time).tz_convert(IST) if pd.Timestamp(plan.time).tzinfo else pd.Timestamp(plan.time, tz=IST)
                    df_sess_sig = df[(df["ts"] >= session_start) & (df["ts"] <= _until)].copy()
                    if not df_sess_sig.empty and df_sess_sig["volume"].sum() > 0:
                        tp_sig = (df_sess_sig["high"] + df_sess_sig["low"] + df_sess_sig["close"]) / 3.0
                        vwap_series = ((tp_sig * df_sess_sig["volume"]).cumsum() / df_sess_sig["volume"].replace(0, pd.NA).cumsum())
                        vwap_for_check = float(vwap_series.iloc[-1])
                        close_for_check = float(df_sess_sig["close"].iloc[-1])
            except Exception:
                pass  # never change behavior on failure

        if plan.side == "LONG" and close_for_check < vwap_for_check:
            log.info(f"{symbol}: rejected by VWAP filter (LONG needs close >= VWAP).")
            return False
        if plan.side == "SHORT" and close_for_check > vwap_for_check:
            log.info(f"{symbol}: rejected by VWAP filter (SHORT needs close <= VWAP).")
            return False
    
    if USE_HTF_TREND:
        if plan.side == "LONG" and last_close_5m < last_sma20:
            log.info(f"{symbol}: rejected by HTF trend filter (LONG needs 5m close >= SMA20).")
            return False
        if plan.side == "SHORT" and last_close_5m > last_sma20:
            log.info(f"{symbol}: rejected by HTF trend filter (SHORT needs 5m close <= SMA20).")
            return False

    if USE_SECTOR_CONFIRM:
        raise NotImplementedError("Sector confirmation requires a sector feed; toggle is provided but disabled.")

    # 1) Size, then round DOWN to the instrument lot; never exceed planned risk/budget
    stop_n, target_n = normalize_stops_targets(plan.side, plan.entry, plan.stop, plan.target)
    per_share = abs(plan.entry - stop_n)                                         # NEW: need before precheck
    raw_qty = calc_quantity(plan.entry, stop_n)                                  # uses normalized stop
    if raw_qty <= 0:
        log.warning(f"{symbol}: qty=0 (risk too high or budget too low). entry={plan.entry:.2f} stop={stop_n:.2f}")
        return False

    # NEW (opt-in): pre-validate remaining daily risk *before* lot rounding
    if MAX_TOTAL_RISK is not None and CFG("PRECHECK_TOTAL_RISK", False):
        remaining = max(0.0, float(MAX_TOTAL_RISK) - float(risk_state.get("committed", 0.0)))
        if remaining <= 0:
            log.info(f"{symbol}: skipping — no remaining daily risk budget.")
            return False
        if per_share > 0:
            max_by_risk = int(remaining // per_share)
            if max_by_risk <= 0:
                log.info(f"{symbol}: skipping — remaining daily risk too small for even 1 share.")
                return False
            raw_qty = min(raw_qty, max_by_risk)                                   # clamp upfront

    lot = lot_size_for(symbol, security_id)
    qty_lot = int((raw_qty // lot) * lot)
    if qty_lot < lot:
        log.info(f"{symbol}: sized {raw_qty} but < one lot ({lot}); skipping to respect risk/budget.")
        return False

    # 2) Compute risk with the FINAL (lot-rounded) quantity
    risk_rs = per_share * qty_lot
    if MAX_TOTAL_RISK is not None:
        next_risk = risk_state.get("committed", 0.0) + risk_rs

        # NEW (opt-in): trim by whole lots to fit remaining risk instead of skipping
        if CFG("PRECHECK_TOTAL_RISK", False) and (next_risk > MAX_TOTAL_RISK):
            while qty_lot >= lot and (risk_state.get("committed", 0.0) + per_share * qty_lot) > MAX_TOTAL_RISK:
                qty_lot -= lot
            if qty_lot < lot:
                log.info(f"{symbol}: post-lot trimming left no quantity within remaining risk; skipping.")
                return False
            risk_rs = per_share * qty_lot  # recompute after trim
            next_risk = risk_state.get("committed", 0.0) + risk_rs

        if next_risk > MAX_TOTAL_RISK:
            log.info(f"{symbol}: skipping – daily risk cap would be exceeded (next={next_risk:.2f} > cap={MAX_TOTAL_RISK}).")
            return False

    # ✅ NEW (opt-in): stale-plan guard — confirms candle recency & LTP drift before placing the order
    if USE_STALE_PLAN_GUARD:
        try:
            # 1) latest 1m candle must be fresh enough
            age_sec = (now_ist() - pd.Timestamp(df["ts"].iloc[-1]).to_pydatetime()).total_seconds()
            if age_sec > MAX_CANDLE_STALENESS_SEC:
                log.info(f"{symbol}: rejecting plan — last 1m candle stale by {age_sec:.0f}s (> {MAX_CANDLE_STALENESS_SEC}s).")
                return False

            # 2) current LTP must be close to planned entry (bounded in R)
            ltp = None
            if hasattr(dh, "get_live_price"):
                try:
                    ltp = dh.get_live_price(symbol, security_id)
                except Exception:
                    ltp = None
            if isinstance(ltp, (int, float)) and ltp not in (None, 429):
                drift = abs(float(ltp) - float(plan.entry))
                tol   = MAX_ENTRY_SLIPPAGE_R * per_share
                if drift > tol:
                    log.info(f"{symbol}: skipping — LTP {float(ltp):.2f} drift {drift:.2f} > {tol:.2f} ({MAX_ENTRY_SLIPPAGE_R}R).")
                    return False
        except Exception as _e:
            # Guard must never break existing flow; it only rejects when clearly stale
            log.debug(f"{symbol}: stale-plan guard check skipped due to: {_e}")

    # 3) Place with the exact qty we used for risk
    code, resp = place_bracket_order(symbol, security_id, plan.side, qty_lot, plan.entry, stop_n, target_n)  # ← pass normalized SL/TP

    if code == 200:
        oid = None
        try:
            oid = resp.get("data", {}).get("order_id") or resp.get("order_id")
        except Exception:
            pass
    
        # 🔎 Optional reconciliation: read filled qty once live
        qty_final = qty_lot
        if RECONCILE_FILLED_QTY and oid and hasattr(dh, "get_order_details"):
            filled = None
            for _ in range(FILL_RETRY_ATTEMPTS):
                try:
                    od = dh.get_order_details(oid) or {}
                    # common broker keys we might see
                    for k in ("filledQty", "tradedQuantity", "quantityTraded", "filled_quantity"):
                        if k in od and str(od[k]).strip() != "":
                            filled = int(float(od[k]))
                            break
                    if filled is not None:
                        break
                except Exception:
                    pass
                time.sleep(FILL_RETRY_DELAY)
            if isinstance(filled, int) and filled > 0:
                qty_final = min(filled, qty_lot)
                if qty_final != qty_lot:
                    log.info(f"{symbol}: partial fill detected: requested={qty_lot} filled={qty_final}")
        else:
            log.info(f"{symbol}: RECONCILE_FILLED_QTY=False — ledger will assume requested qty={qty_lot}.")  # visibility only
        
        # compute committed risk using the FINAL qty
        risk_final = abs(plan.entry - stop_n) * qty_final  # ← normalized stop        
    
        # Update state & logs
        risk_state["committed"] = risk_state.get("committed", 0.0) + risk_final
        append_trade_ledger([
            datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S%z"),
            symbol, plan.side, qty_final, f"{plan.entry:.2f}", f"{stop_n:.2f}", f"{target_n:.2f}",  # ← normalized in ledger
            f"{risk_final:.2f}", oid or "", plan.note
        ])
        if _HAVE_DB:
            try:
                insert_portfolio_log_to_db(datetime.now(IST), symbol, security_id, qty_final,
                                        plan.entry, None, None, order_id=oid,
                                        target_price=target_n, stop_price=stop_n)  # ← normalized to DB
            except Exception as e:
                log.warning(f"{symbol}: DB portfolio log failed: {e}")
    
        if oid:
            open_orders.append({"order_id": oid, "symbol": symbol, "security_id": security_id,
                                "qty": qty_final, "side": plan.side})
    
        def _fmt_fvg(low: float, high: float) -> str:
            # Single place to enforce audit-friendly delimiter
            return f"FVG[{low:.2f}..{high:.2f}]"
        
        log.info(f"✅ {symbol}: {plan.side} FVG{'+bounce' if USE_BOUNCE_RULE else ''} "
                f"qty={qty_final} entry={plan.entry:.2f} SL={stop_n:.2f} TP={target_n:.2f} "
                f"{_fmt_fvg(plan.fvg_bottom, plan.fvg_top)} oid={oid} risk={risk_final:.2f}")
        
        
        return True
    
    else:
        # latch symbol on explicit OCO verification failures to avoid repeated attempts this session
        try:
            err = ""
            if isinstance(resp, dict):
                err = str((resp.get("error") or resp.get("message") or "")).strip().lower()
        except Exception:
            err = ""
        if code == 409 and err in {"oco_verification_failed", "manual_legs_failed"}:
            if done_symbols is not None:
                done_symbols.add(symbol)  # ← respect “one trade per symbol” even on OCO verify failures
            log.error(f"{symbol}: OCO verification failed; latching symbol for today to avoid repeated attempts.")
            # 🔌 If operator asked for abort-on-OCO, request engine stop (opt-in; default False)
            if OCO_FAIL_ABORTS:                       # ⬅️ new
                ABORT_REQUESTED = True                # ⬅️ new: ask main loop to stop this run
                log.error(f"{symbol}: OCO_FAIL_ABORTS=True → requesting engine abort.")  # ⬅️ new
            # ✅ NEW: also mark this specific plan as consumed so the same signal is not retried (opt-in)
            if USE_CONSUME_PLAN_LATCH:
                CONSUMED_PLANS.add((symbol, plan.time))
            return False

        # ✅ NEW: for any other placement failure, optionally consume just this plan (not the whole symbol)
        if USE_CONSUME_PLAN_LATCH:
            CONSUMED_PLANS.add((symbol, plan.time))

        log.error(f"Order failed for {symbol}: code={code} resp={resp}")
        return False

def main():
    if not in_session():
        log.warning("Outside market hours. Script will still run but may not place trades.")

    wl = load_dynamic_list()
    log.info(
        f"FVG-Bounce 3R engine started. dynamic-list-size={len(wl)} "
        f"source={DYNAMIC_STOCK_PATH} ORB={ORB_START_HM}-{ORB_END_HM} "
        f"RR={RR_RATIO} bounce={USE_BOUNCE_RULE}"
    )

    # 🔒 latch engine start for late-start logic
    global ENGINE_START_TS
    ENGINE_START_TS = now_ist()
    if ENGINE_START_TS >= within(ORB_END_HM, ENGINE_START_TS):
        log.info(f"Late-start guard active — engine began at {ENGINE_START_TS.strftime('%H:%M:%S')} (IST).")

    # state for risk cap and open orders
    risk_state: Dict[str, float] = {"committed": 0.0}
    open_orders: List[dict] = []
    sqoff_dt = within(AUTO_SQUARE_OFF_IST)

    trades_done = 0
    done_symbols: set[str] = set()  # used for both 'traded' and 'missed' symbols

    while trades_done < MAX_TRADES:
        # Optional abort latch — respects OCO_FAIL_ABORTS without changing default behaviour
        if ABORT_REQUESTED:  # ⬅️ new
            log.error("⛔ Abort requested (OCO verification failure). Stopping engine loop.")  # ⬅️ new
            break  # ⬅️ new

        # NEW (opt-in): prune filled/cancelled parents from open_orders during the day (housekeeping only)
        if CFG("PRUNE_OPEN_ORDERS", True) and open_orders and hasattr(dh, "get_order_details"):  # NEW
            try:  # NEW
                _keep = []  # NEW
                for _od in open_orders:  # NEW
                    _oid = _od.get("order_id")  # NEW
                    _status = ""  # NEW
                    if _oid:  # NEW
                        _det = dh.get_order_details(_oid) or {}  # NEW
                        _status = str(_det.get("orderStatus") or _det.get("status") or "").upper()  # NEW
                    if not _status or (("CANCEL" not in _status) and ("COMPLETE" not in _status) and ("FILLED" not in _status)):  # NEW
                        _keep.append(_od)  # NEW
                if len(_keep) != len(open_orders):  # NEW
                    log.debug(f"🧹 Pruned {len(open_orders) - len(_keep)} stale parent(s) from open_orders.")  # NEW
                open_orders[:] = _keep  # NEW
            except Exception:
                pass  # NEW: never affect core behavior if the check fails

        # EOD square-off guard
        if now_ist() >= sqoff_dt:
            log.info("⏰ Reached AUTO_SQUARE_OFF time; sending exits and stopping new entries.")
            square_off_all(open_orders)
            break

        # Suspend symbol processing when market is closed; keep square-off guard above active
        if not in_session():
            time.sleep(POLL_SEC)
            continue

        loop_ts = time.time()
        for sym, sid in wl:
            if trades_done >= MAX_TRADES:
                break
            if sym in done_symbols:
                continue  # already traded or marked as missed

            try:
                placed = process_symbol(
                    sym, sid,
                    open_orders=open_orders,
                    risk_state=risk_state,
                    done_symbols=done_symbols,     # NEW: allow latching “missed”
                )
                if placed:
                    trades_done += 1
                    done_symbols.add(sym)          # lock symbol for the day
            except Exception as e:
                log.exception(f"Runtime error on {sym}: {e}")

        # pacing
        elapsed = time.time() - loop_ts
        sleep_for = max(0.0, POLL_SEC - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    log.info("Stopping engine (max trades or square-off reached).")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user. Bye.")
