#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline scanner for ORB → FVG → Bounce (3R).
Input CSV (IST): columns = symbol, timestamp_ist, open, high, low, close, volume
Keeps only the latest trading day in the file (09:15–15:30 IST).
Outputs: orb_fvg_bounce_signals.csv (one row per symbol if a valid setup exists).

Matches the spec:
- ORB window: 09:15–09:20 (first five 1-min bars), ORB_HIGH=max(high), ORB_LOW=min(low)
- Valid break requires CLOSE beyond the boundary (no wick-only breaks)
- Displacement FVG outside ORB (prev, mid, last)
- Prefer Bounce (tap of FVG zone + body-to-body engulf in direction), else Standard
- Fixed 3R; bar conflict (TP & SL hittable same bar) counted as SL
- VWAP (H+L+C)/3 × volume from 09:15 up to & including the entry bar; NA if no volume
"""

from __future__ import annotations
import sys, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import pandas as pd
from datetime import datetime, time
import pytz

IST = pytz.timezone("Asia/Kolkata")
ORB_MODE = "STRICT_END"
# ---------- Helpers


def slice_and_validate_session(df: pd.DataFrame, trade_day) -> pd.DataFrame:
    """
    Ensure the 1m feed:
      - is tz-aware IST,
      - is sliced to [09:15, 15:30] inclusive,
      - contains exactly {09:15..09:19}.
    Raises RuntimeError if any of those five minutes are missing.
    Keeps all other bars (09:20..15:30) intact.
    """
    # 1) make timestamps tz-aware IST
    if "ts" not in df.columns:
        raise ValueError("Expected a 'ts' column with timestamps.")
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    ts = ts.dt.tz_convert(IST) if ts.dt.tz is not None else ts.dt.tz_localize(IST)
    df = df.copy()
    df["ts"] = ts

    # 2) strict session slice
    start = pd.Timestamp(datetime.combine(trade_day, time(9,15)), tz=IST)
    end   = pd.Timestamp(datetime.combine(trade_day, time(15,30)), tz=IST)
    df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()

    # 3) validate the first five minutes
    expected = pd.date_range(start, periods=5, freq="min", tz=IST)
    have = set(df["ts"].dt.floor("min").unique())
    missing = [t for t in expected if t not in have]
    if missing:
        raise RuntimeError(f"Fetch incomplete for {trade_day}: missing minutes {[t.time() for t in missing]}")

    return df

def parse_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"symbol","timestamp_ist","open","high","low","close","volume"}
    miss = need - set(c.lower() for c in df.columns)
    # accept case variations
    df.columns = [c.strip().lower() for c in df.columns]
    if miss:
        raise ValueError(f"Input CSV missing columns: {sorted(miss)}")

    # tz-aware IST
    ts = pd.to_datetime(df["timestamp_ist"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df["timestamp_ist"] = ts

    # keep only valid rows
    df = df.dropna(subset=["timestamp_ist","open","high","low","close"])
    df = df.rename(columns={"timestamp_ist":"ts"})
    return df

def latest_trading_day_slice(df: pd.DataFrame) -> pd.DataFrame:
    # use max date present in data (IST)
    last_date = df["ts"].dt.date.max()
    start = pd.Timestamp(datetime.combine(last_date, time(9,15)), tz=IST)
    end   = pd.Timestamp(datetime.combine(last_date, time(15,30)), tz=IST)
    return df[(df["ts"] >= start) & (df["ts"] <= end)].copy()

def orb_window(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    ORB minute pack:
      - STRICT_START: 09:15..09:19
      - STRICT_END:   09:16..09:20
      - AUTO: prefer START; if incomplete but END complete, use END.
    Returns empty if the chosen pack is incomplete.
    """
    df_sym = df_sym.sort_values("ts").reset_index(drop=True)
    d = df_sym["ts"].dt.date.iloc[0]

    start_0915 = pd.Timestamp(datetime.combine(d, time(9,15)), tz=IST)
    pack_start = pd.date_range(start_0915, periods=5, freq="min", tz=IST)   # 09:15..09:19
    pack_end   = pd.date_range(start_0915 + pd.Timedelta(minutes=1),
                               periods=5, freq="min", tz=IST)               # 09:16..09:20

    # rows for the day session
    day_rows = df_sym[df_sym["ts"] >= start_0915].copy()

    def build_pack(expected: pd.DatetimeIndex) -> pd.DataFrame:
        have = (day_rows.set_index("ts")
                .reindex(expected)
                .dropna(subset=["open","high","low","close"]))
        return have.reset_index().rename(columns={"index":"ts"}) if len(have) == 5 else have.iloc[0:0]

    if ORB_MODE == "STRICT_START":
        return build_pack(pack_start)
    if ORB_MODE == "STRICT_END":
        return build_pack(pack_end)

    # AUTO: try START, else END
    w = build_pack(pack_start)
    if len(w) == 5:
        w.attrs["__orb_pack__"] = "START"
        return w
    w2 = build_pack(pack_end)
    if len(w2) == 5:
        w2.attrs["__orb_pack__"] = "END"
        return w2
    return day_rows.iloc[0:0]

def after_orb(df_sym: pd.DataFrame, orb_win: pd.DataFrame) -> pd.DataFrame:
    """
    FIX: Start scanning *after the actual 5th ORB candle*,
    regardless of how the vendor stamps times.
    """
    df_sym = df_sym.sort_values("ts").reset_index(drop=True)
    last_orb_ts = orb_win["ts"].max()
    return df_sym[df_sym["ts"] > last_orb_ts].reset_index(drop=True)

@dataclass
class FVG:
    side: str                  # "LONG" or "SHORT"
    i_prev: int
    i_mid: int
    i_last: int
    zone_low: float
    zone_high: float

@dataclass
class Entry:
    path: str                  # "BOUNCE" or "STANDARD"
    side: str                  # "LONG"/"SHORT"
    time: pd.Timestamp
    entry: float
    stop: float
    target: float

def compute_orb(df_sym: pd.DataFrame) -> Tuple[float, float]:
    w = orb_window(df_sym)
    if len(w) != 5:
        raise ValueError(f"{df_sym['symbol'].iloc[0]}: ORB window has {len(w)} bars (expected 5).")
    return float(w["high"].max()), float(w["low"].min())

def first_valid_fvg(df_after: pd.DataFrame, orb_high: float, orb_low: float) -> Tuple[Optional[FVG], Optional[pd.Timestamp]]:
    """Return first FVG outside ORB and its mid (break) time for CSV."""
    if len(df_after) < 3:
        return None, None
    for i in range(2, len(df_after)):
        prev = df_after.iloc[i-2]
        mid  = df_after.iloc[i-1]
        last = df_after.iloc[i]

        # Bullish FVG: mid.low > prev.high AND mid.close > ORB_HIGH
        if (mid["low"] > prev["high"]) and (mid["close"] > orb_high):
            zone_low, zone_high = float(prev["high"]), float(mid["low"])
            return FVG("LONG", i-2, i-1, i, zone_low, zone_high), pd.Timestamp(mid["ts"])

        # Bearish FVG: mid.high < prev.low AND mid.close < ORB_LOW
        if (mid["high"] < prev["low"]) and (mid["close"] < orb_low):
            zone_low, zone_high = float(mid["high"]), float(prev["low"])
            return FVG("SHORT", i-2, i-1, i, zone_low, zone_high), pd.Timestamp(mid["ts"])
    return None, None

def find_bounce_entry(df_after: pd.DataFrame, fvg: FVG) -> Optional[Entry]:
    """Tap of zone + body-to-body engulf in direction; entry on engulf close; stop per conservative rule."""
    idx = fvg.i_last
    # search forward from the candle after the FVG completes
    for k in range(idx+1, len(df_after)):
        c_prev = df_after.iloc[k-1]
        c_k    = df_after.iloc[k]

        # Tap check: any candle between (idx+1..k) must overlap the zone
        tapped = False
        for j in range(idx+1, k+1):
            c = df_after.iloc[j]
            if fvg.side == "LONG":
                if (c["low"] <= fvg.zone_high) and (c["high"] >= fvg.zone_low):
                    tapped = True; break
            else:
                if (c["high"] >= fvg.zone_low) and (c["low"] <= fvg.zone_high):
                    tapped = True; break
        if not tapped:
            continue

        # Body-to-body engulf
        if fvg.side == "LONG":
            prev_body_high = max(c_prev["open"], c_prev["close"])
            if c_k["close"] > prev_body_high:
                entry = float(c_k["close"])
                stop  = float(min(c_k["low"], c_prev["low"]))  # conservative default
                risk  = entry - stop
                if risk > 0:
                    target = entry + 3.0 * risk
                    return Entry("BOUNCE","LONG", pd.Timestamp(c_k["ts"]), entry, stop, target)
        else:
            prev_body_low = min(c_prev["open"], c_prev["close"])
            if c_k["close"] < prev_body_low:
                entry = float(c_k["close"])
                stop  = float(max(c_k["high"], c_prev["high"]))  # conservative default
                risk  = stop - entry
                if risk > 0:
                    target = entry - 3.0 * risk
                    return Entry("BOUNCE","SHORT", pd.Timestamp(c_k["ts"]), entry, stop, target)
    return None

def standard_entry(df_after: pd.DataFrame, fvg: FVG) -> Optional[Entry]:
    last = df_after.iloc[fvg.i_last]
    mid  = df_after.iloc[fvg.i_mid]
    if fvg.side == "LONG":
        entry = float(last["close"]); stop = float(mid["low"]); risk = entry - stop
        if risk <= 0: return None
        target = entry + 3.0 * risk
        return Entry("STANDARD","LONG", pd.Timestamp(last["ts"]), entry, stop, target)
    else:
        entry = float(last["close"]); stop = float(mid["high"]); risk = stop - entry
        if risk <= 0: return None
        target = entry - 3.0 * risk
        return Entry("STANDARD","SHORT", pd.Timestamp(last["ts"]), entry, stop, target)

def evaluate_outcome(df_after: pd.DataFrame, entry_idx_from_start: int, entry: Entry) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], str, float]:
    """Walk forward bar-by-bar; if TP & SL both hittable in same bar → SL (conservative). Return (tp_time, sl_time, conflict, max_R)."""
    # find index of entry.time
    k0 = df_after.index[df_after["ts"] == entry.time][0]
    tp = entry.target
    sl = entry.stop
    conflict = "NO"
    max_R = 0.0

    for k in range(k0+1, len(df_after)):
        row = df_after.iloc[k]
        hi, lo = float(row["high"]), float(row["low"])

        # per-bar conflict
        hit_tp = (hi >= tp) if entry.side == "LONG" else (lo <= tp)
        hit_sl = (lo <= sl) if entry.side == "LONG" else (hi >= sl)
        if hit_tp and hit_sl:
            conflict = "YES"
            return None, pd.Timestamp(row["ts"]), conflict, max_R

        if hit_sl:
            return None, pd.Timestamp(row["ts"]), conflict, max_R
        if hit_tp:
            return pd.Timestamp(row["ts"]), None, conflict, 3.0  # exactly 3R

        # track MFE in R
        if entry.side == "LONG":
            mfe = max(0.0, hi - entry.entry)
            max_R = max(max_R, mfe / (entry.entry - entry.stop))
        else:
            mfe = max(0.0, entry.entry - lo)
            max_R = max(max_R, mfe / (entry.stop - entry.entry))

    return None, None, conflict, max_R

def scan_symbol(df_sym: pd.DataFrame) -> Optional[Dict]:
    sym = str(df_sym["symbol"].iloc[0])
    df_sym = df_sym.sort_values("ts").reset_index(drop=True)
    # compute ORB window first, then highs/lows from it
    w = orb_window(df_sym)
    if len(w) != 5:
        print(f"ERROR  [{df_sym['symbol'].iloc[0]}]: ORB 5-pack incomplete (START & END) — fix fetch; skipping.")
        return None
    
    pack = getattr(w, "__orb_pack__", "START")  # START if strict_start or auto-start
    if pack == "END":
        print(f"NOTE   [{df_sym['symbol'].iloc[0]}]: Using END-stamped ORB (09:16..09:20).")
    
    
    orb_h, orb_l = float(w["high"].max()), float(w["low"].min())

    # 🔎 DEBUG toggle (set to None to silence)
    DEBUG_SYMBOL = None  # e.g., "IOC"
    if DEBUG_SYMBOL and sym == DEBUG_SYMBOL:
        print(f"\n[DEBUG: {DEBUG_SYMBOL} ORB window (5 rows)]")
        print(w[["ts","open","high","low","close"]].to_string(index=False))
        print(f"ORB_HIGH: {orb_h:.2f}  ORB_LOW: {orb_l:.2f}")
    
    
    # scan after the actual 5th ORB bar
    aft = after_orb(df_sym, w)
    
    
    fvg, break_time = first_valid_fvg(aft, orb_h, orb_l)
    if fvg is None:
        return None

    # prefer bounce
    entry = find_bounce_entry(aft, fvg)
    if entry is None:
        entry = standard_entry(aft, fvg)
        if entry is None:
            return None

    # outcome evaluation
    tp_t, sl_t, conflict, maxR = evaluate_outcome(aft, fvg.i_last, entry)

    # VWAP (informational)
    vwap, pass_flag = None, "NA"
    if "volume" in df_sym.columns:
        start = pd.Timestamp(datetime.combine(df_sym["ts"].dt.date.iloc[0], time(9,15)), tz=IST)
        upto = entry.time
        sub = df_sym[(df_sym["ts"] >= start) & (df_sym["ts"] <= upto)]
        if sub["volume"].sum() > 0:
            tp = (sub["high"] + sub["low"] + sub["close"]) / 3.0
            vwap = float((tp * sub["volume"]).sum() / sub["volume"].sum())
            if entry.side == "LONG":
                pass_flag = "YES" if entry.entry >= vwap else "NO"
            else:
                pass_flag = "YES" if entry.entry <= vwap else "NO"

    prev = aft.iloc[fvg.i_prev]; mid = aft.iloc[fvg.i_mid]; last = aft.iloc[fvg.i_last]

    row = {
        "symbol": sym,
        "direction": entry.side,
        "entry_path": entry.path,
        "orb_high": round(orb_h, 2),
        "orb_low": round(orb_l, 2),
        "orb_break_time": break_time,                 # mid candle close time
        "fvg_prev_time": prev["ts"],
        "fvg_mid_time":  mid["ts"],
        "fvg_last_time": last["ts"],
        "fvg_zone_low":  round(fvg.zone_low, 2),
        "fvg_zone_high": round(fvg.zone_high, 2),
        "tap_back_time": None,                        # filled if bounce found
        "engulf_entry_time": entry.time,
        "entry_price": round(entry.entry, 2),
        "stop_price":  round(entry.stop, 2),
        "target_3R":   round(entry.target, 2),
        "tp_hit_time": tp_t,
        "sl_hit_time": sl_t,
        "bar_conflict_tp_sl": conflict,
        "max_R_reached": round(float(maxR), 3),
        "vwap_at_entry": round(vwap, 3) if isinstance(vwap, float) else None,
        "vwap_filter_passed": pass_flag,
    }

    # fill tap_back_time if we found bounce (we can infer from first overlap)
    if entry.path == "BOUNCE":
        # re-run minimal search to mark earliest tap time
        tap_ts = None
        for j in range(fvg.i_last+1, aft.index[aft["ts"] == entry.time][0]+1):
            c = aft.iloc[j]
            if entry.side == "LONG":
                if (c["low"] <= fvg.zone_high) and (c["high"] >= fvg.zone_low):
                    tap_ts = pd.Timestamp(c["ts"]); break
            else:
                if (c["high"] >= fvg.zone_low) and (c["low"] <= fvg.zone_high):
                    tap_ts = pd.Timestamp(c["ts"]); break
        row["tap_back_time"] = tap_ts
    return row

def main(in_csv: str = "nse_1m_last_trading_day.csv", out_csv: str = "orb_fvg_bounce_signals.csv"):
    df = parse_input(in_csv)
    df = latest_trading_day_slice(df)
    
    # 🔎 PRE-FLIGHT: assert 09:15..09:19 exist per symbol (IST)
    from datetime import time as _time
    session_date = df["ts"].dt.date.max()
    start_0915   = pd.Timestamp(datetime.combine(session_date, _time(9,15)), tz=IST)
    pack_start   = pd.date_range(start_0915, periods=5, freq="min", tz=IST)
    pack_end     = pd.date_range(start_0915 + pd.Timedelta(minutes=1), periods=5, freq="min", tz=IST)
    
    for sym, g in df.groupby("symbol"):
        have = set(g["ts"].dt.floor("min").unique())
        miss_start = [t for t in pack_start if t not in have]
        miss_end   = [t for t in pack_end   if t not in have]
    
        if ORB_MODE == "STRICT_START" and miss_start:
            print(f"ERROR [{sym}]: missing START pack minutes {[t.time() for t in miss_start]} → fix fetch; skipping.")
        elif ORB_MODE == "STRICT_END" and miss_end:
            print(f"ERROR [{sym}]: missing END pack minutes {[t.time() for t in miss_end]} → fix fetch; skipping.")
        elif ORB_MODE == "AUTO":
            if miss_start and not miss_end:
                print(f"NOTE  [{sym}]: START pack incomplete; END pack present → will use END (09:16..09:20).")
            elif miss_start and miss_end:
                print(f"ERROR [{sym}]: both START and END packs incomplete → fix fetch; skipping.")
    
        # raise SystemExit("Abort: fetch is missing the 09:15 pack for one or more symbols.")
    
    out_rows: List[Dict] = []
    for sym, g in df.groupby("symbol"):
        try:
            r = scan_symbol(g.copy())
            if r: out_rows.append(r)
    
        except Exception as e:
            print(f"WARNING [{sym}]: {e}")

    if not out_rows:
        print("No valid setups found.")
        pd.DataFrame(columns=[
            "symbol","direction","entry_path","orb_high","orb_low","orb_break_time",
            "fvg_prev_time","fvg_mid_time","fvg_last_time","fvg_zone_low","fvg_zone_high",
            "tap_back_time","engulf_entry_time","entry_price","stop_price","target_3R",
            "tp_hit_time","sl_hit_time","bar_conflict_tp_sl","max_R_reached",
            "vwap_at_entry","vwap_filter_passed"
        ]).to_csv(out_csv, index=False)
        return

    out = pd.DataFrame(out_rows)
    # nice ordering + tz-naive ISO strings for CSV
    for c in [c for c in out.columns if "time" in c]:
        out[c] = pd.to_datetime(out[c]).dt.tz_convert(IST).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(out_csv, index=False)
    print(f"✓ Saved {len(out)} rows to {out_csv}")

if __name__ == "__main__":
    in_csv  = sys.argv[1] if len(sys.argv) > 1 else "nse_1m_last_trading_day.csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "orb_fvg_bounce_signals.csv"
    main(in_csv, out_csv)
