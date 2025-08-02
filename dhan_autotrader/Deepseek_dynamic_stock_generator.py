"""
Deepseek_dynamic_stock_generator.py
===================================

This script generates a dynamic list of small‑cap and mid‑cap stocks each
morning for intraday trading.  It reads API credentials and capital
allocation from config.json, maps symbols to security IDs via
dhan_master.csv, assigns sectors using Sector_Map.csv, fetches live and
historical data from Dhan, applies a series of filters (price, market
cap, volume, ATR, SMA, RSI) mirroring the checks in the main trading
engine, and ranks survivors by recent momentum.  To align with your
sector‑rotation logic, it then keeps only sectors whose average stock
momentum is positive.  The top 50 stocks are written to
dynamic_stock_list.csv.

IMPORTANT: Network access to the NSE and Dhan APIs is required.

"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import pytz
from nsepython import nsefetch  # relies on your installed NSE library
from rapidfuzz import fuzz, process 

# ========== Paths (update if your environment differs) ==========
CONFIG_PATH: str = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"
MASTER_CSV: str = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
SECTOR_MAP_CSV: str = "D:/Downloads/Dhanbot/dhan_autotrader/Sector_Map.csv"
OUTPUT_CSV: str = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"

# ========== Filter thresholds ==========
MIN_MARKET_CAP: float = 5000.0       # in Cr
MAX_PRICE: float = 500.0             # ensures you can buy at least 10 shares
MIN_AVG_VOLUME: float = 100_000.0    # average volume over 5 days
MIN_ATR: float = 1.2                 # minimum average true range (₹)

# Index names defining the small/mid‑cap universe
SMALL_CAP_INDEX: str = "NIFTY SMALLCAP 250"
MID_CAP_INDEX: str = "NIFTY MIDCAP 150"

# Mapping of NSE sector indices to trading‑engine sector keys (not used for
# mapping but kept for completeness if you wish to extend the script)
SECTOR_INDEX_MAP: Dict[str, str] = {
    "NIFTY BANK": "BANKING",
    "NIFTY IT": "IT",
    "NIFTY FMCG": "FMCG",
    "NIFTY FIN SERVICE": "FINANCIAL SERVICES",
    "NIFTY AUTO": "AUTO",
    "NIFTY PHARMA": "PHARMACEUTICALS",
    "NIFTY REALTY": "REALTY",
    "NIFTY METAL": "METAL",
    "NIFTY ENERGY": "ENERGY",
    "NIFTY MEDIA": "MEDIA",
    "NIFTY PSU BANK": "BANKING",
    "NIFTY PRIVATE BANK": "BANKING",
    "NIFTY OIL & GAS": "ENERGY",
    "NIFTY CONSUMER DURABLES": "FMCG",
    "NIFTY HEALTHCARE": "PHARMACEUTICALS",
    "NIFTY INFRA": "INFRASTRUCTURE",
    "NIFTY SERVICES SECTOR": "SERVICES",
    "NIFTY COMMODITIES": "METAL",
}

# ---------- Utility functions ----------

def is_market_open():
    india = pytz.timezone("Asia/Kolkata")
    now = datetime.now(india)
    market_open = dtime(9, 15)
    market_close = dtime(15, 30)
    return market_open <= now.time() <= market_close

def warn(msg: str) -> None:
    print(f"⚠️  {msg}")

def load_config(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["client_id", "access_token", "capital"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in config.json: {', '.join(missing)}")
    return cfg

def load_master_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master CSV not found: {path}")
    df = pd.read_csv(path)
    if "SM_SYMBOL_NAME" not in df.columns or "SEM_SMST_SECURITY_ID" not in df.columns:
        raise KeyError("Master CSV must contain SM_SYMBOL_NAME and SEM_SMST_SECURITY_ID")
    df["base_symbol"] = (
        df["SM_SYMBOL_NAME"]
        .astype(str)
        .str.replace("-EQ", "", regex=False)
        .str.strip()
        .str.upper()
    )
    return df

def load_sector_map(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        warn(f"Sector map file not found: {path}")
        return None
    df = pd.read_csv(path)
    required = {"SEM_SMST_SECURITY_ID", "SM_SYMBOL_NAME", "SECTOR"}
    missing = required - set(df.columns)
    if missing:
        warn(f"Sector map missing columns: {', '.join(missing)}")
        return None
    df = df.copy()
    df["sec_id_str"] = df["SEM_SMST_SECURITY_ID"].astype(str).str.strip()
    df["base_symbol"] = (
        df["SM_SYMBOL_NAME"]
        .astype(str)
        .str.replace("-EQ", "", regex=False)
        .str.strip()
        .str.upper()
    )
    df["sector"] = df["SECTOR"].astype(str).str.strip()
    return df[["sec_id_str", "base_symbol", "sector"]]

def get_index_constituents(index_name: str) -> List[str]:
    try:
        url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_name.replace(' ', '%20')}"
        data = nsefetch(url)
        symbols: List[str] = []
        for item in data.get("data", []):
            sym = str(item.get("symbol", "")).strip().upper()
            if sym and "NIFTY" not in sym:
                symbols.append(sym)
        return symbols
    except Exception as e:
        warn(f"Failed to fetch constituents for {index_name}: {e}")
        return []

def normalize_sector_name(raw_sector: str) -> str:
    """Map various sector descriptions to the canonical sector keys."""
    sec = raw_sector.upper()
    # Special / compound sectors
    if "AUTO" in sec and ("ANC" in sec or "ANCILLARIES" in sec):
        return "AUTO ANCILLARIES"
    if "INSUR" in sec:
        return "INSURANCE"
    if "AGRO" in sec or "CHEM" in sec:
        return "AGROCHEMICALS"
    if "LOGIST" in sec or "TRANSPORT" in sec:
        return "LOGISTICS"
    if "POWER" in sec:
        return "POWER"
    # General sectors
    if "BANK" in sec:
        return "BANKING"
    if "FMCG" in sec or "CONSUMER" in sec:
        return "FMCG"
    if "IT" in sec or "TECH" in sec:
        return "IT"
    if "PHARM" in sec or "HEALTH" in sec or "MEDIC" in sec:
        return "PHARMACEUTICALS"
    if "AUTO" in sec:
        return "AUTO"
    if "METAL" in sec or "STEEL" in sec or "MINING" in sec:
        return "METAL"
    if "ENERGY" in sec or "OIL" in sec or "GAS" in sec:
        return "ENERGY"
    if "REALTY" in sec or "PROPERTY" in sec:
        return "REALTY"
    if "INFRA" in sec or "CONSTRUCTION" in sec:
        return "INFRASTRUCTURE"
    if "FIN" in sec or "NBFC" in sec:
        return "FINANCIAL SERVICES"
    if "MEDIA" in sec or "ENTERTAINMENT" in sec:
        return "MEDIA"
    if "SERVICES" in sec:
        return "SERVICES"
    return "OTHER"

# ---------- Dhan API helpers ----------

def get_headers(cfg: Dict[str, object]) -> Dict[str, str]:
    return {
        "access-token": str(cfg["access_token"]),
        "client-id": str(cfg["client_id"]),
        "Content-Type": "application/json",
    }

def get_ltp(secid: str, headers: Dict[str, str]) -> Optional[float]:
    try:
        url = f"https://api.dhan.co/quotes/isin?security_id={secid}&exchange=NSE"
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        ltp = data.get("lastTradedPrice") or data.get("ltp") or data.get("openPrice")
        return float(ltp) if ltp is not None else None
    except Exception:
        return None

def get_intraday_candles(secid: str, headers: Dict[str, str], interval: int, days: int) -> Optional[pd.DataFrame]:
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        payload = {
            "securityId": secid,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": str(interval),
            "oi": "false",
            "fromDate": from_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": to_date.strftime("%Y-%m-%d %H:%M:%S"),
        }
        url = "https://api.dhan.co/v2/charts/intraday"
        max_retries = 3
        for attempt in range(max_retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 429:
                wait = (2 ** attempt) + np.random.random()
                warn(f"Rate limited while fetching candles for {secid}. Waiting {wait:.1f}s…")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                warn(f"Candle fetch failed (status {resp.status_code}) for {secid}")
                return None
            data = resp.json()
            required = {"open", "high", "low", "close", "volume", "timestamp"}
            if not required.issubset(data.keys()):
                warn(f"Incomplete candle data for {secid}")
                return None
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["close"],
                    "volume": data["volume"],
                }
            )
            return df
        return None
    except Exception as e:
        warn(f"Exception fetching candles for {secid}: {e}")
        return None

def calculate_rsi_series(closes: np.ndarray, period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    for delta in deltas[period:]:
        gain = max(delta, 0)
        loss = -min(delta, 0)
        up = (up * (period - 1) + gain) / period
        down = (down * (period - 1) + loss) / period
        rs = up / down if down != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def calculate_sma(values: np.ndarray, period: int = 20) -> Optional[float]:
    if len(values) < period:
        return None
    return float(np.mean(values[-period:]))

# ---------- Main scanning function ----------

def scan_stocks() -> List[Dict[str, object]]:
    cfg = load_config(CONFIG_PATH)
    capital: float = float(cfg["capital"])
    headers = get_headers(cfg)
    master_df = load_master_csv(MASTER_CSV)
    smallcap = get_index_constituents(SMALL_CAP_INDEX)
    midcap = get_index_constituents(MID_CAP_INDEX)
    universe = set(smallcap + midcap)

    print(f"📊 Universe size: {len(universe)}")

    if not universe:
        warn("No symbols fetched for small/mid cap indices.")
        return []

    # Normalize and fuzzy match base_symbol to universe
    def normalize_for_match(name: str) -> str:
        ignore_words = {
            "LIMITED", "LTD", "LIMTED", "INDIA", "CO", "COMPANY",
            "GROUP", "IND", "PLC", "CORP", "CORPORATION", "PVT", "PRIVATE",
            "BANK", "BANKING", "FINANCE", "FINANCIAL", "SERVICES", "SERVICE"
        }
        name = name.upper().replace("&", "AND").replace("-", " ")
        tokens = name.split()
        clean_tokens = [t for t in tokens if t not in ignore_words and len(t) > 1]
        return " ".join(clean_tokens)

    master_df["normalized_symbol"] = master_df["base_symbol"].apply(normalize_for_match)
    normalized_universe = [(sym, normalize_for_match(sym)) for sym in universe]

    matched_symbols = []
    for orig_sym, norm_sym in normalized_universe:
        match, score, _ = process.extractOne(
            norm_sym,
            master_df["normalized_symbol"],
            scorer=fuzz.token_set_ratio
        )
        if score >= 90:
            matched_row = master_df[master_df["normalized_symbol"] == match]
            if not matched_row.empty:
                matched_symbols.append(matched_row.iloc[0])

    if matched_symbols:
        candidates_df = pd.DataFrame(matched_symbols)
    else:
        candidates_df = pd.DataFrame()

    print(f"📋 Candidates after fuzzy dhan_master.csv match: {len(candidates_df)}")

    if candidates_df.empty:
        warn("No matching symbols found in master CSV for small/mid cap universe.")
        return []

    # Build sector map from the sector CSV
    sector_df = load_sector_map(SECTOR_MAP_CSV)
    sector_map: Dict[str, str] = {}
    if sector_df is not None:
        for _, r in sector_df.iterrows():
            sid = r["sec_id_str"]
            raw = r["sector"]
            sector_map[sid] = normalize_sector_name(raw)

    selected_rows: List[Dict[str, object]] = []
    for _, row in candidates_df.iterrows():
        base_symbol = row["base_symbol"]
        secid = str(row["SEM_SMST_SECURITY_ID"])

        print(f"\n🔍 Evaluating {base_symbol} (SecID: {secid})")

        # Market cap filter via NSE
        mcap = get_market_cap(base_symbol)
        print(f"   🏦 Market cap: {mcap:.2f} Cr")
        if mcap < MIN_MARKET_CAP:
            print("   ❌ Skipped due to market cap")
            continue

        # LTP and price filter via Dhan
        ltp = get_ltp(secid, headers)
        print(f"   📈 LTP: {ltp}")
        if ltp is None or ltp <= 0 or ltp >= MAX_PRICE:
            print("   ❌ Skipped due to LTP")
            continue

        qty = int(capital // ltp)
        if qty < 10:
            print(f"   ❌ Skipped due to insufficient capital (Qty: {qty})")
            continue

        df5 = get_intraday_candles(secid, headers, interval=5, days=5)
        if df5 is None or df5.empty:
            continue
        df5["date"] = df5["timestamp"].dt.date
        daily_vol = df5.groupby("date")["volume"].sum()
        avg_vol = float(daily_vol.tail(5).mean()) if not daily_vol.empty else float("nan")
        if math.isnan(avg_vol) or avg_vol < MIN_AVG_VOLUME:
            continue
        df5["range"] = df5["high"] - df5["low"]
        daily_range = df5.groupby("date")["range"].max()
        atr = float(daily_range.tail(5).mean()) if not daily_range.empty else float("nan")
        if math.isnan(atr) or atr < MIN_ATR:
            continue
        df1 = get_intraday_candles(secid, headers, interval=1, days=25)
        if df1 is None or df1.empty:
            continue
        df1 = df1.sort_values("timestamp")
        closes = df1["close"].astype(float).values
        sma20 = calculate_sma(closes, period=20)
        rsi14 = calculate_rsi_series(closes, period=14)

        print(f"   📊 Qty: {qty} | SMA20: {sma20} | RSI: {rsi14}")

        if sma20 is None or rsi14 is None:
            print("   ❌ Skipped due to missing SMA or RSI")
            continue
        if ltp < sma20 or rsi14 < 50.0 or rsi14 > 70.0:
            print("   ❌ Skipped due to failed momentum filter")
            continue
        try:
            eod = df5.groupby("date")["close"].last().tail(4).values
            if len(eod) >= 2:
                momentum = ((eod[-1] - eod[-2]) / eod[-2]) * 100
            else:
                momentum = 0.0
        except Exception:
            momentum = 0.0
        # Sector assignment: try by security ID; fall back to symbol; else OTHER
        sector = sector_map.get(secid) or sector_map.get(base_symbol) or "OTHER"
        stock_origin = "Small Cap" if mcap < 50000 else "Mid Cap"
        priority_score = float(atr * avg_vol)
        selected_rows.append(
            {
                "symbol": base_symbol,
                "security_id": secid,
                "ltp": round(ltp, 2),
                "quantity": qty,
                "capital_used": round(qty * ltp, 2),
                "avg_volume": int(avg_vol),
                "avg_range": round(atr, 2),
                "potential_profit": round(qty * atr, 2),
                "sma_20": round(sma20, 2),
                "rsi": round(rsi14, 2),
                "stock_origin": stock_origin,
                "priority_score": round(priority_score, 2),
                "sector": sector,
                "market_cap": round(mcap, 2),
                "momentum": round(momentum, 2),
            }
        )
        time.sleep(0.2)  # brief pause to avoid API throttling
    if not selected_rows:
        return []
    # Sort by momentum descending
    selected_rows.sort(key=lambda x: x.get("momentum", 0.0), reverse=True)
    # Sector momentum filtering
    sector_scores: Dict[str, float] = {}
    sector_counts: Dict[str, int] = {}
    for row in selected_rows:
        sec = row.get("sector", "OTHER")
        sector_scores[sec] = sector_scores.get(sec, 0.0) + row.get("momentum", 0.0)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
    sector_avg: Dict[str, float] = {
        sec: total / sector_counts[sec] for sec, total in sector_scores.items()
    }
    bullish_sectors = {sec for sec, avg in sector_avg.items() if avg > 0}
    final_rows = (
        [r for r in selected_rows if r.get("sector") in bullish_sectors]
        if bullish_sectors
        else selected_rows
    )
    final_rows.sort(key=lambda x: x.get("momentum", 0.0), reverse=True)
    return final_rows[:50]

def get_market_cap(symbol: str) -> float:
    """Fetch market cap (Cr) via NSE quote API. Returns 0.0 on failure."""
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        resp = requests.get(url, headers=headers, timeout=7)
        if resp.status_code != 200:
            print(f"⚠️ NSE API failed for {symbol} — Status: {resp.status_code}")
            return 0.0

        data = resp.json()
        price_info = data.get("priceInfo", {})
        mc = price_info.get("marketCap")
        if mc:
            return float(mc) / 1e7  # Convert to Cr

        # Try fallback using issuedSize * lastPrice
        security_info = data.get("securityInfo", {})
        issued = security_info.get("issuedSize")
        last_price = price_info.get("lastPrice")
        if issued and last_price:
            return float(issued) * float(last_price) / 1e7

        print(f"⚠️ Market cap info missing for {symbol}. priceInfo: {price_info}")
        return 0.0

    except Exception as e:
        print(f"❌ Exception fetching market cap for {symbol}: {e}")
        return 0.0

# ---------- CSV writing ----------

def write_dynamic_csv(rows: List[Dict[str, object]], path: str) -> None:
    fieldnames = [
        "symbol",
        "security_id",
        "ltp",
        "quantity",
        "capital_used",
        "avg_volume",
        "avg_range",
        "potential_profit",
        "sma_20",
        "rsi",
        "stock_origin",
        "priority_score",
        "sector",
        "market_cap",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

def main() -> None:
    if not is_market_open():
        print("🕒 Market is currently CLOSED. Please run this script during live market hours (9:15 AM – 3:30 PM IST).")
        exit(1)

    print("🚀 Starting dynamic small/mid‑cap scan…")
    rows = scan_stocks()
    if not rows:
        print("❌ No stocks passed the filters; check network, credentials, or thresholds.")
        return

    write_dynamic_csv(rows, OUTPUT_CSV)
    print(f"✅ Generated {len(rows)} stock entries → {OUTPUT_CSV}")
    print("📊 Top candidates:")
    for r in rows[:5]:
        print(
            f"{r['symbol']} | LTP ₹{r['ltp']:.2f} | Qty {r['quantity']} | "
            f"ATR {r['avg_range']:.2f} | RSI {r['rsi']:.2f} | Sector {r['sector']}"
        )

if __name__ == "__main__":
    main()
