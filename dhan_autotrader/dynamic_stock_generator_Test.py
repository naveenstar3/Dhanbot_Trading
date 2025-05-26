# ✅ PART 1: Imports and Configuration
import pandas as pd
import openai
import pytz
import os
import requests
import json
import time as systime
from bs4 import BeautifulSoup
from dhan_api import get_historical_price, get_security_id
from dhan_api import get_current_capital
from utils_logger import log_bot_action
from datetime import datetime, timedelta
import datetime as dt

dhan_master_df = pd.read_csv("D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv")
dhan_master_df = dhan_master_df.rename(columns={
    'SEM_TRADING_SYMBOL': 'symbol',
    'SEM_SMST_SECURITY_ID': 'security_id'
})
dhan_master_df['symbol'] = dhan_master_df['symbol'].str.strip().str.upper()
dhan_master_df['security_id'] = dhan_master_df['security_id'].astype(str).str.strip()
# === Credentials and Headers ===
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)
    
PREMARKET_MODE = True  # Enable this to run after market hours
EXTEND_RSI_LOOKBACK = True  # Enable to fetch 2–3 days of 15-min candles for accurate RSI
FINAL_STOCK_LIMIT = 50
ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
OPENAI_API_KEY = config["openai_api_key"]

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# ✅ Debug check for RELIANCE
print("RELIANCE ID:", get_security_id("RELIANCE"))

# ✅ Helper: Check if market is closed

def write_to_dynamic_stock_csv(passed_symbols, dhan_master_df):
    output_path = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"
    matched_rows = []

    for sym in passed_symbols:
        match = dhan_master_df[dhan_master_df['symbol'] == sym]
        if not match.empty:
            sec_id = match['security_id'].values[0]
            matched_rows.append({'symbol': sym, 'security_id': sec_id})
        else:
            print(f"⚠️ Skipped {sym}: No matching security ID in dhan_master.csv")

    if matched_rows:
        df_out = pd.DataFrame(matched_rows)
        df_out.to_csv(output_path, index=False)
        print(f"✅ Written {len(df_out)} stocks to dynamic_stock_list.csv")
    else:
        print("❌ No valid stocks matched for final CSV output.")

def is_market_closed():
    if PREMARKET_MODE:
        return False
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    weekday = now.weekday()
    hour = now.hour + now.minute / 60.0
    return weekday >= 5 or hour < 9.25 or hour > 15.5

def fetch_latest_price(symbol, security_id):
    if not security_id or not security_id.isdigit() or len(security_id) < 3:
        print(f"⚠️ {symbol} skipped — invalid security_id: {security_id}")
        return None
    now = datetime.now()

    if PREMARKET_MODE or now.hour < 9 or now.hour >= 16:
        india = pytz.timezone("Asia/Kolkata")
        prev_day = dt.datetime.now(india) - dt.timedelta(days=1)
        while prev_day.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
            prev_day -= dt.timedelta(days=1)
    
        from_time = prev_day.replace(hour=15, minute=20, second=0, microsecond=0)
        to_time = from_time + timedelta(minutes=5)
    
    else:
        from_time = now - timedelta(minutes=5)
        to_time = now

    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "1",
        "oi": "false",
        "fromDate": from_time.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": to_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        resp = requests.post("https://api.dhan.co/v2/charts/intraday", headers=HEADERS, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            closes = data.get("close", [])
            if closes:
                return float(closes[-1])
        elif resp.status_code == 400:
            print(f"⚠️ {symbol} rejected (400): Likely unsupported for candles or invalid securityId.")
        else:
            print(f"❌ {symbol} response {resp.status_code}")
    except Exception as e:
        print(f"⚠️ {symbol} LTP fetch error: {e}")
    return None

# ✅ PART 2: Load and Filter Affordable Stocks from dhan_master.csv
def load_dhan_master(path):
    master_list = []
    try:
        reader = pd.read_csv(path)

        for _, row in reader.iterrows():
            symbol = str(row.get('SEM_TRADING_SYMBOL', '')).strip().upper()
            secid = str(row.get('SEM_SMST_SECURITY_ID', '')).strip()
            exch_type = str(row.get("SEM_EXCH_INSTRUMENT_TYPE", "")).strip().upper()
            series = str(row.get("SEM_SERIES", "")).strip().upper()

            # 🛑 Skip if any key data is missing or malformed
            if not symbol or not secid or not secid.isdigit() or len(secid) < 4 or len(symbol) < 3:
                continue

            # 🧹 Skip if symbol looks like junk
            if symbol.startswith(tuple('0123456789')):
                continue

            # ⛔ Reject by instrument types or categories (very important)
            skip_keywords = ['-RE', '-PP', 'SGB', 'TS', 'RJ', 'WB', 'AP', 'PN', 'HP',
                             'SFMP', 'M6DD', 'EMKAYTOOLS', 'ICICM', 'TRUST', 'REIT',
                             'INVIT', 'ETF', 'FUND', 'NIFTY', 'BANKNIFTY']
            skip_exch_types = ['DBT', 'DEB', 'MF', 'GS', 'TB']  # Bonds, MF, Govt
            skip_series = ['SG', 'GS', 'YL', 'MF', 'NC', 'TB']

            if any(kw in symbol for kw in skip_keywords):
                continue
            if exch_type in skip_exch_types:
                continue
            if series in skip_series:
                continue

            # ✅ Passed all filters
            master_list.append((symbol, secid))

    except Exception as e:
        print(f"❌ Error loading dhan_master.csv: {e}")

    return master_list

# ✅ Build affordable stock list

def get_affordable_symbols(master_list):
    capital = get_current_capital()
    affordable = []
    unavailable = []

    total = len(master_list)  # 👈 Get total for progress display

    for idx, (symbol, secid) in enumerate(master_list, 1):  # 👈 Start from 1
        if not secid.isdigit() or len(secid) < 3:
            print(f"⚠️ Skipping {symbol} — invalid security ID: {secid}")
            continue

        price = fetch_latest_price(symbol, secid)
        if price is None:
            unavailable.append(symbol)
        elif price <= capital:
            affordable.append((symbol, secid))
            print(f"✅ Affordable ({idx}/{total}): {symbol} @ ₹{price}")
        else:
            print(f"⛔ Skipped {symbol} ({idx}/{total}) — ₹{price} > ₹{capital}")

        systime.sleep(0.5)

    print(f"📊 Final affordable: {len(affordable)} | Unavailable: {len(unavailable)}")
    return affordable

# ✅ PART 3: Apply Technical, Volume, RSI, and Sentiment Filters

def cache_candles(affordable):
    candle_cache = {}
    total = len(affordable)
    for idx, (symbol, secid) in enumerate(affordable, 1):
        try:
            systime.sleep(0.6)
            print(f"🚀 Candle Fetch {idx}/{total}: {symbol} (5m & 15m)")
            df_5m = pd.DataFrame(get_historical_price(secid, interval="5m", limit=5))
            limit_15m = 60 if EXTEND_RSI_LOOKBACK else 6
            if EXTEND_RSI_LOOKBACK:
                from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d 09:15:00")
                to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df_15m = pd.DataFrame(get_historical_price(
                    secid,
                    interval="15",
                    from_date=from_date,
                    to_date=to_date
                ))
            else:
                df_15m = pd.DataFrame(get_historical_price(secid, interval="15", limit=6))            
            if not df_5m.empty and not df_15m.empty:
                candle_cache[symbol] = {"5m": df_5m, "15m": df_15m}
        except Exception as e:
            print(f"⚠️ Candle fetch failed for {symbol}: {e}")
    return candle_cache

def technical_filter(candle_cache):
    passed = []
    for symbol, frames in candle_cache.items():
        try:
            df = frames["5m"]
            change_pct = ((df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]) * 100
            if change_pct >= 0.1:
                passed.append(symbol)
        except Exception as e:
            print(f"⚠️ Technical filter error for {symbol}: {e}")
    return passed

def volume_surge_filter(candle_cache, symbols):
    passed = []
    for symbol in symbols:
        try:
            df = candle_cache[symbol]["15m"]
            recent_vol = df["volume"].iloc[-3:].mean()
            avg_vol = df["volume"].iloc[:-3].mean()
            if recent_vol > 1.1 * avg_vol:
                passed.append(symbol)
        except Exception as e:
            print(f"⚠️ Volume surge error for {symbol}: {e}")
    return passed

def rsi_filter(candle_cache, symbols):
    passed = []
    total = len(symbols)
    for idx, symbol in enumerate(symbols, 1):
        try:
            df = candle_cache[symbol]["15m"]
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=14).mean().iloc[-1]
            avg_loss = loss.rolling(window=14).mean().iloc[-1]
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            if rsi < 75:
                passed.append(symbol)
            print(f"📉 RSI {idx}/{total}: {symbol} → RSI={round(rsi, 2)}")
        except Exception as e:
            print(f"⚠️ RSI filter error for {symbol}: {e}")
    return passed

def sentiment_filter(symbols):
    filtered = []
    total = len(symbols)
    for idx, symbol in enumerate(symbols, 1):
        try:
            print(f"📰 Sentiment {idx}/{total}: {symbol}")
            url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={symbol.lower()}"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            headlines = [tag.get_text(strip=True) for tag in soup.select(".MT15 h2")][:5]

            if not headlines:
                print(f"⚠️ No news for {symbol}. Assuming neutral-positive sentiment.")
                result = "NEUTRAL"
            else:
                joined = "\n".join(headlines)
                prompt = f"Rate the tone of these news headlines as POSITIVE, NEGATIVE, or NEUTRAL.\n{joined}"
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.choices[0].message.content.strip().upper()

            if "POS" in result or "NEUTRAL" in result:
                filtered.append(symbol)
                print(f"✅ {symbol}: {result}")
            else:
                print(f"❌ {symbol}: {result}")

        except Exception as e:
            print(f"⚠️ Sentiment error for {symbol}: {e}")
    return filtered
	
	# ✅ PART 4: GPT Scoring and Final Stock Selection

def build_momentum_df(candle_cache, selected_symbols):
    df = pd.DataFrame(columns=["symbol", "5min_change_pct", "volume_value"])
    for symbol in selected_symbols:
        try:
            df_5m = candle_cache[symbol]["5m"]
            df_15m = candle_cache[symbol]["15m"]
            change_pct = ((df_5m["close"].iloc[-1] - df_5m["close"].iloc[-2]) / df_5m["close"].iloc[-2]) * 100
            vol_value = df_15m["volume"].iloc[-1] * df_5m["close"].iloc[-1]
            df.loc[len(df)] = {
                "symbol": symbol,
                "5min_change_pct": round(change_pct, 2),
                "volume_value": round(vol_value)
            }
        except Exception as e:
            print(f"⚠️ Error building GPT data for {symbol}: {e}")
    return df

def fallback_safe_pick(filtered, count=FINAL_STOCK_LIMIT):
    return filtered[:count]

def save_final_stock_list(stocks, filepath):
    write_to_dynamic_stock_csv(stocks, dhan_master_df)
    log_bot_action("dynamic_stock_generator.py", "Stock List Updated", "✅ COMPLETE", f"{len(stocks)} stocks saved")
    print(f"✅ Final {len(stocks)} stocks saved to {filepath}: {stocks}")

def select_final_stocks(filtered, candle_cache, output_file):
    df = build_momentum_df(candle_cache, filtered)
    if df.empty:
        print("⚠️ Empty momentum DataFrame. Using fallback.")
        final_stocks = fallback_safe_pick(filtered)
    else:
        print("📊 Skipping GPT. Using top momentum stocks directly...")
        # Sort by 5-min change and volume value
        df_sorted = df.sort_values(["5min_change_pct", "volume_value"], ascending=False)
        final_stocks = df_sorted["symbol"].head(FINAL_STOCK_LIMIT).tolist() if not df_sorted.empty else fallback_safe_pick(filtered, count=FINAL_STOCK_LIMIT)

    save_final_stock_list(final_stocks, output_file)

# ✅ PART 5: Emergency Fallback and Logging

def emergency_override_fallback(affordable_ids, candle_cache):
    print("🚨 Emergency override: picking best fallback stock by momentum...")
    emergency_candidates = []

    for symbol, sec_id in affordable_ids:
        try:
            df_5m = candle_cache.get(symbol, {}).get("5m", pd.DataFrame())
            if df_5m.empty or len(df_5m) < 2:
                continue

            close_now = df_5m["close"].iloc[-1]
            close_prev = df_5m["close"].iloc[-2]
            change_pct = ((close_now - close_prev) / close_prev) * 100

            if change_pct > 0:
                emergency_candidates.append((symbol, change_pct))
        except Exception:
            continue

    emergency_candidates = sorted(emergency_candidates, key=lambda x: x[1], reverse=True)

    if emergency_candidates:
        top_symbol = emergency_candidates[0][0]
        log_bot_action("dynamic_stock_generator.py", "Emergency Pick", "⚠️ FORCED", f"{top_symbol}")
        return [top_symbol]
    else:
        safe_symbol = [symbol for symbol, _ in affordable_ids][:1]
        log_bot_action("dynamic_stock_generator.py", "Emergency Pick", "⚠️ BLIND", f"{safe_symbol}")
        return safe_symbol

def finalize_stock_selection(final_stocks, affordable_ids, candle_cache, output_file):
    if not final_stocks:
        final_stocks = emergency_override_fallback(affordable_ids, candle_cache)
    save_final_stock_list(final_stocks, output_file)

def save_filter_summary(stats):
    file = "D:/Downloads/Dhanbot/dhan_autotrader/filter_summary_log.csv"
    header = ",".join(stats.keys())
    row = ",".join(str(x) for x in stats.values())
    date = datetime.now().strftime("%Y-%m-%d")
    
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("date," + header + "\n")
    with open(file, "a") as f:
        f.write(f"{date},{row}\n")

# ✅ PART 6: Main Execution Wrapper

def run_dynamic_stock_selection():
    print("🚀 Starting dynamic stock selection..." + (" (PRE-MARKET MODE)" if PREMARKET_MODE else ""))

    if is_market_closed():
        print("⏸️ Market is closed. Exiting.")
        return

    master_path = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
    output_file = "D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv"

    master_list = load_dhan_master(master_path)
    affordable_ids = get_affordable_symbols(master_list)

    if not affordable_ids:
        print("🚨 No affordable stocks found. Exiting.")
        return

    candle_cache = cache_candles(affordable_ids)

    # Apply filters
    technical = technical_filter(candle_cache)
    volume = volume_surge_filter(candle_cache, technical)
    sentiment = sentiment_filter(volume)
    rsi_passed = rsi_filter(candle_cache, sentiment)

    if not rsi_passed:
        print("⚠️ No stocks passed all filters. Triggering emergency fallback...")
        final_stocks = []
    else:
        final_stocks = rsi_passed

    # Final decision and output
    select_final_stocks(final_stocks, candle_cache, output_file)
    filter_stats = {
    "total_scanned": len(master_list),
    "affordable": len(affordable_ids),
    "technical_passed": len(technical),
    "volume_passed": len(volume),
    "sentiment_passed": len(sentiment),
    "rsi_passed": len(rsi_passed),
    "dynamic_list_selected": len(open(output_file).read().strip().splitlines())
    }
    save_filter_summary(filter_stats)
    

# 🟢 Trigger execution if run as main script
if __name__ == "__main__":
    run_dynamic_stock_selection()
