import os
import json
import pandas as pd
import requests
from dhan_api import get_live_price  # Must return both LTP and Open if needed

print("\n🔧 Starting generate_live_trending_dashboard.py...")

# Load config
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    print("❌ config.json not found. Exiting.")
    exit()

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

HEADERS = {
    "access-token": config["access_token"],
    "client-id": config["client_id"],
}
print("🔐 Config loaded successfully.")

# Load Nifty100 stock list
nifty_url = "https://www1.nseindia.com/content/indices/ind_nifty100list.csv"
try:
    nifty100_df = pd.read_csv(nifty_url)
    nifty100_symbols = nifty100_df["Symbol"].str.strip().unique().tolist()
    print(f"✅ Fetched {len(nifty100_symbols)} Nifty 100 symbols from NSE.")
except Exception as e:
    print(f"❌ Failed to fetch Nifty100 from NSE: {str(e)}")
    nifty100_symbols = []


# Load dhan_master
master_path = "dhan_master.csv"
if not os.path.exists(master_path):
    print("❌ dhan_master.csv not found. Exiting.")
    exit()

master_df = pd.read_csv(master_path)
print(f"✅ Loaded dhan_master.csv with {len(master_df)} rows.")

# Clean and match
master_df["base_symbol"] = master_df["SEM_TRADING_SYMBOL"].str.replace("-EQ", "", regex=False)
matched_df = master_df[master_df["base_symbol"].isin(nifty100_symbols)]
print(f"🧮 Matching complete: {len(matched_df)} matched, {len(nifty100_symbols) - len(matched_df)} unmatched.")

# Filter to active NSE_EQ segment equities
filtered_df = matched_df[
    (matched_df["SEM_EXCH_INSTRUMENT_TYPE"] == "ES") &
    (matched_df["SEM_SEGMENT"] == "NSE_EQ")
]
print(f"🧹 Filtered to active NSE segment equity stocks: {len(filtered_df)} remaining.")
print("🔍 Sample of filtered stocks:")
print(filtered_df[["base_symbol", "SEM_TRADING_SYMBOL", "SEM_EXCH_INSTRUMENT_TYPE", "SEM_SMST_SECURITY_ID"]].head(10))

# Fetch prices
print("\n📊 Fetching live LTP + Open prices...")
trending = []

for i, row in filtered_df.iterrows():
    symbol = row["base_symbol"]
    secid = str(row["SEM_SMST_SECURITY_ID"])
    print(f"\n🔍 [{i+1}] {symbol} (SecID: {secid})")

    try:
        ltp, open_ = get_live_price(symbol, secid, return_open=True)

        if not ltp or not open_ or ltp <= 0 or open_ <= 0:
            print(f"⚠️ Invalid price data for {symbol}. Skipping.")
            continue

        change_pct = ((ltp - open_) / open_) * 100
        print(f"📊 LTP: ₹{ltp:.2f} | Open: ₹{open_:.2f} | Change %: {change_pct:.2f}")
        trending.append((symbol, secid, ltp, open_, change_pct))

    except Exception as e:
        print(f"❌ Exception while processing {symbol}: {str(e)[:60]}")
        continue

# Save results
if trending:
    trending_df = pd.DataFrame(trending, columns=["Symbol", "SecurityID", "LTP", "Open", "ChangePct"])
    trending_df.sort_values(by="ChangePct", ascending=False, inplace=True)
    trending_df.to_csv("live_trending_stocks.csv", index=False)
    print(f"\n✅ Saved {len(trending_df)} trending stocks to live_trending_stocks.csv")
else:
    print("❌ No trending stocks to write.")
