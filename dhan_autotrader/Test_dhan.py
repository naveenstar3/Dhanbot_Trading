import pandas as pd
from fuzzywuzzy import process
import requests
import time
from bs4 import BeautifulSoup

# Load dhan_master.csv
dhan_df = pd.read_csv("dhan_master.csv")
dhan_df["base_symbol"] = dhan_df["SEM_TRADING_SYMBOL"].str.replace("-EQ", "").str.upper()

# NSE sector indices to scan
sector_indices = [
    "NIFTY BANK", "NIFTY IT", "NIFTY FMCG", "NIFTY FIN SERVICE", "NIFTY AUTO",
    "NIFTY PHARMA", "NIFTY REALTY", "NIFTY METAL", "NIFTY ENERGY"
]

symbol_sector_map = {}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive"
}

session = requests.Session()
print("🔍 Initializing NSE session...")
session.get("https://www.nseindia.com", headers=headers, timeout=10)

def safe_json_response(response):
    try:
        return response.json()
    except:
        try:
            # Try parsing HTML and show NSE block reason
            soup = BeautifulSoup(response.text, 'html.parser')
            err_text = soup.find("title")
            if err_text:
                print(f"🛑 NSE response title: {err_text.text.strip()}")
        except:
            pass
        return None

print("🔍 Fetching sector-wise stock lists from NSE...\n")
for sector in sector_indices:
    try:
        url = f"https://www.nseindia.com/api/equity-stockIndices?index={sector.replace(' ', '%20')}"
        res = session.get(url, headers=headers, timeout=10)
        data = safe_json_response(res)
        if not data or "data" not in data:
            raise ValueError("Invalid NSE response (likely blocked)")

        for item in data["data"]:
            symbol = item["symbol"].strip().upper()
            symbol_sector_map[symbol] = sector

        print(f"✅ Fetched sector: {sector} with {len(data['data'])} stocks")
        time.sleep(1)

    except Exception as e:
        print(f"❌ Failed to fetch {sector}: {str(e)[:80]}")
        continue

# Match safely
def match_sector(symbol):
    if not symbol_sector_map:
        return None
    result = process.extractOne(symbol, list(symbol_sector_map.keys()))
    if result is None:
        return None
    match, score = result
    return symbol_sector_map[match] if score >= 90 else None

print("\n🔁 Matching Dhan symbols to NSE sectors...")
dhan_df["sector"] = dhan_df["base_symbol"].apply(match_sector)

# Save output
dhan_df.to_csv("dhan_master_with_sector.csv", index=False)
matched_count = dhan_df["sector"].notna().sum()

print(f"\n✅ Sector tagging complete.")
print(f"📁 Output saved to: dhan_master_with_sector.csv")
print(f"🎯 Matched {matched_count} out of {len(dhan_df)} symbols")
