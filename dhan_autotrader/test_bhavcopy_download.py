import os
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta

# Paths
BASE_DIR = "D:/Downloads/Dhanbot/nse_bhav"
os.makedirs(BASE_DIR, exist_ok=True)
ZIP_PATH = os.path.join(BASE_DIR, "bhavcopy_latest.zip")
CSV_PATH = os.path.join(BASE_DIR, "bhavcopy_latest.csv")
CACHE_PATH = os.path.join(BASE_DIR, "bhavcopy_cache.csv")
CAPITAL_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/current_capital.csv"

def get_current_capital():
    try:
        df = pd.read_csv(CAPITAL_PATH, header=None)
        capital = float(df.iloc[0, 0])
        if capital <= 0:
            raise ValueError("Capital must be positive.")
        return capital
    except Exception as e:
        print(f"❌ Failed to load capital: {e}")
        return None

def try_download_bhavcopy_for_date(bhav_date):
    day = bhav_date.strftime("%d%b%Y").upper()
    year = bhav_date.strftime("%Y")
    month = bhav_date.strftime("%b").upper()
    url = f"https://www1.nseindia.com/content/historical/EQUITIES/{year}/{month}/cm{day}bhav.csv.zip"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    }

    print(f"\n🌐 Trying NSE ZIP for: {bhav_date.strftime('%Y-%m-%d')} → {url}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if "zip" not in response.headers.get("Content-Type", "").lower():
            print("⚠️ Not a ZIP file. Skipping...")
            return None

        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
            for name in zip_ref.namelist():
                if name.endswith("bhav.csv"):
                    extracted = os.path.join(BASE_DIR, name)
                    return extracted
        print("⚠️ ZIP extracted but CSV not found.")
        return None

    except Exception as e:
        print(f"❌ NSE ZIP failed: {e}")
        return None

def fetch_yahoo_bhavcopy():
    try:
        print("🌐 Fetching from Yahoo Finance...")
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=100"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()

        records = []
        for item in data["finance"]["result"][0]["quotes"]:
            symbol = item.get("symbol", "")
            if symbol.endswith(".NS"):
                symbol = symbol.replace(".NS", "")
                price = item.get("regularMarketPrice", 100.0)
                records.append({"SYMBOL": symbol.upper(), "CLOSE": price})

        if records:
            df = pd.DataFrame(records)
            return df
        print("⚠️ Yahoo returned no records.")
        return None

    except Exception as e:
        print(f"❌ Yahoo fetch failed: {e}")
        return None

def filter_by_capital(df, capital):
    try:
        df = df[df["CLOSE"] <= capital]
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"❌ Capital filter failed: {e}")
        return pd.DataFrame()

def fetch_bhavcopy():
    capital = get_current_capital()
    if capital is None:
        print("🛑 Cannot proceed without valid capital.")
        return False

    today = datetime.now()
    extracted_path = None

    for offset in range(1, 4):
        bhav_date = today - timedelta(days=offset)
        while bhav_date.weekday() >= 5:
            bhav_date -= timedelta(days=1)

        extracted_path = try_download_bhavcopy_for_date(bhav_date)
        if extracted_path:
            break

    if extracted_path:
        try:
            df = pd.read_csv(extracted_path)
            df = df[df["SERIES"] == "EQ"]
            df = df[["SYMBOL", "CLOSE"]]
            df = filter_by_capital(df, capital)
            if df.empty:
                print("⚠️ NSE Bhavcopy: No affordable stocks found.")
                return False
            df.to_csv(CSV_PATH, index=False)
            df.to_csv(CACHE_PATH, index=False)
            print(f"✅ NSE Bhavcopy saved to: {CSV_PATH}")
            return True
        except Exception as e:
            print(f"❌ Error processing NSE Bhavcopy: {e}")

    print("🔁 Trying Yahoo fallback...")
    df = fetch_yahoo_bhavcopy()
    if df is not None:
        df = filter_by_capital(df, capital)
        if df.empty:
            print("⚠️ Yahoo fallback: No affordable stocks found.")
            return False
        df.to_csv(CSV_PATH, index=False)
        df.to_csv(CACHE_PATH, index=False)
        print(f"✅ Yahoo fallback saved to: {CSV_PATH}")
        return True

    print("🔁 Trying cache fallback...")
    if os.path.exists(CACHE_PATH):
        try:
            df = pd.read_csv(CACHE_PATH)
            df = filter_by_capital(df, capital)
            if df.empty:
                print("⚠️ Cache: No affordable stocks in backup.")
                return False
            df.to_csv(CSV_PATH, index=False)
            print(f"♻️ Cache used: {CSV_PATH}")
            return True
        except Exception as e:
            print(f"❌ Cache load failed: {e}")
            return False

    print("🛑 All sources failed.")
    return False

# Run the script
if __name__ == "__main__":
    fetch_bhavcopy()