import requests

# --- Single Stock Details ---
stock = {
    "symbol": "HDFCBANK",
    "security_id": "1333"  # Replace with actual if needed
}

# --- Function to Fetch India VIX ---
def get_india_vix():
    """Fetch India VIX from NSE using valid session + headers"""
    url = "https://www.nseindia.com/api/allIndices"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    try:
        session = requests.Session()
        session.headers.update(headers)

        # NSE requires this pre-call to set cookies
        session.get("https://www.nseindia.com", timeout=5)

        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()["data"]

        for item in data:
            if item["index"] == "India VIX":
                return float(item["last"])
        return None
    except Exception as e:
        print(f"❌ Failed to fetch India VIX: {e}")
        return None

# --- Main Trade Simulation ---
def check_vix_and_trade():
    print("📊 Checking India VIX...")
    vix = get_india_vix()

    if vix is None:
        print("❌ Could not retrieve India VIX. Exiting.")
        return

    print(f"📉 India VIX: {vix}")
    if vix > 20:
        print(f"🚫 VIX too high ({vix}) — skipping trade")
        return

    # ✅ VIX Passed — Proceed with mock trade
    print(f"✅ VIX check passed: {vix}")
    print(f"📈 Executing trade for {stock['symbol']} (Security ID: {stock['security_id']})")

# --- Run ---
if __name__ == "__main__":
    check_vix_and_trade()
