import requests
import pandas as pd
import json

# Load credentials from config.json
with open("config.json") as f:
    config_data = json.load(f)

ACCESS_TOKEN = config_data["access_token"]
CLIENT_ID = config_data["client_id"]
TRADE_BOOK_URL = config_data.get("trade_book_url", "https://api.dhan.co/trade-book")

# Build headers
headers = {
    "accept": "application/json",
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID
}

# Fetch trade book
response = requests.get(TRADE_BOOK_URL, headers=headers)
if response.status_code == 200:
    data = response.json()
    trades = data.get("data", [])

    if not trades:
        print("⚠️ Trade book is empty.")
    else:
        df = pd.DataFrame(trades)
        df.to_csv("full_trade_book.csv", index=False)
        print("✅ Trade book downloaded to full_trade_book.csv")
else:
    print(f"❌ Failed to fetch trade book: {response.status_code}")
