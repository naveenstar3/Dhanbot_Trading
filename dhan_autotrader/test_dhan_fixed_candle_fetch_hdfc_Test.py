import requests
import datetime
import logging

# ✅ Setup logging (as requested by Dhan support)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ4MDcyMDEzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNjg1NzM1OSJ9.ISl7D5ixliWbjnpWQwSXOXJToLpJ8FEGCIIwZTCKPCk6pOGnrO74jQa1SvZpsHhAm7tC1vjwnK1tH8vXaqoQaQ"
CLIENT_ID = "1106857359"

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

payload = {
    "securityId": "1333",  # Replace with a valid known ID if needed
    "exchangeSegment": "NSE_EQ",
    "instrument": "EQUITY",
    "interval": "1",
    "oi": "false",
    "fromDate": "2025-05-12 09:30:00",
    "toDate": "2025-05-13 13:00:00"
}

def fetch_intraday_data():
    url = "https://api.dhan.co/v2/charts/intraday"
    logging.info("🔍 Fetching intraday 5-min candle data...")
    logging.debug(f"Request URL: {url}")
    logging.debug(f"Request Headers: {HEADERS}")
    logging.debug(f"Request Payload: {payload}")

    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        logging.debug(f"Response Status: {response.status_code}")
        logging.debug(f"Response Body: {response.text}")

        if response.status_code == 200:
            data = response.json()
            candles = data.get("data", [])
            print(f"✅ Received {len(candles)} candles.")
            for row in candles[:5]:
                print(row)
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        logging.exception("❌ Exception while fetching candle data:")

if __name__ == "__main__":
    fetch_intraday_data()
