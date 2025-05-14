import requests
import json
import datetime

# Dhan API credentials
headers = {
    "access-token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ4MDcyMDEzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNjg1NzM1OSJ9.ISl7D5ixliWbjnpWQwSXOXJToLpJ8FEGCIIwZTCKPCk6pOGnrO74jQa1SvZpsHhAm7tC1vjwnK1tH8vXaqoQaQ",
    "client-id": "1106857359",
    "Content-Type": "application/json"
}

# RELIANCE (NSE): Verified security ID
security_id = "2885"

# ✅ Candle Data (Yesterday)
def test_candle_data():
    url = "https://api.dhan.co/v2/charts/intraday"
    
    # Use yesterday's full trading day
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    
    from_date = yesterday.strftime('%Y-%m-%d')
    to_date = yesterday.strftime('%Y-%m-%d')

    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "5",
        "oi": False,
        "from date": from_date,
        "to date": to_date
    }

    print(f"\n📊 Testing 5MIN Candle Data for {from_date}...")
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        data = res.json()
        candles = data.get("data", [])
        if candles:
            print(f"✅ Got {len(candles)} candles. First 3:")
            for row in candles[:3]:
                print(row)
        else:
            print("⚠️ No candle data returned (check if it was a holiday).")
    else:
        print("❌ Failed to fetch candle data:", res.text)

# 🔁 Only test candle when market is closed
if __name__ == "__main__":
    test_candle_data()
