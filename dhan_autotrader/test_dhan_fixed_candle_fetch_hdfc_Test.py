import requests
import json
import datetime
import pytz

# Load credentials directly from dhan_config.json
with open("dhan_config.json", "r") as file:
    config = json.load(file)

access_token = config["access_token"]
client_id = config["client_id"]

# Fixed parameters for testing HDFC
security_id = "1333"  # HDFCBANK security ID (known working)
exchange_segment = "NSE_EQ"
instrument = "EQUITY"
interval = "15"
oi = "false"

# Generate candle time range
ist = pytz.timezone("Asia/Kolkata")
now = datetime.datetime.now(ist)
from_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
to_time = now.strftime("%Y-%m-%d %H:%M:%S")
from_time_str = from_time.strftime("%Y-%m-%d %H:%M:%S")

# Prepare request
url = "https://api.dhan.co/v2/charts/intraday"
headers = {
    "access-token": access_token,
    "client-id": client_id,
    "Content-Type": "application/json"
}
payload = {
    "securityId": security_id,
    "exchangeSegment": exchange_segment,
    "instrument": instrument,
    "interval": interval,
    "oi": oi,
    "fromDate": from_time_str,
    "toDate": to_time
}

print("🚀 Sending payload:")
print(json.dumps(payload, indent=2))

# Execute request
response = requests.post(url, headers=headers, json=payload)

# Show result
if response.status_code == 200:
    print("✅ Success")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
