import requests
import json

# ✅ Load credentials from dhan_config.json
with open("dhan_config.json") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

HOLDINGS_URL = "https://api.dhan.co/holdings"

def get_security_id_from_holdings(symbol: str):
    try:
        response = requests.get(HOLDINGS_URL, headers=HEADERS)
        response.raise_for_status()
        holdings = response.json()  # ✅ response is a list, not a dict

        matched = [
            h for h in holdings
            if h.get("tradingSymbol", "").upper() == symbol.upper()
        ]

        if not matched:
            print(f"❌ No matching holding found for {symbol}")
            return

        for entry in matched:
            print(f"\n🔍 Found Holding for {symbol}")
            print(f"✅ Security ID : {entry['securityId']}")
            print(f"📈 Exchange    : {entry['exchangeSegment']}")
            print(f"🎯 Qty Held    : {entry['netQty']}")
            print(f"💰 Buy Price   : ₹{entry['buyAvg']}")
            print("-" * 50)

    except Exception as e:
        print(f"⚠️ Error fetching holdings: {e}")

# ✅ Run this to test
if __name__ == "__main__":
    get_security_id_from_holdings("POWERGRID")