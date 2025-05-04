from dhanhq import dhanhq, DhanContext
import json

# ✅ Load your Dhan credentials from the same file you already use
with open("dhan_config.json") as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]

# ✅ Initialize SDK
context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(context)

# ✅ Fetch Historical Minute Chart for POWERGRID
response = dhan.historical_minute_charts(
    symbol='POWERGRID',
    exchange_segment='NSE_EQ',
    instrument_type='EQUITY',
    expiry_code=0,
    from_date='2024-04-01',
    to_date='2024-04-30'
)
print(response)

# ✅ Print first 5 entries
for candle in response[:5]:
    print(candle)

print(dir(dhan))
