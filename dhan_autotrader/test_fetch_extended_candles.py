import pandas as pd
from datetime import datetime, timedelta
from dhan_api import get_historical_price

# ✅ RELIANCE sample securityId
security_id = "2885"

# ✅ Fetch 3 days of 15-minute candles
from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d 09:15:00")
to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"Fetching 15m candles for RELIANCE from {from_date} to {to_date}...")

# ⚠️ Ensure you have updated the function to accept from_date/to_date
candles = get_historical_price(
    security_id=security_id,
    interval="15",
    from_date=from_date,
    to_date=to_date
)

# ✅ Display output
df = pd.DataFrame(candles)
print(df.tail(10))  # Show last 10 candles
