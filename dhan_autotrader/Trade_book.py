import pandas as pd
import json
import datetime
from dhanhq import dhanhq, DhanContext

# Load credentials from config.json
with open("config.json") as f:
    config_data = json.load(f)

ACCESS_TOKEN = config_data["access_token"]
CLIENT_ID = config_data["client_id"]

# Initialize SDK context
context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(context)

try:
    # Fetch trade book from SDK
    raw_data = dhan.get_trade_book()

    # Save raw API response for debug audit
    with open("trade_book_debug.json", "w") as dbg:
        json.dump(raw_data, dbg, indent=4)
    print("🛠️ trade_book_debug.json saved for API diagnostics")

    # ✅ Extract actual trades
    equity_trades = raw_data.get("data", [])

    if not equity_trades:
        print("⚠️ Trade book is empty or has no trades yet.")
    else:
        df = pd.DataFrame(equity_trades)

        # Add local timestamp
        df["fetched_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to CSV
        df.to_csv("full_trade_book.csv", index=False)
        print(f"✅ {len(df)} trades saved to full_trade_book.csv")

except Exception as e:
    print(f"❌ Error fetching trade book: {e}")
