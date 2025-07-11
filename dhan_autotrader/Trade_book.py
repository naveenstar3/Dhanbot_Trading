import pandas as pd
import json
import datetime
from dhanhq import dhanhq, DhanContext

# Load credentials
with open("config.json") as f:
    config_data = json.load(f)

ACCESS_TOKEN = config_data["access_token"]
CLIENT_ID = config_data["client_id"]

# Initialize SDK
context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
dhan = dhanhq(context)

try:
    # Step 1: Fetch trade book
    raw_data = dhan.get_trade_book()
    with open("trade_book_debug.json", "w") as dbg:
        json.dump(raw_data, dbg, indent=4)
    print("🛠️ trade_book_debug.json saved for API diagnostics")

    trade_entries = raw_data.get("data", [])

    if trade_entries:
        df = pd.DataFrame(trade_entries)
        df["fetched_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv("full_trade_book.csv", index=False)
        print(f"✅ {len(df)} trades saved to full_trade_book.csv")
    else:
        print("⚠️ No successful trades found. Checking all orders for failures...")

        # Step 2: Fallback to get all orders
        order_list_raw = dhan.get_order_list()
        with open("order_list_debug.json", "w") as f:
            json.dump(order_list_raw, f, indent=4)
        print("🧪 order_list_debug.json saved for diagnostics")

        if order_list_raw and isinstance(order_list_raw, dict):
            raw_orders = order_list_raw.get("data", [])

            if not raw_orders:
                print("⚠️ No order data found in fallback.")
            else:
                orders_df = pd.DataFrame(raw_orders)

                # Filter non-TRADED only
                if "orderStatus" in orders_df.columns:
                    orders_df = orders_df[orders_df["orderStatus"] != "TRADED"]

                orders_df["fetched_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                orders_df.to_csv("rejected_orders.csv", index=False)
                print(f"🚫 {len(orders_df)} rejected/cancelled/pending orders saved to rejected_orders.csv")

        elif isinstance(order_list_raw, list):
            # Handle legacy or flat list
            df = pd.DataFrame(order_list_raw)
            df["fetched_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.to_csv("rejected_orders.csv", index=False)
            print(f"🚫 {len(df)} rejected/cancelled/pending orders saved to rejected_orders.csv")

        else:
            print("❌ No order history found or unexpected structure.")

except Exception as e:
    print(f"❌ Error fetching trade or order data: {e}")
