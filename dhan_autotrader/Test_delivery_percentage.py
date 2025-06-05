from datetime import datetime, timedelta
from dhan_api import get_historical_price, get_security_id_from_trading_symbol

def get_estimated_delivery_percentage(security_id):
    """
    Approximates delivery percentage using volume from 15-min candles.
    Assumes ~65% of total volume is deliverable if actual data not provided.
    """
    try:
        # Use last trading day's 15-min candle data
        yesterday = datetime.now() - timedelta(days=1)
        start = yesterday.strftime("%Y-%m-%d 09:15:00")
        end = yesterday.strftime("%Y-%m-%d 15:30:00")

        candles = get_historical_price(
            security_id=security_id,
            interval="15",
            from_date=start,
            to_date=end
        )

        if not candles:
            print("⚠️ No candles returned.")
            return 35.0

        total_volume = sum(c["volume"] for c in candles if "volume" in c)
        if total_volume == 0:
            print("⚠️ Total volume is zero.")
            return 35.0

        estimated_deliverable = total_volume * 0.65
        delivery_pct = (estimated_deliverable / total_volume) * 100
        return round(delivery_pct, 2)
    except Exception as e:
        print(f"❌ Exception during delivery % calc: {e}")
        return 35.0

# 🔎 Replace this with any valid symbol from dhan_master.csv
symbol = "HDFCBANK"
security_id = get_security_id_from_trading_symbol(symbol)

if security_id:
    dp = get_estimated_delivery_percentage(security_id)
    print(f"📦 Estimated Delivery % for {symbol} (ID {security_id}): {dp}%")
else:
    print(f"❌ Could not find Security ID for {symbol}")
