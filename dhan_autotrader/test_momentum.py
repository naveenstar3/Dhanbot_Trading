import csv
import json
import dhanhq

# --- CONFIGURATIONS ---
CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_config.json"
CSV_MASTER_PATH = "D:/Downloads/Dhanbot/api-scrip-master.csv"

# Load credentials from JSON
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        CLIENT_ID = config["client_id"]
        ACCESS_TOKEN = config["access_token"]
except Exception as e:
    print(f"⚠️ Error loading config: {e}")
    exit()

print("🔍 Starting PSU Momentum Tester (Official Intraday Data)")

# Set credentials
dhanhq.client_id = CLIENT_ID
dhanhq.access_token = ACCESS_TOKEN

valid_stocks = []

try:
    with open(CSV_MASTER_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print("Headers in CSV:", reader.fieldnames)

        for row in reader:
            if "EQUITY" not in row.get("SEM_EXCH_INSTRUMENT_TYPE", "").upper():
                continue
            if row.get("SEM_SEGMENT", "").upper() != "NSE_EQ":
                continue
            if "PSU" not in row.get("SM_SYMBOL_NAME", "").upper() and "PSU" not in row.get("SEM_TRADING_SYMBOL", "").upper():
                continue

            try:
                symbol = row['SEM_TRADING_SYMBOL']
                security_id = row['SEM_SECURITY_ID']

                # Fetch intraday minute data (last 5 min candles)
                data = dhanhq.intraday_minute_data(
                    security_id=security_id,
                    exchange_segment="NSE",
                    instrument_type="EQUITY"
                )

                candles = data.get("data", [])

                if len(candles) < 6:
                    print(f"⚠️ Not enough candles for {symbol}")
                    continue

                # Extract current price and 5-min-ago close price
                latest_candle = candles[-1]
                candle_5min_ago = candles[-6]

                current_price = float(latest_candle['close'])
                price_5min_ago = float(candle_5min_ago['close'])

                if price_5min_ago <= 0:
                    continue

                momentum = ((current_price - price_5min_ago) / price_5min_ago) * 100

                valid_stocks.append({
                    "symbol": symbol,
                    "current_price": current_price,
                    "price_5min_ago": price_5min_ago,
                    "momentum": momentum
                })

            except Exception as e:
                print(f"❗ Skipping {row.get('SEM_TRADING_SYMBOL', 'UNKNOWN')}: {e}")
except Exception as e:
    print(f"⚠️ Error reading local CSV: {e}")

valid_stocks.sort(key=lambda x: x["momentum"], reverse=True)

print("\n📊 Momentum Ranking:")
print("-----------------------------------------------")

if not valid_stocks:
    print("No PSU stocks with enough intraday momentum found today.")
else:
    for stock in valid_stocks:
        print(f"{stock['symbol']:<10} | Now: Rs.{stock['current_price']:<7} | 5min Ago: Rs.{stock['price_5min_ago']} | Momentum: {round(stock['momentum'], 2)}%")