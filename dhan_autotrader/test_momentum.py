import csv
import json
from datetime import datetime
from dhan_api import get_live_price, get_historical_price

# Use local Dhan scrip master CSV path
LOCAL_CSV_PATH = "D:/Downloads/Dhanbot/api-scrip-master.csv"

print("🔍 Starting PSU Momentum Tester (No Buy Logic)")

valid_stocks = []
try:
    with open(LOCAL_CSV_PATH, newline='', encoding='utf-8') as f:
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
                current_price = get_live_price(symbol)
                price_5min_ago = get_historical_price(symbol, minutes_ago=5)

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
for stock in valid_stocks:
    print(f"{stock['symbol']:<10} | Now: Rs.{stock['current_price']:<7} | 5min Ago: Rs.{stock['price_5min_ago']} | Momentum: {round(stock['momentum'], 2)}%")