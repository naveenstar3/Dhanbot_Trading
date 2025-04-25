import csv
from datetime import datetime
import yfinance as yf

# ✅ NSE-listed stock symbols (append ".NS" for Yahoo)
STOCK_LIST = [
    {"symbol": "NHPC"},
    {"symbol": "IRFC"},
    {"symbol": "NLCINDIA"},
    {"symbol": "BEL"},
    {"symbol": "BHEL"}
]

def get_live_price(symbol):
    try:
        stock = yf.Ticker(symbol + ".NS")  # NSE symbols
        data = stock.history(period="1d", interval="1m")
        if data.empty:
            raise Exception("No data received.")
        return round(data["Close"].iloc[-1], 2)
    except Exception as e:
        raise Exception(f"Error fetching price for {symbol}: {e}")

def log_live_prices(price_data):
    filename = "live_prices_log.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for entry in price_data:
            writer.writerow([now, entry["symbol"], entry["price"]])

def run_test():
    print("🔍 Testing live price logging...\n")
    price_log = []

    for stock in STOCK_LIST:
        try:
            price = get_live_price(stock["symbol"])
            print(f"{stock['symbol']} → ₹{price}")
            price_log.append({"symbol": stock["symbol"], "price": price})
        except Exception as e:
            print(f"❗ {e}")

    log_live_prices(price_log)
    print("\n✅ Prices saved to live_prices_log.csv")

if __name__ == "__main__":
    run_test()
