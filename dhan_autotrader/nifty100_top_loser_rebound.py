import pandas as pd
import requests
from datetime import datetime, timedelta

# Get Nifty 100 stocks
def get_nifty100_symbols():
    url = 'https://www1.nseindia.com/content/indices/ind_nifty100list.csv'
    headers = {'User-Agent': 'Mozilla/5.0'}
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

# Get historical data using NSE API
def get_price_data(symbol):
    try:
        url = f'https://www.nseindia.com/api/chart-databyindex?index={symbol}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()['grapthData']
        df = pd.DataFrame(data, columns=['Time', 'Price'])
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        return df
    except:
        return None

# Simulate top loser list from EOD % change
def simulate_top_losers(symbols):
    losers = []

    for sym in symbols:
        url = f'https://www.nseindia.com/api/quote-equity?symbol={sym}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers)
            data = r.json()['priceInfo']
            change_pct = data.get('pChange', 0)
            last_price = data.get('lastPrice', 0)
            prev_close = data.get('previousClose', 0)

            if change_pct < -2.0 and last_price > 50:  # heavy drop & not penny
                losers.append({
                    'symbol': sym,
                    'change_pct': change_pct,
                    'last_price': last_price,
                    'prev_close': prev_close
                })

        except:
            continue

    df = pd.DataFrame(losers)
    df = df.sort_values(by='change_pct')
    return df

# Filter bouncers: stocks showing recovery at open
def check_intraday_rebound(symbol):
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        price = r.json()['priceInfo']

        last_price = float(price['lastPrice'])
        prev_close = float(price['previousClose'])

        change_pct = ((last_price - prev_close) / prev_close) * 100
        return change_pct > 0.3  # at least 0.3% bounce back
    except:
        return False

# === MAIN ===
if __name__ == '__main__':
    print("📉 Scanning Nifty 100 for top losers...")
    nifty_symbols = get_nifty100_symbols()
    losers_df = simulate_top_losers(nifty_symbols)

    print(f"\nFound {len(losers_df)} losers. Checking who is bouncing now...\n")
    rebounders = []

    for i, row in losers_df.iterrows():
        if check_intraday_rebound(row['symbol']):
            rebounders.append(row['symbol'])
            print(f"✅ {row['symbol']} is rebounding!")

    print("\n🟢 Final Rebound Picks:")
    print(rebounders)
