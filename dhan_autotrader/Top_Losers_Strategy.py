import pandas as pd
from nsepython import nsefetch

# Define the URL for Nifty 200 data
url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200'

# Fetch the data
data = nsefetch(url)

# Convert the data to a DataFrame
df = pd.DataFrame(data['data'])

# Ensure 'pChange' is numeric
df['pChange'] = pd.to_numeric(df['pChange'], errors='coerce')

# Sort the DataFrame to get top losers
top_losers = df.sort_values(by='pChange').head(30)

# Select relevant columns
top_losers = top_losers[['symbol', 'lastPrice', 'pChange', 'totalTradedVolume']]

# Save to CSV
top_losers.to_csv('Top_Losers_Nifty200.csv', index=False)

print("✅ Top losers from Nifty 200 have been saved to 'Top_Losers_Nifty200.csv'.")
