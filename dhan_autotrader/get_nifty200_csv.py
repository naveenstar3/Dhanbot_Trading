import pandas as pd
import requests
from io import StringIO, BytesIO
from datetime import datetime

url = "https://www1.nseindia.com/content/indices/ind_nifty200list.csv"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www1.nseindia.com"
}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Decode with UTF-8 and remove BOM if present
    csv_text = response.content.decode("utf-8-sig")

    # Try reading only relevant columns
    df = pd.read_csv(StringIO(csv_text), on_bad_lines='skip')  # skip bad lines if present

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Save both standard and date-tagged versions
    today_str = datetime.now().strftime("%Y%m%d")
    df.to_csv("nifty200.csv", index=False)
    df.to_csv(f"nifty200_{today_str}.csv", index=False)

    print(f"✅ Saved: nifty200.csv and nifty200_{today_str}.csv")

except Exception as e:
    print("❌ Error downloading or processing Nifty 200 list:", e)
