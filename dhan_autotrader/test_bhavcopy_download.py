import os
import requests

def fetch_nse_bhavcopy_csv():
    url = "https://www1.nseindia.com/content/nsccl/bulk.csv"
    save_path = "D:/Downloads/Dhanbot/nse_bhav/bhavcopy_latest.csv"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    }

    try:
        print(f"🌐 Downloading Bhavcopy from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"✅ Bhavcopy saved to: {save_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    fetch_nse_bhavcopy_csv()
