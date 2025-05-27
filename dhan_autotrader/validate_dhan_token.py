import json
import requests

CONFIG_PATH = "D:/Downloads/Dhanbot/dhan_autotrader/config.json"

def validate_token():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)

        token = config.get("access_token")
        client_id = config.get("client_id")

        if not token or not client_id:
            print("❌ Missing token or client_id in config.json.")
            return False

        headers = {
            "access-token": token,
            "client-id": client_id,
            "Content-Type": "application/json"
        }

        url = "https://api.dhan.co/positions"
        resp = requests.get(url, headers=headers)

        if resp.status_code == 200:
            print("✅ Token is valid and active.")
            return True
        elif resp.status_code == 401:
            print("❌ Token expired or invalid. Please update config.json.")
            return False
        else:
            print(f"⚠️ Unexpected status: {resp.status_code} - {resp.text}")
            return False

    except Exception as e:
        print(f"❌ Error during token validation: {e}")
        return False

if __name__ == "__main__":
    validate_token()
