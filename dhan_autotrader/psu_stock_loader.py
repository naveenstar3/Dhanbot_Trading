import csv
import requests

# ✅ Load PSU stock list (cleaned from file)
def load_psu_symbols():
    with open("psu_list.txt", "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

# 🔍 Load Dhan master CSV from live site and match PSU symbols
def fetch_psu_rows_from_master():
    psu_symbols = load_psu_symbols()
    matched = {}
    unmatched = []

    try:
        response = requests.get("https://images.dhan.co/api-data/api-scrip-master.csv")
        lines = response.text.splitlines()
        reader = csv.DictReader(lines)

        for row in reader:
            row_symbol = row.get("SEM_TRADING_SYMBOL", "").strip().upper()
            instr_type = row.get("SEM_INSTRUMENT_NAME", "").strip().upper()
            exch_segment = row.get("SEM_SEGMENT", "").strip().upper()

            if row_symbol in psu_symbols:
                print(f"🔍 Found {row_symbol} in PSU list | INSTRUMENT: {instr_type} | SEGMENT: {exch_segment}")

                if instr_type == "EQUITY":
                    if row_symbol not in matched:  # Deduplicate by symbol
                        matched[row_symbol] = {
                            "symbol": row_symbol,
                            "security_id": row.get("SEM_SMST_SECURITY_ID"),
                            "lot_size": int(float(row.get("SEM_LOT_UNITS", 1)))
                        }
                else:
                    unmatched.append(row_symbol)

    except Exception as e:
        print(f"⚠️ Error fetching live master CSV: {e}")

    if unmatched:
        print("⚠️ Symbols skipped due to non-equity instrument type:", set(unmatched))

    return list(matched.values())

if __name__ == "__main__":
    psu_rows = fetch_psu_rows_from_master()
    print(f"\n✅ Found {len(psu_rows)} PSU stocks with valid data:")
    for stock in psu_rows:
        print(f"{stock['symbol']} | Security ID: {stock['security_id']} | Lot Size: {stock['lot_size']}")
