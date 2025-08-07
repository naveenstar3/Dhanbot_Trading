# Small_Cap_Mid_Cap_Stocks.py

import csv
import time
import pandas as pd
import dhan_api as dh
from datetime import datetime

MASTER_CSV = "dhan_master.csv"
OUTPUT_CSV = "small_midcap_ltps.csv"

# Try these in order until one is found in your CSV:
POSSIBLE_CAP_COLUMNS = [
    "SEM_CAP",
    "SEM_MARKET_CAP",
    "SEM_MARKET_SEGMENT",
]

def find_cap_column(df):
    for col in POSSIBLE_CAP_COLUMNS:
        if col in df.columns:
            return col
    raise KeyError(
        f"None of {POSSIBLE_CAP_COLUMNS} found in {MASTER_CSV}. "
        "Please rename your small/mid-cap column to one of these."
    )

def main():
    # Load master list
    df = pd.read_csv(MASTER_CSV, dtype=str)
    cap_col = find_cap_column(df)
    print(f"→ using cap column: {cap_col}")

    # Filter to only small-cap & mid-cap
    mask = df[cap_col].str.strip().str.lower().isin({"smallcap", "midcap"})
    sm_df = df.loc[mask].reset_index(drop=True)

    output_rows = []
    for idx, row in sm_df.iterrows():
        sym = row["SEM_TRADING_SYMBOL"].strip().upper()
        sid = row["SEM_SMST_SECURITY_ID"].strip()
        code, price = dh.get_live_price(sym, sid)
        ltp = price if code == 200 else None
        if ltp is None:
            print(f"⚠️  Failed LTP for {sym} ({sid}), code={code}")
        output_rows.append({
            "SecurityID": sid,
            "Symbol":    sym,
            "LTP":       ltp,
        })
        time.sleep(1.05)  # respect ~1 req/sec

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["SecurityID","Symbol","LTP"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} → Wrote {len(output_rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
