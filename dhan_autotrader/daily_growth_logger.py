import csv
import datetime

PORTFOLIO_LOG = "portfolio_log.csv"
GROWTH_LOG = "growth_log.csv"
CURRENT_CAPITAL_FILE = "current_capital.csv"


def log_daily_growth():
    try:
        with open(PORTFOLIO_LOG, newline="") as f:
            reader = csv.DictReader(f)
            total_invested = 0
            total_current = 0
            has_data = False

            for row in reader:
                if row.get("status") != "SOLD":
                    continue
                has_data = True
                try:
                    quantity = int(row["quantity"])
                    buy_price = float(row["buy_price"])
                    exit_price = float(row["exit_price"])

                    total_invested += buy_price * quantity
                    total_current += exit_price * quantity
                except:
                    continue

        if not has_data:
            print("⏹️ No trades sold today. Skipping growth log.")
            return

        growth = total_current - total_invested
        growth_pct = (growth / total_invested) * 100 if total_invested > 0 else 0
        notes = "PROFIT" if growth > 0 else "LOSS"

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(GROWTH_LOG, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now, round(total_invested, 2), round(total_current, 2), round(growth, 2), round(growth_pct, 2), notes])

        print(f"📈 Growth logged: ₹{round(growth, 2)} ({round(growth_pct, 2)}%) [{notes}]")

        # Optional: update current capital file
        with open(CURRENT_CAPITAL_FILE, "w") as f:
            f.write(str(round(total_current, 2)))

    except FileNotFoundError:
        print("❌ portfolio_log.csv not found. Please run portfolio_tracker first.")


if __name__ == "__main__":
    log_daily_growth()
