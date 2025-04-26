import csv
import datetime

PORTFOLIO_LOG = "portfolio_log.csv"
GROWTH_LOG = "growth_log.csv"
CURRENT_CAPITAL_FILE = "current_capital.csv"

def log_daily_growth():
    today = datetime.datetime.now().date()
    try:
        # Check if already logged
        with open(GROWTH_LOG, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row_date = datetime.datetime.strptime(row['date'].split()[0], "%m/%d/%Y").date()
                    if row_date == today:
                        print("⏹️ Already logged for today.")
                        return
                except:
                    continue
    except FileNotFoundError:
        pass

    try:
        with open(PORTFOLIO_LOG, newline="") as f:
            reader = csv.DictReader(f)
            total_invested = 0
            total_current = 0

            for row in reader:
                try:
                    quantity = int(row["quantity"])
                    buy_price = float(row["buy_price"])
                    live_price = float(row.get("live_price", 0))

                    total_invested += buy_price * quantity
                    total_current += live_price * quantity
                except:
                    continue

        growth = total_current - total_invested
        growth_pct = (growth / total_invested) * 100 if total_invested > 0 else 0

        try:
            with open(CURRENT_CAPITAL_FILE, "r") as f:
                capital = float(f.read().strip())
        except:
            capital = 0

        now = datetime.datetime.now().strftime("%m/%d/%Y %H:%M")
        notes = "PROFIT" if growth > 0 else "LOSS"

        with open(GROWTH_LOG, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now,
                round(capital, 2),
                round(total_invested, 2),
                round(growth, 2),
                round(total_current, 2),
                notes
            ])

        print(f"📈 Growth logged: ₹{round(growth, 2)} ({round(growth_pct, 2)}%)")
    except FileNotFoundError:
        print("❌ portfolio_log.csv not found. Please run portfolio_tracker first.")


if __name__ == "__main__":
    log_daily_growth()
