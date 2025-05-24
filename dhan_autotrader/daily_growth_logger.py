import csv
import json
from datetime import datetime
from config import *
from utils_logger import log_bot_action
from utils_safety import safe_read_csv

# ✅ Load Dhan credentials
with open("dhan_config.json") as f:
    config_data = json.load(f)

PORTFOLIO_LOG = "portfolio_log.csv"
GROWTH_LOG = "growth_log.csv"
CURRENT_CAPITAL_FILE = "current_capital.csv"

# ✅ Function to estimate net profit after service charges
def estimate_net_profit(buy_price, sell_price, quantity):
    gross_profit = (sell_price - buy_price) * quantity

    brokerage_total = BROKERAGE_PER_ORDER * 2  # Buy + Sell
    gst_on_brokerage = brokerage_total * (GST_PERCENTAGE / 100)
    stt_sell = sell_price * quantity * (STT_PERCENTAGE / 100)
    exchg_txn_charge = (buy_price + sell_price) * quantity * (EXCHANGE_TXN_CHARGE_PERCENTAGE / 100)
    sebi_charge = (buy_price + sell_price) * quantity * (SEBI_CHARGE_PERCENTAGE / 100)

    total_charges = brokerage_total + gst_on_brokerage + stt_sell + exchg_txn_charge + sebi_charge + DP_CHARGE_PER_SELL

    net_profit = gross_profit - total_charges
    return net_profit

# ✅ Check if today is 1st trading day of the month
def is_first_trading_day():
    today = datetime.now()
    if today.day != 1:
        return False

    try:
        raw_lines = safe_read_csv(PORTFOLIO_LOG)
        reader = csv.DictReader(raw_lines)
        for row in reader:      
            if row.get("status", "").strip().upper() == "SOLD":
                return True
    except:
        return False

    return False

# ✅ Main function to update growth_log
def update_growth_log():
    today = datetime.now().strftime("%Y-%m-%d")
    starting_capital = 0

    try:
        raw_lines = safe_read_csv(CURRENT_CAPITAL_FILE)
        starting_capital = float(raw_lines[0].strip())
    except:
        starting_capital = float(input("Enter starting capital: "))
        with open(CURRENT_CAPITAL_FILE, "w") as f:
            f.write(str(starting_capital))

    # Read portfolio_log to find today's SELL trades
    try:
        raw_lines = safe_read_csv(PORTFOLIO_LOG)
        reader = csv.DictReader(raw_lines)
        rows = list(reader)
    except FileNotFoundError:
        print("⚠️ portfolio_log.csv not found.")
        return

    total_realized_profit = 0
    trade_found = False

    for row in rows:
        status = row.get("status", "")
        if status == "SOLD":
            symbol = row["symbol"]
            buy_price = float(row["buy_price"])
            quantity = int(row["quantity"])
            exit_price = float(row.get("exit_price", 0))

            if buy_price > 0 and quantity > 0 and exit_price > 0:
                net_profit = estimate_net_profit(buy_price, exit_price, quantity)
                total_realized_profit += net_profit
                trade_found = True
                print(f"✅ Sold {symbol}: Net Profit ₹{round(net_profit,2)} after charges")

    if not trade_found:
        log_bot_action("daily_growth_logger.py", "Growth Log Skipped", "⏹️ NO TRADE", "No SELL trade found for today")
        print("⏹️ No trades closed today. Skipping growth log update.")
        return

    # 💸 Deduct monthly subscription on first trading day
    if is_first_trading_day():
        print("💸 Deducting ₹588.82 for monthly subscription (first trading day)")
        total_realized_profit -= 588.82

    total_realized_profit = round(total_realized_profit, 2)
    capital_after_exit = starting_capital

    # Step-Up logic
    if total_realized_profit >= 5:
        capital_after_exit += total_realized_profit
        notes = "PROFIT"
    elif total_realized_profit < 0:
        notes = "LOSS"
    else:
        notes = "HOLD"

    print(f"🔵 Today's Net Realized Profit: ₹{total_realized_profit}")
    print(f"💰 Capital after today's trading: ₹{capital_after_exit}")

    # Append to growth_log.csv
    try:
        with open(GROWTH_LOG, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                today,
                starting_capital,
                starting_capital,   # deployed = full capital used
                total_realized_profit,
                capital_after_exit,
                notes
            ])
    except Exception as e:
        print(f"❌ Error updating growth_log.csv: {e}")
        return

    # Update current_capital.csv
    try:
        with open(CURRENT_CAPITAL_FILE, "w") as f:
            f.write(str(capital_after_exit))
    except Exception as e:
        print(f"❌ Error updating current_capital.csv: {e}")
        
    log_bot_action("daily_growth_logger.py", "Growth Log Update", "✅ COMPLETE", f"Profit: ₹{total_realized_profit} | Capital: ₹{capital_after_exit}")
    print("✅ Growth Log updated successfully.")

if __name__ == "__main__":
    if os.path.exists("emergency_exit.txt"):
        send_telegram_message("ℹ️ Emergency day. Skipping growth log.")
        log_bot_action("daily_growth_logger.py", "SKIPPED", "EMERGENCY EXIT", "Logging skipped due to emergency.")
    else:
        update_growth_log()
