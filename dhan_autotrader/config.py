# config.py

# 🛠️ TEST MODE Toggle
TEST_MODE = False  # Set to False for real production trading

# 📊 Dhan Trading Configuration
EXCHANGE_SEGMENT = "NSE_EQ"  # Always "NSE_EQ" for Equity Cash
PRODUCT_TYPE = "CNC"         # Delivery based trading
ORDER_TYPE = "MARKET"
VALIDITY = "DAY"

# 💰 Brokerage and Charges Settings
BROKERAGE_PER_ORDER = 0  # Assuming Dhan is free for CNC, otherwise Rs 20
DP_CHARGE_PER_SELL = 15  # Approx Rs 13–20 per sell from DP
GST_PERCENTAGE = 18      # 18% GST on brokerage
STT_PERCENTAGE = 0.1     # 0.1% STT on Sell side
EXCHANGE_TXN_CHARGE_PERCENTAGE = 0.00345  # Exchange charges
SEBI_CHARGE_PERCENTAGE = 0.0001           # SEBI charges

# 📈 Profit Target
MINIMUM_NET_PROFIT_REQUIRED = 5  # ₹5 net minimum profit after all charges

# 📂 File Paths (can be customized)
PORTFOLIO_LOG = "portfolio_log.csv"
SELL_LOG = "sell_log.csv"
GROWTH_LOG = "growth_log.csv"
CURRENT_CAPITAL_FILE = "current_capital.csv"
LIVE_LOG = "live_prices_log.csv"
PSU_MASTER_FILE = "psu_list.txt"

# 🔗 Dhan API URLs
BASE_URL = "https://api.dhan.co/orders"
TRADE_BOOK_URL = "https://api.dhan.co/trade-book"
