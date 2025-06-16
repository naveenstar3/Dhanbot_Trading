import json
import pandas as pd
import pytz
import time
import sys
from datetime import datetime, timedelta
from dhan_api import get_security_id_from_trading_symbol, get_historical_price, get_live_price

print("=" * 50)
print("RULE-BASED STOCK ANALYSIS PIPELINE")
print(f"Current IST time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')}")
print("=" * 50)

# Enhanced rate limit handling
class RateLimiter:
    def __init__(self, max_retries=5, base_delay=1.5, backoff_factor=1.8):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
    
    def execute(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "Rate_Limit" in str(e):
                    delay = self.base_delay * (self.backoff_factor ** attempt)
                    print(f"⏳ Rate limited. Retry #{attempt+1} in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise e
        raise Exception(f"❌ Max retries exceeded for {func.__name__}")

# Load configuration
def load_config():
    print("\n[CONFIG] Loading configuration...")
    try:
        config_path = r"D:\Downloads\Dhanbot\dhan_autotrader\config.json"
        with open(config_path) as f:
            config = json.load(f)
        print("✓ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Config error: {e}")
        sys.exit(1)

# Get stock symbols
def get_nifty100_symbols():
    print("\n[DATA] Fetching Nifty 100 symbols...")
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
        df = pd.read_csv(url)
        symbols = df['Symbol'].tolist()
        print(f"✓ Retrieved {len(symbols)} stocks")
        return symbols
    except Exception as e:
        print(f"❌ Symbol fetch error: {e}")
        sys.exit(1)

# Format dataframe as simple text table
def dataframe_to_text(df, max_rows=10):
    """Convert DataFrame to simple text table"""
    if len(df) == 0:
        return "No data available"
    
    headers = list(df.columns)
    col_widths = [max(len(str(h)), 10) for h in headers]
    
    for _, row in df.head(max_rows).iterrows():
        for i, value in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(value)))
    
    header_row = " | ".join([str(h).ljust(w) for h, w in zip(headers, col_widths)])
    separator = "-+-".join(["-" * w for w in col_widths])
    
    data_rows = []
    for _, row in df.head(max_rows).iterrows():
        data_row = " | ".join([str(row[h]).ljust(w) for h, w in zip(headers, col_widths)])
        data_rows.append(data_row)
    
    return f"{header_row}\n{separator}\n" + "\n".join(data_rows)

# Optimized stock data processing
def get_stock_data_with_backoff(symbols):
    data = []
    rate_limiter = RateLimiter()
    print("\n[DATA] Processing stocks with rate limit handling...")
    print(f"Estimated time: {len(symbols)*1.5/60:.1f} minutes")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            # Get security ID with retry
            security_id = rate_limiter.execute(
                get_security_id_from_trading_symbol, symbol
            )
            if not security_id:
                print(f"  ⚠ {symbol}: Security ID not found")
                continue
                
            # Get current price with retry
            current_price = rate_limiter.execute(
                get_live_price, symbol, security_id
            )
            if current_price is None or current_price == 429:
                print(f"  ⚠ {symbol}: Price unavailable")
                continue
                
            # Get historical data with retry
            hist_data = rate_limiter.execute(
                get_historical_price,
                security_id,
                interval="1d",
                limit=15,
                from_date=(datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d 00:00:00"),
                to_date=datetime.now().strftime("%Y-%m-%d 23:59:59")
            )
            
            if not hist_data or len(hist_data) < 10:
                print(f"  ⚠ {symbol}: Insufficient historical data")
                continue
                
            # Calculate technical indicators
            closes = [d['close'] for d in hist_data]
            volumes = [d['volume'] for d in hist_data]
            
            # RSI calculation
            price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [max(0, change) for change in price_changes]
            losses = [max(0, -change) for change in price_changes]
            
            avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0.01
            avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.01
            
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            # Moving averages
            ma50 = sum(closes[-50:])/50 if len(closes) >= 50 else sum(closes)/len(closes)
            ma20 = sum(closes[-20:])/20 if len(closes) >= 20 else sum(closes)/len(closes)
            ma10 = sum(closes[-10:])/10 if len(closes) >= 10 else sum(closes)/len(closes)
            
            # Momentum indicators
            price_change_1d = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0
            price_change_5d = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
            
            # Volume analysis
            avg_volume = sum(volumes[-5:])/5 if len(volumes) >= 5 else volumes[-1]
            volume_change = (volumes[-1] - avg_volume) / avg_volume * 100
            
            data.append({
                "Symbol": symbol,
                "Price": current_price,
                "RSI": round(rsi, 2),
                "MA50": round(ma50, 2),
                "MA20": round(ma20, 2),
                "MA10": round(ma10, 2),
                "1D Change": round(price_change_1d, 2),
                "5D Change": round(price_change_5d, 2),
                "Volume": volumes[-1] if volumes else 0,
                "Volume Change": round(volume_change, 2)
            })
            
            if i % 10 == 0:
                print(f"  ✓ Processed {i}/{len(symbols)} stocks")
                
            # Strategic delay to prevent rate limiting
            time.sleep(1.2)
                
        except Exception as e:
            print(f"  ⚠ Critical error with {symbol}: {str(e)}")
            time.sleep(3)  # Longer pause after critical errors
            
    print(f"✓ Data collection completed for {len(data)} stocks")
    return pd.DataFrame(data)

# Advanced rule-based analysis
def rule_based_analysis(stock_df):
    print("\n[ANALYSIS] Running rule-based stock analysis...")
    
    # Calculate technical scores
    stock_df['RSI_Score'] = stock_df.apply(
        lambda x: 2 if x['RSI'] < 30 else 1 if x['RSI'] < 45 else 0 if x['RSI'] < 55 else -1 if x['RSI'] < 70 else -2, 
        axis=1
    )
    
    stock_df['Trend_Score'] = stock_df.apply(
        lambda x: 3 if x['Price'] > x['MA10'] > x['MA20'] > x['MA50'] else 
                 2 if x['Price'] > x['MA20'] > x['MA50'] else
                 1 if x['Price'] > x['MA50'] else 
                 -1 if x['Price'] < x['MA50'] else 0,
        axis=1
    )
    
    stock_df['Momentum_Score'] = stock_df.apply(
        lambda x: 2 if x['1D Change'] > 1 and x['5D Change'] > 3 else
                 1 if x['1D Change'] > 0.5 and x['5D Change'] > 1.5 else
                 0,
        axis=1
    )
    
    stock_df['Volume_Score'] = stock_df.apply(
        lambda x: 1.5 if x['Volume Change'] > 50 else 
                 1 if x['Volume Change'] > 25 else 
                 0.5 if x['Volume Change'] > 10 else 
                 0,
        axis=1
    )
    
    # Calculate total score
    stock_df['Total_Score'] = (
        stock_df['RSI_Score'] + 
        stock_df['Trend_Score'] + 
        stock_df['Momentum_Score'] + 
        stock_df['Volume_Score']
    )
    
    # Sort by highest score
    top_stocks = stock_df.sort_values('Total_Score', ascending=False).head(5)
    
    # Generate analysis report
    analysis = "Top 5 Stocks with Highest Upside Potential:\n\n"
    analysis += "Based on technical indicators:\n"
    analysis += "• RSI < 30: Oversold (Bullish)\n"
    analysis += "• Price > MA10 > MA20 > MA50: Strong Uptrend\n"
    analysis += "• Recent price momentum\n"
    analysis += "• Volume increase > 25%: Confirmation\n\n"
    
    for i, (_, row) in enumerate(top_stocks.iterrows(), 1):
        reasons = []
        
        if row['RSI'] < 35:
            reasons.append("Oversold RSI")
        elif row['RSI'] < 45:
            reasons.append("Recovering RSI")
            
        if row['Price'] > row['MA50']:
            reasons.append("Above MA50")
        if row['Price'] > row['MA20'] > row['MA50']:
            reasons.append("Strong trend")
            
        if row['1D Change'] > 0.5:
            reasons.append("Positive momentum")
            
        if row['Volume Change'] > 25:
            reasons.append("Volume surge")
            
        # Calculate potential upside
        potential = min(5, max(1, row['Total_Score'] * 0.8))
        
        analysis += (
            f"{i}. {row['Symbol']}: +{potential:.1f}% | "
            f"Price: ₹{row['Price']} | "
            f"{', '.join(reasons[:3])}\n"
        )
    
    analysis += "\nTechnical Summary:\n"
    analysis += f"- Strongest Trend: {top_stocks.iloc[0]['Symbol']} (Score: {top_stocks.iloc[0]['Total_Score']:.1f})\n"
    analysis += f"- Most Oversold: {stock_df.loc[stock_df['RSI'].idxmin()]['Symbol']} (RSI: {stock_df['RSI'].min():.1f})\n"
    analysis += f"- Best Volume Surge: {stock_df.loc[stock_df['Volume Change'].idxmax()]['Symbol']} (+{stock_df['Volume Change'].max():.1f}%)\n"
    
    return analysis

# Main execution flow
def main():
    config = load_config()
    symbols = get_nifty100_symbols()
    
    # Process stock data with enhanced rate handling
    stock_df = get_stock_data_with_backoff(symbols[:50])  # Process first 50
    
    # Save results to Excel
    excel_path = "stock_analysis_data.xlsx"
    stock_df.to_excel(excel_path, index=False)
    print(f"\n✓ Data saved to {excel_path}")
    
    # Generate analysis
    current_time_ist = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M IST')
    analysis = rule_based_analysis(stock_df)
    
    # Add timestamp to analysis
    full_analysis = f"Stock Analysis Report ({current_time_ist})\n"
    full_analysis += "=" * 50 + "\n\n"
    full_analysis += analysis
    
    # Display results
    print("\n" + "=" * 50)
    print("RULE-BASED ANALYSIS RESULTS")
    print("=" * 50)
    print(full_analysis)
    
    # Save full report
    log_path = "stock_analysis_report.txt"
    with open(log_path, "w") as f:
        f.write(full_analysis)
    print(f"\n✓ Full analysis saved to {log_path}")

if __name__ == "__main__":
    main()