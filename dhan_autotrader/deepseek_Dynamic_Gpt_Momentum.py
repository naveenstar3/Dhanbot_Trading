from itertools import islice
import pandas as pd
import numpy as np
import pytz
import json
import os
import time
import csv
import requests
import openai
from datetime import datetime, timedelta
import traceback
import sys
import io
import atexit
from utils_logger import log_bot_action
from textblob import TextBlob

# ======== LOGGING SETUP ========
log_buffer = io.StringIO()

class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
        
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
            
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = TeeLogger(sys.__stdout__, log_buffer)

def save_logs_on_exit():
    try:
        log_file_path = "D:/Downloads/Dhanbot/dhan_autotrader/Logs/Dynamic_Gpt_Momentum.txt"
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(log_buffer.getvalue())
    except Exception as e:
        print(f"⚠️ Log write failed: {e}")

atexit.register(save_logs_on_exit)

# ======== CONFIGURATION ========
now = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")
with open('D:/Downloads/Dhanbot/dhan_autotrader/config.json', 'r') as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
CLIENT_ID = config["client_id"]
OPENAI_API_KEY = config["openai_api_key"]
NEWS_API_KEY = config.get("news_api_key", "")

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json"
}

# Global counters
sentiment_call_count = 0

# ======== DATA LOADING FUNCTIONS ========
def load_dynamic_stocks():
    try:
        stocks = []
        with open('D:/Downloads/Dhanbot/dhan_autotrader/dynamic_stock_list.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row["symbol"].strip().upper()
                secid = row["security_id"].strip()
                stocks.append((symbol, secid))
        return stocks
    except Exception as e:
        print(f"⚠️ Error loading stock list: {e}")
        return []
        
STOCKS_TO_WATCH = load_dynamic_stocks()

# ======== MARKET ANALYSIS FUNCTIONS ========
def get_market_health():
    """Analyze Nifty 50 to determine overall market health"""
    try:
        nifty_data, _ = fetch_candle_data("NIFTY 50", "999920000", "INDICES", "INDEX")
        if nifty_data is None or len(nifty_data) < 2:
            return "NEUTRAL", 0.15
        
        # Calculate short-term trend
        last_close = nifty_data['Close'].iloc[-1]
        prev_close = nifty_data['Close'].iloc[-2]
        trend = "BULLISH" if last_close > prev_close else "BEARISH"
        
        # Calculate volatility (average true range)
        nifty_data['H-L'] = abs(nifty_data['High'] - nifty_data['Low'])
        nifty_data['H-PC'] = abs(nifty_data['High'] - nifty_data['Close'].shift())
        nifty_data['L-PC'] = abs(nifty_data['Low'] - nifty_data['Close'].shift())
        nifty_data['TR'] = nifty_data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = nifty_data['TR'].mean() / last_close
        
        return trend, max(0.10, min(atr, 0.30))
    except Exception as e:
        print(f"⚠️ Market health error: {e}")
        return "NEUTRAL", 0.15

def get_sector_strength():
    """Identify strongest performing sectors"""
    try:
        sectors = ["BANK", "IT", "FMCG", "AUTO", "PHARMA", "METAL", "ENERGY"]
        sector_strength = {}
        
        for sector in sectors:
            try:
                # Fetch sector index data
                sector_data, _ = fetch_candle_data(f"NIFTY {sector}", f"99992{sectors.index(sector):03d}", "INDICES", "INDEX")
                if sector_data is None or len(sector_data) < 2:
                    continue
                
                # Calculate momentum
                momentum = (sector_data['Close'].iloc[-1] - sector_data['Close'].iloc[-2]) / sector_data['Close'].iloc[-2]
                sector_strength[sector] = momentum
            except:
                continue
        
        # Return top 3 sectors
        return sorted(sector_strength, key=sector_strength.get, reverse=True)[:3]
    except Exception as e:
        print(f"⚠️ Sector strength error: {e}")
        return ["BANK", "IT", "AUTO"]  # Default sectors

# ======== DATA FETCHING FUNCTIONS ========
def fetch_candle_data(symbol, security_id=None, exchange="NSE_EQ", instrument="EQUITY"):
    """Fetch 5min and 15min candle data with robust error handling"""
    try:
        if not security_id:
            security_id = get_security_id(symbol)
        if not security_id:
            return None, None

        india = pytz.timezone("Asia/Kolkata")
        now = datetime.now(india)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        if now < market_open:
            from_dt = market_open - timedelta(days=1)
            to_dt = market_open - timedelta(days=1) + timedelta(hours=6, minutes=15)
        else:
            from_dt = market_open
            to_dt = now

        def fetch_ohlc(interval):
            payload = {
                "securityId": security_id,
                "exchangeSegment": exchange,
                "instrument": instrument,
                "interval": interval,
                "fromDate": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": to_dt.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            try:
                response = requests.post(
                    "https://api.dhan.co/v2/charts/intraday",
                    headers=HEADERS,
                    json=payload,
                    timeout=10
                )
                if response.status_code != 200:
                    return None
                data = response.json()
                return pd.DataFrame({
                    "Open": data["open"],
                    "High": data["high"],
                    "Low": data["low"],
                    "Close": data["close"],
                    "Volume": data["volume"],
                    "Timestamp": pd.to_datetime(data["timestamp"], unit='s')
                        .tz_localize('UTC')
                        .tz_convert('Asia/Kolkata')
                })
            except Exception as e:
                print(f"⚠️ Fetch error for {symbol} [{interval}]: {e}")
                return None
        
        return fetch_ohlc("5MIN"), fetch_ohlc("15MIN")
    except Exception as e:
        print(f"⚠️ Candle data error for {symbol}: {e}")
        return None, None

def get_security_id(symbol):
    """Get security ID from master CSV"""
    try:
        master_path = "D:/Downloads/Dhanbot/dhan_autotrader/dhan_master.csv"
        with open(master_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key in ["SM_SYMBOL_NAME", "SEM_CUSTOM_SYMBOL", "SEM_TRADING_SYMBOL"]:
                    if row.get(key) and row[key].strip().upper() == symbol.strip().upper():
                        return row["SEM_SMST_SECURITY_ID"]
    except Exception as e:
        print(f"⚠️ Security ID error for {symbol}: {e}")
    return None

def is_positive_sentiment(symbol):
    """
    Fetch recent news articles for the stock and evaluate average sentiment.
    Applies API call throttling and respects a 50-call rate limit.
    """
    global sentiment_call_count

    if sentiment_call_count >= 50:
        print(f"⛔ Sentiment skipped for {symbol} due to 50-request limit. Treated as neutral/pass.")
        return True

    to_date = datetime.now()
    from_date = to_date - timedelta(days=3)
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&"
        f"from={from_date_str}&"
        f"to={to_date_str}&"
        f"language=en&"
        f"sortBy=relevancy&"
        f"apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url)
        data = response.json()

        # Count only successful API hits
        sentiment_call_count += 1
        time.sleep(0.2)

        if data.get("status") != "ok":
            if data.get("code") == "rateLimited":
                print(f"⛔ Sentiment skipped for {symbol} due to 50-request limit. Treated as neutral/pass.")
                return True
            print(f"⚠️ News API error for {symbol}: {data}")
            return True        

        articles = data.get("articles", [])
        if not articles:
            print(f"⚠️ No news articles found for {symbol}. Treating as neutral.")
            return True  # Allow it if no news

        sentiments = []
        for article in articles:
            content = f"{article.get('title', '')} {article.get('description', '')}"
            blob = TextBlob(content)
            sentiments.append(blob.sentiment.polarity)

        avg_sentiment = sum(sentiments) / len(sentiments)
        print(f"📰 {symbol} - Avg Sentiment Polarity: {avg_sentiment:.2f}")
        return avg_sentiment >= 0.05

    except Exception as e:
        print(f"❌ News fetch error for {symbol}: {e}")
        return False

# ======== TECHNICAL ANALYSIS FUNCTIONS ========
def calculate_momentum_score(data_5, data_15):
    """Calculate composite momentum score using multiple timeframes"""
    if data_5 is None or data_15 is None or len(data_5) < 3 or len(data_15) < 3:
        return 0
    
    try:
        # 5-minute metrics
        price_change_5m = (data_5['Close'].iloc[-1] - data_5['Open'].iloc[-1]) / data_5['Open'].iloc[-1]
        volume_growth_5m = data_5['Volume'].iloc[-1] / data_5['Volume'].iloc[-2] if data_5['Volume'].iloc[-2] > 0 else 1
        
        # 15-minute metrics
        price_change_15m = (data_15['Close'].iloc[-1] - data_15['Open'].iloc[-1]) / data_15['Open'].iloc[-1]
        volatility = (data_15['High'].iloc[-1] - data_15['Low'].iloc[-1]) / data_15['Open'].iloc[-1]
        
        # Trend strength (last 3 candles)
        trend_strength = 1 if all(data_15['Close'].iloc[i] > data_15['Open'].iloc[i] for i in range(-3, 0)) else 0
        
        # Composite score (weighted)
        score = (
            0.4 * price_change_5m * 100 +
            0.3 * price_change_15m * 100 +
            0.2 * min(volume_growth_5m, 3) +  # Cap volume growth impact
            0.1 * volatility * 100 +
            0.5 * trend_strength
        )
        
        return max(0, min(score, 100))  # Bound between 0-100
    except Exception as e:
        print(f"⚠️ Momentum calculation error: {e}")
        return 0

def calculate_breakout_potential(data_15):
    """Calculate breakout potential based on recent price action"""
    if data_15 is None or len(data_15) < 5:
        return 0
    
    try:
        # Recent high/low
        recent_high = data_15['High'].rolling(5).max().iloc[-1]
        recent_low = data_15['Low'].rolling(5).min().iloc[-1]
        
        # Current position in range
        current_price = data_15['Close'].iloc[-1]
        position_in_range = (current_price - recent_low) / (recent_high - recent_low)
        
        # Breakout probability
        if position_in_range > 0.85:
            return min(100, 80 + (position_in_range - 0.85) * 400)  # 80-100 range
        elif position_in_range < 0.15:
            return min(100, 80 + (0.15 - position_in_range) * 400)  # 80-100 range
        return 0
    except:
        return 0

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use EMA instead of SMA for better accuracy
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Neutral 50 when no loss

# ======== GPT RANKING FUNCTION ========
def ask_gpt_to_rank_stocks(df):
    """Use GPT-4 to rank stocks based on technical criteria"""
    openai.api_key = OPENAI_API_KEY
    if df.empty:
        return []
    
    market_trend, volatility = get_market_health()
    strong_sectors = get_sector_strength()
    
    try:
        prompt = f"""
📅 Today is {now} IST. You're an intraday trading expert. Analyze these momentum opportunities:

{df[['symbol', 'momentum_score', 'breakout_potential', 'volume']].to_string(index=False)}

📌 Market Context:
- Trend: {market_trend}
- Volatility: {volatility*100:.2f}%
- Strong Sectors: {', '.join(strong_sectors)}

⚡ Selection Criteria:
1. Prioritize HIGH momentum_score (≥65) + HIGH breakout_potential (≥75)
2. Prefer stocks with volume ≥ ₹10L for liquidity
3. Favor stocks in strong sectors: {', '.join(strong_sectors)}
4. Avoid stocks with RSI ≥ 70
5. Consider market volatility: {volatility*100:.2f}%

💡 Output Format: 
- Only stock symbols in ranking order (max 10)
- Comma-separated, uppercase: RELIANCE, TCS, INFY
- If no good options, say "SKIP"
"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        gpt_response = response.choices[0].message.content.strip().upper()

        if "SKIP" in gpt_response:
            log_bot_action("GPT Ranking", "Decision", "SKIP", "No suitable stocks")
            return []

        # Extract symbols from response
        candidates = [s.strip() for s in gpt_response.split(",") if s.strip() in df["symbol"].values]
        
        if not candidates:
            log_bot_action("GPT Ranking", "Error", "FALLBACK", "No valid symbols parsed")
            return df.sort_values("momentum_score", ascending=False).head(5)["symbol"].tolist()
        
        log_bot_action("GPT Ranking", "Success", f"Selected {len(candidates)}", gpt_response)
        return candidates[:10]  # Return max 10 stocks

    except Exception as e:
        print(f"⚠️ GPT error: {str(e)[:100]}")
        log_bot_action("GPT Ranking", "Error", "FALLBACK", str(e))
        return df.sort_values("momentum_score", ascending=False).head(5)["symbol"].tolist()

# ======== MAIN SCANNER FUNCTION ========
def find_intraday_opportunities():
    """Main function to scan for intraday opportunities"""
    log_bot_action("Dynamic_Gpt_Momentum.py", "start", "MARKET_SCAN", "Starting intraday scan")
    
    # Load stock universe
    if STOCKS_TO_WATCH:
        stocks = STOCKS_TO_WATCH
        print(f"📊 Scanning {len(stocks)} stocks from dynamic list")
    else:
        # Fallback to Nifty 100 + sector leaders
        try:
            stocks = []
            with open("D:/Downloads/Dhanbot/dhan_autotrader/nifty100_constituents.csv", "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                stocks = [row[0] for row in reader if row]
            
            # Add sector leaders from strongest sectors
            strong_sectors = get_sector_strength()
            sector_leaders = {
                "BANK": ["HDFCBANK", "ICICIBANK", "KOTAKBANK"],
                "IT": ["TCS", "INFY", "HCLTECH"],
                "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND"],
                "AUTO": ["MARUTI", "M&M", "TATAMOTORS"],
                "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA"],
                "METAL": ["TATASTEEL", "HINDALCO", "VEDL"],
                "ENERGY": ["RELIANCE", "ONGC", "IOC"]
            }
            
            for sector in strong_sectors:
                if sector in sector_leaders:
                    stocks.extend(sector_leaders[sector])
            
            stocks = list(set(stocks))  # Deduplicate
            print(f"⚠️ Using fallback stock list: {len(stocks)} stocks")
        except Exception as e:
            print(f"⚠️ Stock universe error: {e}")
            stocks = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", 
                     "HINDUNILVR", "ITC", "KOTAKBANK", "SBIN", "AXISBANK"]
    
    # Get market context
    market_trend, volatility = get_market_health()
    print(f"📈 Market: {market_trend} | Volatility: {volatility*100:.2f}%")
    
    # Adaptive thresholds - more dynamic
    momentum_threshold = max(0.2, 0.4 * volatility)  # Lower threshold in volatile markets
    volume_threshold = 300000 * (1 + volatility * 1.5)  # Lower base volume threshold
    
    # Scan stocks
    opportunities = []
    for i, symbol in enumerate(stocks):
        if isinstance(symbol, tuple):  # Handle (symbol, security_id) tuples
            symbol, secid = symbol
        else:
            secid = None
            
        print(f"⏳ Processing {i+1}/{len(stocks)}: {symbol}")
        
        try:
            # Fetch data
            data_5, data_15 = fetch_candle_data(symbol, secid)
            if data_5 is None or data_15 is None or len(data_5) < 3 or len(data_15) < 3:
                continue
            
            # Calculate metrics
            momentum_score = calculate_momentum_score(data_5, data_15)
            breakout_potential = calculate_breakout_potential(data_15)
            
            # Robust volume calculation (handle pre-market cases)
            india = pytz.timezone("Asia/Kolkata")
            now = datetime.now(india)
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            
            if now < market_open:
                # Use yesterday's volume if pre-market
                total_volume = data_5['Volume'].mean() * data_5['Close'].mean()
            else:
                total_volume = data_5['Volume'].iloc[-1] * data_5['Close'].iloc[-1]
            
            # Calculate RSI
            rsi_series = calculate_rsi(data_15['Close'])
            rsi = round(rsi_series.iloc[-1], 2) if not rsi_series.empty else 0
            
            # Skip low-quality candidates with more flexible thresholds
            if total_volume < volume_threshold:
                continue
            if momentum_score < momentum_threshold and breakout_potential < 60:
                continue
            if rsi >= 75:  # Relaxed RSI threshold
                continue
            
            # Add to opportunities
            opportunities.append({
                "symbol": symbol,
                "momentum_score": momentum_score,
                "breakout_potential": breakout_potential,
                "rsi": rsi,
                "volume": total_volume,
                "last_price": data_5['Close'].iloc[-1],
                "security_id": secid or get_security_id(symbol)
            })
            
            time.sleep(0.3)  # Rate limit protection
        except Exception as e:
            print(f"⚠️ Processing error for {symbol}: {e}")
            traceback.print_exc()
    
    # Enhanced fallback mechanism
    if len(opportunities) < 10:  # More aggressive fallback
        print("⚠️ Low opportunity count. Adding top momentum performers...")
        momentum_sorted = sorted(opportunities, key=lambda x: x["momentum_score"], reverse=True)
        
        # Add top performers regardless of other filters
        for stock in momentum_sorted[:15]:  # Increase candidate pool
            if stock not in opportunities:
                opportunities.append(stock)
    
    # Convert to DataFrame for GPT
    df_opportunities = pd.DataFrame(opportunities)
    
    # Get GPT-ranked stocks
    gpt_selected = ask_gpt_to_rank_stocks(df_opportunities)
    
    # Filter to GPT-selected opportunities
    final_opportunities = [opp for opp in opportunities if opp['symbol'] in gpt_selected]
    
    # More robust fallback if GPT skipped
    if not final_opportunities:
        print("⚠️ GPT skipped selection. Using top 10 momentum stocks")
        final_opportunities = sorted(opportunities, key=lambda x: x["momentum_score"], reverse=True)[:10]
    
    # Final selection
    selected = sorted(final_opportunities, key=lambda x: (x["momentum_score"], x["breakout_potential"]), reverse=True)[:10]
    print(f"✅ Selected {len(selected)} opportunities")
    
    # Save results
    output_path = "D:/Downloads/Dhanbot/dhan_autotrader/Today_Trade_Stocks.csv"
    pd.DataFrame(selected).to_csv(output_path, index=False)
    print(f"💾 Saved results to {output_path}")
    
    log_bot_action("Dynamic_Gpt_Momentum.py", "complete", "SUCCESS", f"Found {len(selected)} opportunities")
    return selected

# ======== MARKET HOURS CHECK ========
def is_market_open():
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    if not is_market_open():
        print("🚫 Market closed. Exiting.")
        exit(0)
        
    print("\n" + "="*50)
    print(f"🚀 DYNAMIC INTRADAY MOMENTUM SCANNER - {now}")
    print("="*50)
    
    opportunities = find_intraday_opportunities()
    
    # Print top opportunities
    if opportunities:
        print("\n🔥 GPT-SELECTED OPPORTUNITIES:")
        for i, stock in enumerate(opportunities):
            print(f"{i+1}. {stock['symbol']} - Momentum: {stock['momentum_score']:.1f} | "
                  f"Breakout: {stock['breakout_potential']:.1f} | Volume: ₹{stock['volume']/100000:.1f}L")
    
    # Final log save
    save_logs_on_exit()