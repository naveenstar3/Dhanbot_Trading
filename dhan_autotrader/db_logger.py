import psycopg2
import json
from datetime import datetime

# Load database config from config.json
with open("config.json", "r") as file:
    config = json.load(file)
db_config = config["db"]

def log_to_postgres(timestamp, script, status, message):
    """
    Logs execution data to the bot_execution_log table in PostgreSQL.
    Creates table if it does not exist.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dhan_logs",
            user="postgres",
            password="admin"
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_execution_log (
                timestamp TIMESTAMP,
                script TEXT,
                status TEXT,
                message TEXT
            )
        """)
        cur.execute("""
            INSERT INTO bot_execution_log (timestamp, script, status, message)
            VALUES (%s, %s, %s, %s)
        """, (timestamp, script, status, message))
        conn.commit()
        cur.close()
        print(f"✅ Logged to DB: {script} - {status}")
    except Exception as e:
        print(f"❌ DB Logging Failed: {e}")
    finally:
        if conn:
            conn.close()
        
def insert_live_trail_to_db(timestamp, symbol, price, change_pct):
    """
    Inserts live trail data into the live_trail_buffer table in PostgreSQL.
    Creates table if it does not exist.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dhan_logs",
            user="postgres",
            password="admin"
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS live_trail_buffer (
                timestamp TIMESTAMP,
                symbol TEXT,
                price NUMERIC,
                change_pct NUMERIC
            )
        """)
        cur.execute("""
            INSERT INTO live_trail_buffer (timestamp, symbol, price, change_pct)
            VALUES (%s, %s, %s, %s)
        """, (timestamp, symbol, price, change_pct))
        conn.commit()
        cur.close()
        print(f"🗃️ DB Logged live trail for {symbol}")
    except Exception as e:
        print(f"❌ Failed to insert into DB for {symbol}: {e}")
    finally:
        if conn:
            conn.close()


def insert_portfolio_log_to_db(trade_date, symbol, security_id, qty, buy_price, stop_pct):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="dhan_logs",
            user="postgres",
            password="admin"
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_log (
                trade_date TIMESTAMP,
                symbol TEXT,
                security_id TEXT,
                quantity INTEGER,
                buy_price NUMERIC,
                stop_pct NUMERIC
            )
        """)
        cur.execute("""
            INSERT INTO portfolio_log (trade_date, symbol, security_id, quantity, buy_price, stop_pct)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (trade_date, symbol, security_id, qty, buy_price, stop_pct))
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"⚠️ DB log failed (portfolio_log): {e}")
    finally:
        if conn:
            conn.close()


# Optional test run
if __name__ == "__main__":
    log_to_postgres(datetime.now(), "test_logger.py", "Success", "Initial test log from db_logger.py")
