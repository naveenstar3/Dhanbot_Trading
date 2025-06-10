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
            host=db_config["host"],
            database=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            port=db_config.get("port", 5432)
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
        
def insert_live_trail_to_db(timestamp, symbol, price, change_pct, order_id=None):
    """
    Inserts live trail data into the live_trail_buffer table in PostgreSQL.
    Creates table if it does not exist.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            database=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            port=db_config.get("port", 5432)
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS live_trail_buffer (
                timestamp TIMESTAMP,
                symbol TEXT,
                price NUMERIC,
                change_pct NUMERIC,
                order_id TEXT
            )
        """)
        cur.execute("""
            INSERT INTO live_trail_buffer (timestamp, symbol, price, change_pct, order_id)
            VALUES (%s, %s, %s, %s, %s)
        """, (timestamp, symbol, price, change_pct, order_id))        
        conn.commit()
        cur.close()
        print(f"🗃️ DB Logged live trail for {symbol}")
    except Exception as e:
        print(f"❌ Failed to insert into DB for {symbol}: {e}")
    finally:
        if conn:
            conn.close()

def insert_portfolio_log_to_db(trade_date, symbol, security_id, qty, buy_price, stop_pct, order_id=None):
    conn = None
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            database=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            port=db_config.get("port", 5432)
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_log (
                trade_date TIMESTAMP,
                symbol TEXT,
                security_id TEXT,
                quantity INTEGER,
                buy_price NUMERIC,
                stop_pct NUMERIC,
                order_id TEXT
            )
        """)
        cur.execute("""
            INSERT INTO portfolio_log (trade_date, symbol, security_id, quantity, buy_price, stop_pct, order_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (trade_date, symbol, security_id, qty, buy_price, stop_pct, order_id))        
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"⚠️ DB log failed (portfolio_log): {e}")
    finally:
        if conn:
            conn.close()
            
def update_portfolio_log_to_db(trade_date, symbol, security_id, quantity, buy_price, stop_pct, exit_price=None, live_price=None, status="SOLD"):
    conn = None
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            database=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            port=db_config.get("port", 5432)
        )
        cur = conn.cursor()

        # Ensure table exists with the new fields
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_log (
                trade_date TIMESTAMP,
                symbol TEXT,
                security_id TEXT,
                quantity INTEGER,
                buy_price NUMERIC,
                stop_pct NUMERIC,
                exit_price NUMERIC,
                live_price NUMERIC,
                last_checked TIMESTAMP,
                status TEXT
            )
        """)

        cur.execute("""
            UPDATE portfolio_log
            SET
                quantity = %s,
                buy_price = %s,
                stop_pct = %s,
                exit_price = %s,
                live_price = %s,
                last_checked = %s,
                status = %s
            WHERE symbol = %s AND trade_date = %s
        """, (
            quantity,
            buy_price,
            stop_pct,
            exit_price,
            live_price,
            datetime.now(),
            status,
            symbol,
            trade_date
        ))

        conn.commit()
        cur.close()
        print(f"🗃️ DB updated for {symbol} on {trade_date}")
    except Exception as e:
        print(f"❌ Failed to update portfolio_log DB for {symbol}: {e}")
    finally:
        if conn:
            conn.close()




# Optional test run
if __name__ == "__main__":
    log_to_postgres(datetime.now(), "test_logger.py", "Success", "Initial test log from db_logger.py")
