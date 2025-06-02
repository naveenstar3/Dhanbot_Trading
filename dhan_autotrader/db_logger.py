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
    """
    try:
        conn = psycopg2.connect(
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config["port"]
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO bot_execution_log (timestamp, script, status, message)
            VALUES (%s, %s, %s, %s)
        """, (timestamp, script, status, message))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Logged to DB: {script} - {status}")
    except Exception as e:
        print(f"❌ DB Logging Failed: {e}")

# Optional test run
if __name__ == "__main__":
    log_to_postgres(datetime.now(), "test_logger.py", "Success", "Initial test log from db_logger.py")
