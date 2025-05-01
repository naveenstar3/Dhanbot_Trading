import csv
import datetime
import os
import pytz

LOG_FILE = "bot_execution_log.csv"

def log_bot_action(script, action, status, message=""):
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "script", "action", "status", "message"])
        writer.writerow([now, script, action, status, message])
