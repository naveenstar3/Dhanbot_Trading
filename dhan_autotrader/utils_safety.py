import shutil
import os
from datetime import datetime
import time
from functools import wraps

def safe_read_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Allow single-line float file (e.g., current_capital.csv)
        if len(lines) == 1 and lines[0].strip().replace(".", "", 1).isdigit():
            return lines

        # Enforce minimum 2 lines for proper CSV (header + data)
        if len(lines) < 2:
            print(f"ℹ️ CSV has only header: {filepath}")
            return lines  # Valid empty CSV with just header
        
    except Exception as e:
        print(f"⚠️ CSV load failed: {filepath} — {str(e)}")
        return []  # Skip processing, do not raise or backup
        
def retry(max_attempts=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator
