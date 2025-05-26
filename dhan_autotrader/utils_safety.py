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
            raise ValueError(f"Corrupt or empty file: {filepath}")

        return lines

    except Exception as e:
        backup_path = filepath + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(filepath, backup_path)
        raise RuntimeError(
            f"⚠️ CSV Health Check Failed: {filepath}\n"
            f"Backup saved at: {backup_path}\nError: {e}"
        )
        
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
