import shutil
import os
from datetime import datetime

def safe_read_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError(f"Corrupt or empty file: {filepath}")

        return lines
    except Exception as e:
        backup_path = filepath + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(filepath, backup_path)
        raise RuntimeError(f"⚠️ CSV Health Check Failed: {filepath}\nBackup saved at: {backup_path}\nError: {e}")
