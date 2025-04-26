import os
import shutil
from datetime import datetime
import csv
import requests

# Configuration
PROMPT_FOLDER = r"D:\\Downloads\\Dhanbot\\Prompts"
MAX_PROMPTS = 5

# Find the most recent existing prompt file
def get_latest_prompt_file():
    all_prompts = [file for file in os.listdir(PROMPT_FOLDER) if file.startswith("Prompt_")]
    if not all_prompts:
        raise FileNotFoundError("No existing prompt file found to update.")
    latest_prompt = max(
        all_prompts,
        key=lambda x: os.path.getmtime(os.path.join(PROMPT_FOLDER, x))
    )
    return os.path.join(PROMPT_FOLDER, latest_prompt)

# Load the most recent prompt
latest_prompt_path = get_latest_prompt_file()
with open(latest_prompt_path, "r", encoding="utf-8") as f:
    prompt_text = f.read()

# Add latest trading logic upgrade: dynamic PSU universe
psu_logic_patch = """
# 🔄 Strategy Enhancement (Updated on {today})
- Replaced hardcoded PSU stock list with dynamic PSU filter
- Now scans all stocks in Dhan's master CSV
- Selects those where:
  - SEM_COMPANY_NAME contains 'PSU'
  - Not SME (SEM_SME_FLAG != 'Y')
  - Is standard NSE_EQ stock
- Then applies same 5-min momentum logic
- Result: Safer but broader daily coverage for high-momentum PSU trades
""".format(today=datetime.now().strftime("%Y-%m-%d"))

if "# 🔄 Strategy Enhancement" not in prompt_text:
    prompt_text += "\n\n" + psu_logic_patch

# Update timestamp dynamically
current_time = datetime.now().strftime("%Y-%m-%d")
prompt_filename = f"Prompt_{current_time}.txt"
prompt_path = os.path.join(PROMPT_FOLDER, prompt_filename)

# Save updated prompt
os.makedirs(PROMPT_FOLDER, exist_ok=True)
with open(prompt_path, "w", encoding="utf-8") as f:
    f.write(prompt_text)

print(f"✅ New prompt saved: {prompt_filename}")

# Housekeeping: Keep only last 5 prompt files
all_prompts = sorted(
    [file for file in os.listdir(PROMPT_FOLDER) if file.startswith("Prompt_")],
    key=lambda x: os.path.getmtime(os.path.join(PROMPT_FOLDER, x))
)

if len(all_prompts) > MAX_PROMPTS:
    prompts_to_delete = all_prompts[:len(all_prompts) - MAX_PROMPTS]
    for old_file in prompts_to_delete:
        old_path = os.path.join(PROMPT_FOLDER, old_file)
        os.remove(old_path)
        print(f"🗑️ Deleted old prompt: {old_file}")

print("📁 Prompt update complete.")
print("🧠 Note: This was an automatic file update. LLM memory is only refreshed manually via 'Prompt Update'.")
