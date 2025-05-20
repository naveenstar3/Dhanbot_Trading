import subprocess
import datetime
import glob
from utils_logger import log_bot_action


def push_to_github(file_patterns):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"✪ Auto push: {', '.join(file_patterns)} @ {now}"

    # Find all files matching the patterns
    files_to_push = []
    for pattern in file_patterns:
        files_to_push.extend(glob.glob(pattern, recursive=True))

    if not files_to_push:
        print("⚠️ No matching files found. Nothing to push.")
        return

    try:
        print(f"🔄 Staging files: {files_to_push}...")
        subprocess.run(["git", "add"] + files_to_push, check=True)

        print(f"📝 Committing changes with message: {commit_msg}...")
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True
        )

        if "nothing to commit" in commit_result.stdout.lower():
            print("⚠️ No new changes detected. Nothing to push.")
            log_bot_action("git_sync.py", "Git Push", "⚠️ SKIPPED", "No new file changes")
            return

        print("🚀 Pushing to GitHub...")
        subprocess.run(["git", "push"], check=True)
        print(f"✅ Git push successful for: {files_to_push}")
        log_bot_action("git_sync.py", "Git Push", "✅ COMPLETE", f"{len(files_to_push)} file(s) pushed")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")

if __name__ == "__main__":
    # Push all CSV and PY files
    push_to_github(["*.csv", "*.py"])
