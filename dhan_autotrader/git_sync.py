import subprocess
import datetime

def push_to_github(file_list):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"✪ Auto push: {', '.join(file_list)} @ {now}"

    try:
        print(f"🔄 Staging files: {file_list}...")
        subprocess.run(["git", "add"] + file_list, check=True)

        print(f"📝 Committing changes with message: {commit_msg}...")
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True
        )

        if "nothing to commit" in commit_result.stdout.lower():
            print("⚠️ No new changes detected. Nothing to push.")
            return

        print("🚀 Pushing to GitHub...")
        subprocess.run(["git", "push"], check=True)
        print(f"✅ Git push successful for: {file_list}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")

if __name__ == "__main__":
    push_to_github(["portfolio_log.csv", "growth_log.csv"])
