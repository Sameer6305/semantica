import subprocess
import sys

def log(msg):
    with open(r"C:\Users\Mohd Kaif\semantica\pr_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

def run_command(command):
    log(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
        log("STDOUT: " + result.stdout)
        log("STDERR: " + result.stderr)
        return result.stdout
    except Exception as e:
        log(f"Exception: {e}")
        return None

with open(r"C:\Users\Mohd Kaif\semantica\pr_log.txt", "w") as f:
    f.write("Starting PR process\n")

log("--- Pushing to origin ---")
run_command("git push origin knowledge-engineering")

log("\n--- Checking PR list ---")
pr_list = run_command("gh pr list --head knowledge-engineering")

if pr_list is not None and "knowledge-engineering" not in pr_list:
    log("\n--- Creating PR ---")
    run_command('gh pr create --title "feat: Knowledge Engineering Module Enhancements and Testing" --body "Enhancements to KG module including unit tests, conflict resolution placeholders, and documentation updates." --head knowledge-engineering --base main')
else:
    log("\n--- PR might already exist ---")
    log(f"PR List output: {pr_list}")
