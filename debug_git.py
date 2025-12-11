import subprocess
import os

def run_git_cmd(cmd):
    print(f"--- Running: {cmd} ---")
    try:
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error running {cmd}: {e}")

print(f"CWD: {os.getcwd()}")
run_git_cmd("git status")
run_git_cmd("git branch -v")
run_git_cmd("git remote -v")
run_git_cmd("git push origin knowledge-engineering")
