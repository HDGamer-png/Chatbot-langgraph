#!/usr/bin/env python3
import subprocess
import os

os.chdir(r"d:\Python\Thực tập\Thuc_tap\Chatbot AI")

# Check status
status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
print("Git Status:")
print(status.stdout)

if status.stdout.strip():
    print("\nAdding changes...")
    subprocess.run(["git", "add", "."])
    
    print("Committing...")
    result = subprocess.run(
        ["git", "commit", "-m", "Update: Verify multi-agent fixes and sanitize config"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    print("Pushing to GitHub...")
    push_result = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
    print(push_result.stdout)
    if push_result.stderr:
        print("Errors:", push_result.stderr)
else:
    print("✓ No changes to commit")
    
# Show final status
log = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
print("\nLatest commits:")
print(log.stdout)
