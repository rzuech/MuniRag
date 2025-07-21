#!/usr/bin/env python3
"""Start the FastAPI server for automated testing"""

import subprocess
import time
import requests

print("Starting FastAPI server on port 8000...")

# Start the server
proc = subprocess.Popen(
    ["python", "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for it to start
print("Waiting for server to start...")
for i in range(30):  # 30 second timeout
    try:
        resp = requests.get("http://localhost:8000/health")
        if resp.status_code == 200:
            print("✓ API server is running!")
            print("\nYou can now run automated tests with:")
            print("  python automated_accuracy_test.py")
            print("\nTo stop the server, press Ctrl+C")
            break
    except:
        pass
    time.sleep(1)
    if i % 5 == 0:
        print(f"  Still waiting... ({i}s)")

try:
    # Keep running
    proc.wait()
except KeyboardInterrupt:
    print("\nStopping server...")
    proc.terminate()