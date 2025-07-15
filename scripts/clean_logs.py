#!/usr/bin/env python3
"""
Clean log viewer - Filters out noise and shows only important messages
"""

import subprocess
import re
import sys

# Patterns to IGNORE (noise)
IGNORE_PATTERNS = [
    r"flash_attn is not installed",
    r"UserWarning.*torchvision",
    r"A new version of the following files",
    r"Make sure to double-check",
    r"downloading new versions",
    r"HTTP Error 429",  # Rate limiting messages
    r"Retrying in \d+s",
    r"resolve/.*/modules\.json",
    r"No sentence-transformers model found",
    r"Creating a new one with mean pooling",
    r"should have a .model_type. key",
    r"Unrecognized model in jinaai",
]

# Patterns to HIGHLIGHT (important)
HIGHLIGHT_PATTERNS = [
    (r"ERROR", "\033[91m"),      # Red
    (r"WARNING", "\033[93m"),    # Yellow
    (r"SUCCESS", "\033[92m"),    # Green
    (r"dimension.*\d+", "\033[96m"),  # Cyan for dimensions
    (r"GPU.*", "\033[95m"),      # Magenta for GPU info
]

def should_ignore(line):
    """Check if line should be filtered out"""
    for pattern in IGNORE_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False

def highlight_line(line):
    """Add color to important parts"""
    for pattern, color in HIGHLIGHT_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return f"{color}{line}\033[0m"
    return line

def main():
    print("=== Clean Log Viewer (Ctrl+C to stop) ===\n")
    
    # Follow logs with docker-compose
    cmd = ["docker-compose", "logs", "-f", "--tail", "50", "munirag"]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            line = line.strip()
            
            # Skip empty lines and noise
            if not line or should_ignore(line):
                continue
            
            # Highlight and print important lines
            print(highlight_line(line))
            
    except KeyboardInterrupt:
        print("\n\nStopped log viewer")
        process.terminate()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()