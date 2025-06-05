#!/usr/bin/env python3
"""
Format Python stage files under src/asr_got_reimagined/domain/stages using Black.
"""

import subprocess
import sys
import os

def format_file(file_path):
    """Format the given file in place using Black."""
    try:
        result = subprocess.run(
            ["black", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to format {file_path}\n{e.stderr}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stages_dir = os.path.join(base_dir, "src", "asr_got_reimagined", "domain", "stages")
    target_file = os.path.join(stages_dir, "stage_4_evidence.py")
    if not os.path.isfile(target_file):
        print(f"Error: {target_file} does not exist.", file=sys.stderr)
        sys.exit(1)
    format_file(target_file)