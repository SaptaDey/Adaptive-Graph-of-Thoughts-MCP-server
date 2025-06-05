#!/usr/bin/env python3
"""
Script to fix Python module imports by adding 'src.' prefix to 'asr_got_reimagined' imports.

Usage:
  python3 scripts/fix_imports.py [--root /path/to/project] [--src /path/to/src] [--dry-run]
"""

import os
import re
from typing import List
import argparse
from pathlib import Path

# Determine paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

parser = argparse.ArgumentParser(
    description="Fix Python imports by prefixing 'src.' to 'asr_got_reimagined'"
)
parser.add_argument(
    "--root", "-r",
    type=Path,
    default=PROJECT_ROOT,
    help="Project root directory"
)
parser.add_argument(
    "--src", "-s",
    type=Path,
    default=PROJECT_ROOT / "src",
    help="Source directory to scan"
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show changes without writing files"
)
args = parser.parse_args()
SRC_DIR = args.src

# Precompiled pattern for fixing imports
IMPORT_PATTERN = re.compile(r"(from|import)\s+(?!src\.)asr_got_reimagined")

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files recursively in a directory using pathlib."""
    return list(directory.rglob("*.py"))

def fix_imports_in_file(file_path: Path) -> int:
    """
    Adds a 'src.' prefix to 'asr_got_reimagined' import statements in a Python file.

    Scans the specified file for import statements that reference 'asr_got_reimagined' without the 'src.' prefix and updates them in place. Returns the number of import statements modified.
    """
    content = file_path.read_text(encoding="utf-8")
    new_content, num_replacements = IMPORT_PATTERN.subn(r"\1 src.asr_got_reimagined", content)
    if num_replacements > 0:
        print(f"Fixing {num_replacements} imports in {file_path}")
        if not args.dry_run:
            file_path.write_text(new_content, encoding="utf-8")
    return num_replacements

def main() -> None:
    """
    Scans all Python files in the source directory and updates import statements to include the 'src.' prefix for 'asr_got_reimagined' imports.

    After processing, prints the total number of import statements fixed across all files.
    """
    python_files = find_python_files(SRC_DIR)
    total_fixes = 0

    for file_path in python_files:
        total_fixes += fix_imports_in_file(file_path)

    print(f"Fixed {total_fixes} imports across all Python files.")

if __name__ == "__main__":
    main()