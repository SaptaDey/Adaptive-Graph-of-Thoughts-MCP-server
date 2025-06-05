#!/usr/bin/env python3
"""
This script adds type hints and ignores for common linting issues in the codebase.
Run this script after making changes to ensure consistent type checking behavior.
"""
import os
import re
import sys

# Unused import removed
# from pathlib import Path


def add_type_ignore_to_loguru_imports(filepath: str) -> None:
    """
    Adds `# type: ignore` comments to loguru import statements and logger method calls in a Python file.
    
    Reads the specified file, appends `# type: ignore` to `from loguru import logger` statements and to all `logger` method calls (`debug`, `info`, `warning`, `error`, `critical`) to suppress type checking errors, then writes the changes back to the file.
    """
    with open(filepath, encoding='utf-8') as file:
        content = file.read()

    # Add type: ignore to loguru imports
    content = re.sub(r'from loguru import logger(?!\s+#\s+type:\s+ignore)',
                    'from loguru import logger  # type: ignore',
                    content)

    # Add type: ignore comments to logger method calls
    content = re.sub(r'(logger\.(debug|info|warning|error|critical))',
                    r'\1  # type: ignore',
                    content)

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Updated {filepath}")


def find_python_files(start_dir: str):
    """Find all Python files in the given directory and its subdirectories.
    
    Returns:
        Generator[str, None, None]: A generator that yields file paths.
    """
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(root, file)


def main():
    """
    Processes all Python files in the project's `src` directory to add type ignore comments for `loguru` usage.
    
    Determines the base directory from the first command-line argument or by navigating two levels up from the script's location. If the `src` directory exists, iterates through all Python files within it and updates them to ensure consistent type ignore comments for `loguru` imports and logger method calls.
    """
    # Get the base directory of the project
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    src_dir = os.path.join(base_dir, 'src')

    if not os.path.exists(src_dir):
        print(f"Error: Source directory '{src_dir}' not found.")
        return

    # Process each Python file
    for python_file in find_python_files(src_dir):
        add_type_ignore_to_loguru_imports(python_file)


if __name__ == "__main__":
    main()
