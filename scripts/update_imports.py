#!/usr/bin/env python3
"""Script to update all imports from asr_got_reimagined to adaptive_graph_of_thoughts"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    """Update imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Track if any changes were made
        original_content = content
        
        # Update all import statements
        patterns = [
            # Pattern 1: from asr_got_reimagined.* import
            (r'from asr_got_reimagined\.', 'from adaptive_graph_of_thoughts.'),
            # Pattern 2: import asr_got_reimagined.*
            (r'import asr_got_reimagined\.', 'import adaptive_graph_of_thoughts.'),
            # Pattern 3: from src.asr_got_reimagined.* import
            (r'from src\.asr_got_reimagined\.', 'from src.adaptive_graph_of_thoughts.'),
            # Pattern 4: import src.asr_got_reimagined.*
            (r'import src\.asr_got_reimagined\.', 'import src.adaptive_graph_of_thoughts.'),
            # Pattern 5: commented imports
            (r'# from asr_got_reimagined\.', '# from adaptive_graph_of_thoughts.'),
            (r'# from src\.asr_got_reimagined\.', '# from src.adaptive_graph_of_thoughts.'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to update all Python files"""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"Updating imports in project: {project_root}")
    
    # Find all Python files
    python_files = []
    for pattern in ['**/*.py']:
        python_files.extend(project_root.glob(pattern))
    
    # Filter out __pycache__ and .git directories
    python_files = [f for f in python_files if '__pycache__' not in str(f) and '.git' not in str(f)]
    
    updated_count = 0
    total_count = len(python_files)
    
    print(f"Found {total_count} Python files to process...")
    
    for file_path in python_files:
        if update_imports_in_file(file_path):
            updated_count += 1
    
    print(f"\nCompleted! Updated {updated_count} out of {total_count} files.")

if __name__ == "__main__":
    main()
