"""
This script adds missing type annotations to the stage_4_evidence.py file
"""
import re
from pathlib import Path


def add_context_update_type(file_path):
    """
    Adds missing type annotations for specific variables in a Python file.
    
    Scans the target file for assignments to `context_update` and `created_hyperedge_ids` and inserts type annotations if they are not already present.
    """
    with open(file_path, encoding='utf-8') as file:
        content = file.read()

    # Add type annotations for context_update if missing
    if 'context_update: Dict[str, Any]' not in content:
        content = re.sub(
            r'context_update\s*=\s*\{',
            'context_update: Dict[str, Any] = {',
            content
        )

    # Add type annotations for created_hyperedge_ids if missing
    if 'created_hyperedge_ids: List[str]' not in content:
        content = re.sub(
            r'created_hyperedge_ids\s*=\s*\[\]',
            'created_hyperedge_ids: List[str] = []',
            content
        )

    # Write back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Added missing type annotations to {file_path}")

if __name__ == "__main__":
    # Determine the base directory
    base_dir = Path(__file__).parent.parent

    # Path to the stage_4_evidence.py file
    evidence_file = base_dir / "src" / "asr_got_reimagined" / "domain" / "stages" / "stage_4_evidence.py"

    add_context_update_type(evidence_file)
