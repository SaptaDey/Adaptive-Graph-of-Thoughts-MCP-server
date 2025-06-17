#!/usr/bin/env python3
"""Script to update all references from Adaptive Graph of Thoughts to Adaptive Graph of Thoughts"""

from pathlib import Path


def update_adaptive_graph_of_thoughts_references(file_path):
    """Update Adaptive Graph of Thoughts references in a single file"""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Track if any changes were made
        original_content = content

        # Define replacement patterns
        replacements = [
            # Main project name
            ("Adaptive Graph of Thoughts", "Adaptive Graph of Thoughts"),
            ("adaptive-graph-of-thoughts", "adaptive-graph-of-thoughts"),
            ("ADAPTIVE-GRAPH-OF-THOUGHTS", "ADAPTIVE-GRAPH-OF-THOUGHTS"),
            # GitHub references
            ("Adaptive Graph of Thoughts-2.0", "Adaptive-Graph-of-Thoughts-MCP"),
            (
                "SaptaDey/Adaptive Graph of Thoughts",
                "SaptaDey/Adaptive-Graph-of-Thoughts-MCP",
            ),
            (
                "sapta-dey/Adaptive Graph of Thoughts-2.0",
                "SaptaDey/Adaptive-Graph-of-Thoughts-MCP",
            ),
            # Documentation references
            (
                "Adaptive Graph of Thoughts Documentation",
                "Adaptive Graph of Thoughts Documentation",
            ),
            ("Adaptive Graph of Thoughts MCP", "Adaptive Graph of Thoughts MCP"),
            ("Adaptive Graph of Thoughts server", "Adaptive Graph of Thoughts server"),
            (
                "Adaptive Graph of Thoughts Development Team",
                "Adaptive Graph of Thoughts Development Team",
            ),
            # MCP specific
            (
                "Adaptive Graph of Thoughts MCP Server",
                "Adaptive Graph of Thoughts MCP Server",
            ),
            (
                "Adaptive Graph of Thoughts MCP Integration",
                "Adaptive Graph of Thoughts MCP Integration",
            ),
            # Comments and descriptions
            ("the Adaptive Graph of Thoughts", "the Adaptive Graph of Thoughts"),
            ("for Adaptive Graph of Thoughts", "for Adaptive Graph of Thoughts"),
            ("to Adaptive Graph of Thoughts", "to Adaptive Graph of Thoughts"),
            ("of Adaptive Graph of Thoughts", "of Adaptive Graph of Thoughts"),
            ("using Adaptive Graph of Thoughts", "using Adaptive Graph of Thoughts"),
            ("with Adaptive Graph of Thoughts", "with Adaptive Graph of Thoughts"),
            ("from Adaptive Graph of Thoughts", "from Adaptive Graph of Thoughts"),
            # Directory and file references that might need updating
            ("Adaptive Graph of Thoughts/", "Adaptive-Graph-of-Thoughts-MCP/"),
        ]

        # Apply replacements
        for old, new in replacements:
            content = content.replace(old, new)

        # If content changed, write it back
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to update all relevant files"""
    # Get the project root directory
    project_root = Path(".")

    print(
        f"Updating Adaptive Graph of Thoughts references in project: {project_root.absolute()}"
    )

    # Define file patterns to process
    file_patterns = [
        "**/*.md",
        "**/*.rst",
        "**/*.txt",
        "**/*.yaml",
        "**/*.yml",
        "**/*.json",
        "**/*.py",
        "**/*.html",
        "**/*.sh",
    ]

    # Find all relevant files
    files_to_process = []
    for pattern in file_patterns:
        files_to_process.extend(project_root.glob(pattern))

    # Filter out unwanted directories
    exclude_dirs = {
        "__pycache__",
        ".git",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
    }
    files_to_process = [
        f for f in files_to_process if not any(part in exclude_dirs for part in f.parts)
    ]

    updated_count = 0
    total_count = len(files_to_process)

    print(f"Found {total_count} files to process...")

    for file_path in files_to_process:
        if update_adaptive_graph_of_thoughts_references(file_path):
            updated_count += 1

    print(f"\nCompleted! Updated {updated_count} out of {total_count} files.")


if __name__ == "__main__":
    main()
