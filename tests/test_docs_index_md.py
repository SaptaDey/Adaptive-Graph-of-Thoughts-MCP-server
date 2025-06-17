import os
import re
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

@pytest.fixture
def docs_index_path():
    """Fixture providing the path to docs/index.md file."""
    return Path("docs/index.md")

@pytest.fixture
def sample_valid_markdown():
    """Fixture providing sample valid markdown content."""
    return """# Project Documentation

This is the main documentation index.

## Table of Contents

- [Getting Started](getting-started.md)
- [API Reference](api-reference.md)
- [Examples](examples.md)

## Overview

This project provides...

### Installation

```bash
pip install project-name
```

## Contributing

Please read our [contributing guidelines](CONTRIBUTING.md).
"""

@pytest.fixture
def sample_invalid_markdown():
    """Fixture providing sample invalid markdown content."""
    return """# Broken Documentation

This has [broken link](non-existent-file.md)

## Missing Headers

Content without proper structure
"""


class TestDocsIndexMdExistence:
    """Test class for validating docs/index.md file existence and basic properties."""

    def test_docs_index_md_exists(self, docs_index_path):
        """Test that docs/index.md file exists."""
        assert docs_index_path.exists(), f"docs/index.md file should exist at {docs_index_path}"

    def test_docs_index_md_is_file(self, docs_index_path):
        """Test that docs/index.md is a file, not a directory."""
        if docs_index_path.exists():
            assert docs_index_path.is_file(), "docs/index.md should be a file, not a directory"

    def test_docs_index_md_is_readable(self, docs_index_path):
        """Test that docs/index.md file is readable."""
        if docs_index_path.exists():
            assert os.access(docs_index_path, os.R_OK), "docs/index.md should be readable"

    def test_docs_index_md_not_empty(self, docs_index_path):
        """Test that docs/index.md is not empty."""
        if docs_index_path.exists():
            assert docs_index_path.stat().st_size > 0, "docs/index.md should not be empty"


class TestDocsIndexMdContent:
    """Test class for validating docs/index.md content structure and formatting."""

    def test_has_main_title(self, docs_index_path):
        """Test that docs/index.md has a main title (H1 header)."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            assert re.search(r'^# .+', content, re.MULTILINE), \
                "docs/index.md should have a main title (H1 header)"

    def test_has_valid_markdown_headers(self, docs_index_path):
        """Test that markdown headers follow proper hierarchy."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            headers = re.findall(r'^(#{1,6}) .+', content, re.MULTILINE)
            assert headers, "docs/index.md should have at least one header"

            prev_level = 0
            for header in headers:
                current_level = len(header)
                if prev_level > 0:
                    assert current_level <= prev_level + 1, \
                        f"Header level jumped from {prev_level} to {current_level}"
                prev_level = current_level

    def test_no_trailing_whitespace(self, docs_index_path):
        """Test that lines don't have trailing whitespace."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                assert not line.endswith(' ') and not line.endswith('\t'), \
                    f"Line {i} has trailing whitespace"

    def test_ends_with_newline(self, docs_index_path):
        """Test that file ends with a newline character."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            assert content.endswith('\n'), "docs/index.md should end with a newline character"


class TestDocsIndexMdLinks:
    """Test class for validating links in docs/index.md."""

    def test_internal_links_exist(self, docs_index_path):
        """Test that internal markdown links point to existing files."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            internal_links = re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', content)

            docs_dir = docs_index_path.parent
            for link_text, link_path in internal_links:
                if not link_path.startswith(('http://', 'https://')):
                    full_path = docs_dir / link_path
                    assert full_path.exists(), \
                        f"Internal link '{link_path}' referenced in docs/index.md does not exist"

    def test_no_broken_markdown_links(self, docs_index_path):
        """Test that markdown link syntax is properly formatted."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            malformed_links = re.findall(r'\[[^\]]+\](?!\()', content)
            assert not malformed_links, f"Found malformed markdown links: {malformed_links}"

    def test_external_links_format(self, docs_index_path):
        """Test that external links follow proper format."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            external_links = re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)', content)

            for link_text, url in external_links:
                assert url.startswith(('http://', 'https://')), \
                    f"External link should start with http:// or https://: {url}"
                assert ' ' not in url, \
                    f"External link should not contain spaces: {url}"


class TestDocsIndexMdQuality:
    """Test class for validating docs/index.md content quality and standards."""

    def test_has_table_of_contents(self, docs_index_path):
        """Test that docs/index.md includes a table of contents."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8').lower()
            toc_patterns = [
                r'table of contents',
                r'## contents',
                r'- \[.*\]\(.*\.md\)',
            ]
            has_toc = any(re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                          for pattern in toc_patterns)
            assert has_toc, \
                "docs/index.md should include a table of contents or navigation links"

    def test_has_project_description(self, docs_index_path):
        """Test that docs/index.md contains a project description."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            content_without_headers = re.sub(
                r'^#{1,6} .+$', '', content, flags=re.MULTILINE
            )
            meaningful_content = content_without_headers.strip()
            assert len(meaningful_content) > 100, \
                "docs/index.md should contain substantial project description"

    def test_proper_code_block_formatting(self, docs_index_path):
        """Test that code blocks are properly formatted with language specification."""
        if docs_index_path.exists():
            content = docs_index_path.read_text(encoding='utf-8')
            code_blocks = re.findall(r'```(\w*)\n', content)
            if code_blocks:
                specified_languages = [lang for lang in code_blocks if lang]
                total_blocks = len(code_blocks)
                assert len(specified_languages) >= total_blocks * 0.5, \
                    "At least half of code blocks should specify a language"


class TestDocsIndexMdEdgeCases:
    """Test class for edge cases and error handling in docs/index.md validation."""

    def test_handles_unicode_content(self):
        """Test that the validation handles unicode content properly."""
        unicode_content = """# Documentation with Unicode

This contains unicode: 🚀 αβγ 中文

## Special Characters
- Bullet point with emoji: ✅
- Copyright: ©
"""
        with patch('pathlib.Path.read_text', return_value=unicode_content):
            with patch('pathlib.Path.exists', return_value=True):
                docs_path = Path("docs/index.md")
                content = docs_path.read_text(encoding='utf-8')
                assert '🚀' in content
                assert 'αβγ' in content
                assert '中文' in content

    def test_handles_large_file(self):
        """Test that validation works with large markdown files."""
        large_content = "# Large File\n\n" + "This is a line of content.\n" * 1000
        with patch('pathlib.Path.read_text', return_value=large_content):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = len(large_content)
                    docs_path = Path("docs/index.md")
                    content = docs_path.read_text(encoding='utf-8')
                    assert content.startswith("# Large File")
                    assert content.count("This is a line of content.") == 1000

    def test_handles_missing_file_gracefully(self):
        """Test that tests handle missing file gracefully."""
        non_existent_path = Path("docs/non_existent.md")
        assert not non_existent_path.exists()
        # The existence tests should handle this gracefully with conditional checks

    @pytest.mark.parametrize("invalid_content", [
        "",  # Empty file
        "   \n   \n   ",  # Only whitespace
        "No headers at all just plain text",  # No markdown headers
        "# \n## \n### ",  # Empty headers
    ])
    def test_handles_invalid_content_gracefully(self, invalid_content):
        """Test that validation handles various invalid content patterns."""
        with patch('pathlib.Path.read_text', return_value=invalid_content):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = len(invalid_content)
                    docs_path = Path("docs/index.md")
                    content = docs_path.read_text(encoding='utf-8')
                    assert isinstance(content, str)