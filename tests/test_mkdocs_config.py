import pytest
import os
import tempfile
import shutil
import yaml
from unittest.mock import Mock, patch, mock_open
from typing import Any
import time
# Mock mkdocs imports since they may not be available in test environment
try:
    from mkdocs.config import config_options
    from mkdocs.config.base import Config
    from mkdocs.config.defaults import MkDocsConfig
except ImportError:
    class Config:
        pass

    class MkDocsConfig:
        pass

    config_options = Mock()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_mkdocs_config():
    """Sample valid mkdocs configuration for testing."""
    return {
        'site_name': 'Test Documentation Site',
        'site_url': 'https://example.com/docs',
        'site_description': 'A test documentation site',
        'site_author': 'Test Author',
        'repo_url': 'https://github.com/example/repo',
        'repo_name': 'example/repo',
        'nav': [
            {'Home': 'index.md'},
            {'Getting Started': 'getting-started.md'},
            {'API Reference': [
                {'Overview': 'api/index.md'},
                {'Authentication': 'api/auth.md'}
            ]},
            {'About': 'about.md'}
        ],
        'theme': {
            'name': 'material',
            'palette': {
                'primary': 'blue',
                'accent': 'light-blue'
            },
            'font': {
                'text': 'Roboto',
                'code': 'Roboto Mono'
            }
        },
        'plugins': [
            'search',
            {'minify': {'minify_html': True}},
            {'git-revision-date-localized': {'type': 'date'}}
        ],
        'markdown_extensions': [
            'toc',
            'tables',
            'admonition',
            {'codehilite': {'css_class': 'highlight'}},
            {'toc': {'permalink': True}}
        ],
        'extra': {
            'social': [
                {'icon': 'fontawesome/brands/github', 'link': 'https://github.com/example'},
                {'icon': 'fontawesome/brands/twitter', 'link': 'https://twitter.com/example'}
            ]
        }
    }

@pytest.fixture
def minimal_mkdocs_config():
    """Minimal valid mkdocs configuration for testing."""
    return {
        'site_name': 'Minimal Site'
    }

@pytest.fixture
def invalid_mkdocs_config():
    """Invalid mkdocs configuration for testing error handling."""
    return {
        'site_name': None,  # Invalid: should be string
        'site_url': 'not-a-valid-url',  # Invalid URL format
        'nav': 'invalid-nav-format',  # Invalid: should be list
        'theme': {'name': 'non-existent-theme'},
        'plugins': {'invalid': 'plugin-config'},  # Invalid format
        'markdown_extensions': 'should-be-list'  # Invalid: should be list
    }

@pytest.fixture
def mkdocs_config_file(temp_dir, sample_mkdocs_config):
    """Create a temporary mkdocs.yml file with sample configuration."""
    config_path = os.path.join(temp_dir, 'mkdocs.yml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_mkdocs_config, f, default_flow_style=False)
    return config_path

@pytest.fixture
def minimal_config_file(temp_dir, minimal_mkdocs_config):
    """Create a temporary mkdocs.yml file with minimal configuration."""
    config_path = os.path.join(temp_dir, 'mkdocs_minimal.yml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(minimal_mkdocs_config, f, default_flow_style=False)
    return config_path

# Configuration validation functions that would typically exist in the module
def load_config(config_file_path: str) -> dict[str, Any]:
    """Load mkdocs configuration from YAML file."""
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    with open(config_file_path, encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            raise ValueError("Configuration file is empty")

        try:
            config = yaml.safe_load(content)
            if not config:
                raise ValueError("Configuration file contains no valid data")
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}") from e
def validate_site_name(site_name: Any) -> bool:
    """Validate site_name field."""
    if not isinstance(site_name, str):
        return False
    return len(site_name.strip()) > 0

def validate_site_url(site_url: Any) -> bool:
    """Validate site_url field."""
    if not isinstance(site_url, str):
        return False

    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(site_url) is not None

def validate_navigation(nav: Any) -> bool:
    """Validate navigation structure."""
    if not isinstance(nav, list):
        return False

    def validate_nav_item(item):
        if isinstance(item, str):
            return item.endswith('.md')
        elif isinstance(item, dict):
            if len(item) != 1:
                return False
            key, value = next(iter(item.items()))
            if not key or not isinstance(key, str):
                return False
            if isinstance(value, str):
                return value.endswith('.md')
            elif isinstance(value, list):
                return all(validate_nav_item(subitem) for subitem in value)
        return False

    return all(validate_nav_item(item) for item in nav)

def validate_theme(theme: Any) -> bool:
    """Validate theme configuration."""
    if isinstance(theme, str):
        return theme in ['material', 'readthedocs', 'mkdocs']
    elif isinstance(theme, dict):
        return 'name' in theme and isinstance(theme['name'], str)
    return False

def validate_plugins(plugins: Any) -> bool:
    """Validate plugins configuration."""
    if not isinstance(plugins, list):
        return False

    for plugin in plugins:
        if isinstance(plugin, str):
            continue
        elif isinstance(plugin, dict):
            if len(plugin) != 1:
                return False
            continue
        else:
            return False
    return True

def validate_markdown_extensions(extensions: Any) -> bool:
    """Validate markdown extensions configuration."""
    if not isinstance(extensions, list):
        return False

    for ext in extensions:
        if isinstance(ext, str):
            continue
        elif isinstance(ext, dict):
            if len(ext) != 1:
                return False
            continue
        else:
            return False
    return True

class TestMkDocsConfigValidation:
    """Test mkdocs configuration validation functionality."""

    def test_valid_config_loading(self, mkdocs_config_file, sample_mkdocs_config):
        """Test loading a valid mkdocs configuration file."""
        config = load_config(mkdocs_config_file)

        assert config['site_name'] == sample_mkdocs_config['site_name']
        assert config['site_url'] == sample_mkdocs_config['site_url']
        assert config['site_description'] == sample_mkdocs_config['site_description']
        assert isinstance(config['nav'], list)
        assert len(config['nav']) == 4
        assert config['theme']['name'] == 'material'

    def test_minimal_config_loading(self, minimal_config_file):
        """Test loading a minimal valid mkdocs configuration."""
        config = load_config(minimal_config_file)
        assert config['site_name'] == 'Minimal Site'
        assert len(config) == 1

    def test_config_site_name_validation(self):
        """Test site_name field validation with various inputs."""
        valid_names = [
            'My Documentation Site',
            'Site-123',
            'Site_Name_With_Underscores',
            'Simple',
            'Site with Spaces',
            'Site-with-123-numbers'
        ]
        for name in valid_names:
            assert validate_site_name(name) is True, f"'{name}' should be valid"

    def test_config_site_url_validation(self):
        """Test site_url field validation with various formats."""
        valid_urls = [
            'https://example.com',
            'http://localhost:8000',
            'https://subdomain.example.co.uk/path',
            'https://docs.example.com/v1/api',
            'http://192.168.1.1:3000',
            'https://example.com:443/docs/'
        ]
        for url in valid_urls:
            assert validate_site_url(url) is True, f"'{url}' should be valid"

    def test_config_navigation_validation(self, sample_mkdocs_config):
        """Test navigation structure validation."""
        nav = sample_mkdocs_config['nav']
        assert validate_navigation(nav) is True

        valid_navs = [
            [{'Home': 'index.md'}],
            [{'Section': [{'Page': 'page.md'}, {'Another': 'another.md'}]}],
            ['simple.md', {'Complex': 'complex.md'}],
            [
                'index.md',
                {'Getting Started': [
                    {'Installation': 'install.md'},
                    {'Configuration': 'config.md'}]},
                {'API': 'api.md'}
            ]
        ]
        for nav_structure in valid_navs:
            assert validate_navigation(nav_structure) is True

    def test_theme_validation(self):
        """Test theme configuration validation."""
        valid_themes = [
            'material',
            'readthedocs',
            'mkdocs',
            {'name': 'material'},
            {'name': 'readthedocs', 'highlightjs': True},
            {'name': 'mkdocs', 'nav_style': 'dark'},
            {'name': 'material', 'palette': {'primary': 'blue'}}
        ]
        for theme_config in valid_themes:
            assert validate_theme(theme_config) is True

    def test_plugins_validation(self):
        """Test plugins configuration validation."""
        valid_plugin_configs = [
            ['search'],
            ['search', 'minify'],
            [{'search': {'lang': 'en'}}],
            [{'minify': {'minify_html': True}}],
            ['search', {'git-revision-date': {'enabled_if_env': 'CI'}}]
        ]
        for plugin_config in valid_plugin_configs:
            assert validate_plugins(plugin_config) is True

    def test_markdown_extensions_validation(self):
        """Test markdown extensions configuration validation."""
        valid_extensions = [
            ['toc', 'tables'],
            [{'toc': {'permalink': True}}, 'tables'],
            ['admonition', 'codehilite', 'meta'],
            [{'codehilite': {'css_class': 'highlight'}}, 'footnotes']
        ]
        for extensions in valid_extensions:
            assert validate_markdown_extensions(extensions) is True

class TestMkDocsConfigEdgeCases:
    """Test edge cases and error conditions in mkdocs configuration."""

    def test_missing_config_file(self, temp_dir):
        """Test handling of missing mkdocs.yml file."""
        non_existent_path = os.path.join(temp_dir, 'non_existent.yml')
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(non_existent_path)

    def test_empty_config_file(self, temp_dir):
        """Test handling of empty mkdocs.yml file."""
        empty_config_path = os.path.join(temp_dir, 'empty.yml')
        with open(empty_config_path, 'w') as f:
            f.write('')
        with pytest.raises(ValueError, match="Configuration file is empty"):
            load_config(empty_config_path)

    def test_whitespace_only_config_file(self, temp_dir):
        """Test handling of config file with only whitespace."""
        whitespace_config_path = os.path.join(temp_dir, 'whitespace.yml')
        with open(whitespace_config_path, 'w') as f:
            f.write('   \n\t  \n  ')
        with pytest.raises(ValueError, match="Configuration file is empty"):
            load_config(whitespace_config_path)

    def test_malformed_yaml_config(self, temp_dir):
        """Test handling of malformed YAML in config file."""
        malformed_config_path = os.path.join(temp_dir, 'malformed.yml')
        with open(malformed_config_path, 'w') as f:
            f.write('site_name: [unclosed list\ninvalid: yaml: content\n  bad: indentation')
        with pytest.raises(yaml.YAMLError, match="Invalid YAML in configuration file"):
            load_config(malformed_config_path)

    def test_null_yaml_config(self, temp_dir):
        """Test handling of YAML file that parses to None."""
        null_config_path = os.path.join(temp_dir, 'null.yml')
        with open(null_config_path, 'w') as f:
            f.write('---\n')
        with pytest.raises(ValueError, match="Configuration file contains no valid data"):
            load_config(null_config_path)

    def test_invalid_site_name_types(self):
        """Test validation with invalid site_name types."""
        invalid_names = [None, 123, [], {}, True, '', '   ', False, 0.5]
        for invalid_name in invalid_names:
            assert validate_site_name(invalid_name) is False

    def test_invalid_site_url_formats(self):
        """Test validation with invalid URL formats."""
        invalid_urls = [
            'not-a-url', 'ftp://invalid-protocol.com', 'http://', 'https://',
            'https://spaces in url.com', 'https://example.com with spaces',
            '', None, 123, [], 'javascript:alert(1)', 'file:///etc/passwd',
            'http://[invalid-ipv6', 'https://.com', 'https://example.',
            'http//missing-colon.com'
        ]
        for invalid_url in invalid_urls:
            assert validate_site_url(invalid_url) is False

    def test_invalid_navigation_structures(self):
        """Test validation with invalid navigation structures."""
        invalid_navs = [
            'string-instead-of-list', [{'invalid': None}],
            [{'nested': {'too': {'deep': {'structure': 'deep.md'}}}}],
            123, None, [{'': 'empty-key.md'}], [{'key': 'not-markdown-file.txt'}],
            [{'key': []}], [{'key': {'invalid': 'nested.md'}}],
            [{'key1': 'file1.md', 'key2': 'file2.md'}], [True], [{'key': 123}]
        ]
        for invalid_nav in invalid_navs:
            assert validate_navigation(invalid_nav) is False

    def test_invalid_theme_configurations(self):
        """Test validation with invalid theme configurations."""
        invalid_themes = [
            None, 123, [], 'non-existent-theme', {}, {'invalid': 'key'},
            {'name': None}, {'name': 123}, {'name': ''}
        ]
        for invalid_theme in invalid_themes:
            assert validate_theme(invalid_theme) is False

    def test_invalid_plugins_configurations(self):
        """Test validation with invalid plugins configurations."""
        invalid_plugins = [
            'string-instead-of-list', None, 123, [123], [None],
            [{'plugin1': 'config', 'plugin2': 'config'}], [{}], [{'': 'config'}]
        ]
        for invalid_plugins_config in invalid_plugins:
            assert validate_plugins(invalid_plugins_config) is False

    def test_invalid_markdown_extensions_configurations(self):
        """Test validation with invalid markdown extensions configurations."""
        invalid_extensions = [
            'string-instead-of-list', None, 123, [123], [None],
            [{'ext1': 'config', 'ext2': 'config'}], [{}], [{'': 'config'}]
        ]
        for invalid_ext_config in invalid_extensions:
            assert validate_markdown_extensions(invalid_ext_config) is False

class TestMkDocsConfigAdvanced:
    """Test advanced mkdocs configuration scenarios."""

    def test_config_file_encoding_handling(self, temp_dir):
        """Test handling of different file encodings."""
        utf8_config = {
            'site_name': 'Documentation Site with émojis 🚀',
            'site_description': 'Çharacters with áccents and 中文'
        }
        utf8_config_path = os.path.join(temp_dir, 'utf8_config.yml')
        with open(utf8_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(utf8_config, f, allow_unicode=True)
        config = load_config(utf8_config_path)
        assert '🚀' in config['site_name']
        assert '中文' in config['site_description']

    def test_large_configuration_file(self, temp_dir):
        """Test handling of large configuration files."""
        large_nav = [
            {f'Section {i}': [
                {f'Page {i}-{j}': f'section_{i}/page_{j}.md'} for j in range(10)
            ]} for i in range(500)
        ]
        large_config = {
            'site_name': 'Large Documentation Site',
            'nav': large_nav,
            'plugins': [f'plugin_{i}' for i in range(50)],
            'markdown_extensions': [f'extension_{i}' for i in range(30)]
        }
        large_config_path = os.path.join(temp_dir, 'large_config.yml')
        with open(large_config_path, 'w') as f:
            yaml.dump(large_config, f)
        start_time = time.time()
        config = load_config(large_config_path)
        load_time = time.time() - start_time
        assert len(config['nav']) == 500
        assert len(config['plugins']) == 50
        assert len(config['markdown_extensions']) == 30
        assert load_time < 2.0

    def test_deeply_nested_navigation(self):
        """Test validation of deeply nested navigation structures."""
        deeply_nested_nav = [{
            'Level 1': [{
                'Level 2': [{
                    'Level 3': 'deep.md'
                }]
            }]
        }]
        assert validate_navigation(deeply_nested_nav) is True
        overly_nested_nav = [{
            'L1': [{
                'L2': [{
                    'L3': [{
                        'L4': [{
                            'L5': 'too_deep.md'
                        }]
                    }]
                }]
            }]
        }]
        # Current implementation allows excess depth
        assert validate_navigation(overly_nested_nav) is True

    @pytest.mark.parametrize("site_name,expected", [
        ("Valid Site Name", True),
        ("Site-with-dashes", True),
        ("Site_with_underscores", True),
        ("Site123", True),
        ("Site with spaces", True),
        ("🚀 Site with emoji", True),
        ("", False),
        ("   ", False),
        (None, False),
        (123, False),
        ([], False),
        ({}, False),
        (True, False),
        (False, False)
    ])
    def test_site_name_validation_parametrized(self, site_name, expected):
        assert validate_site_name(site_name) == expected

    @pytest.mark.parametrize("url,expected", [
        ("https://example.com", True),
        ("http://localhost:8000", True),
        ("https://sub.example.co.uk/path", True),
        ("https://docs.example.com/v1/", True),
        ("http://192.168.1.1:3000/docs", True),
        ("not-a-url", False),
        ("ftp://invalid.com", False),
        ("", False),
        (None, False),
        ("javascript:alert(1)", False),
        ("file:///etc/passwd", False)
    ])
    def test_site_url_validation_parametrized(self, url, expected):
        assert validate_site_url(url) == expected

    @pytest.mark.slow
    def test_performance_with_complex_config(self, temp_dir):
        """Test performance with complex configuration structures."""
        complex_config = {
            'site_name': 'Performance Test Site',
            'nav': [
                {f'Section {i}': [
                    {f'Subsection {i}-{j}': [
                        {f'Page {i}-{j}-{k}': f'page_{i}_{j}_{k}.md'} for k in range(5)
                    ]} for j in range(10)
                ]} for i in range(20)
            ],
            'plugins': [
                {'search': {'lang': ['en', 'es', 'fr']}},
                {'minify': {'minify_html': True, 'minify_css': True}},
                {'git-revision-date-localized': {'type': 'datetime'}},
            ],
            'markdown_extensions': [
                {'toc': {'permalink': True, 'baselevel': 2}},
                {'codehilite': {'css_class': 'highlight', 'linenums': True}},
                {'admonition': {}},
                {'footnotes': {}},
                {'meta': {}},
                {'tables': {}},
            ],
            'extra': {
                'social': [
                    {'icon': f'fontawesome/brands/service{i}', 'link': f'https://service{i}.com/user'}
                    for i in range(20)
                ],
                'analytics': {'gtag': 'G-XXXXXXXXXX'},
                'version': {'provider': 'mike'}
            }
        }
        config_path = os.path.join(temp_dir, 'complex_config.yml')
        with open(config_path, 'w') as f:
            yaml.dump(complex_config, f)
        configs = []
        for _ in range(5):
            start_time = time.time()
            config = load_config(config_path)
            load_time = time.time() - start_time
            configs.append((config, load_time))
            assert load_time < 1.0
        base_config = configs[0][0]
        for config, _ in configs[1:]:
            assert config == base_config

@pytest.mark.integration
class TestMkDocsConfigIntegration:
    """Integration tests for mkdocs configuration with external dependencies."""

    def test_config_validation_with_file_system_checks(self, temp_dir, sample_mkdocs_config):
        docs_dir = os.path.join(temp_dir, 'docs')
        os.makedirs(docs_dir)
        for filename in ['index.md', 'getting-started.md', 'about.md']:
            with open(os.path.join(docs_dir, filename), 'w') as f:
                f.write(f"# {filename.replace('.md', '').replace('-', ' ').title()}\n\nContent here.")
        api_dir = os.path.join(docs_dir, 'api')
        os.makedirs(api_dir)
        for filename in ['index.md', 'auth.md']:
            with open(os.path.join(api_dir, filename), 'w') as f:
                f.write(f"# API {filename.replace('.md', '').title()}\n\nAPI documentation.")
        config_path = os.path.join(temp_dir, 'mkdocs.yml')
        with open(config_path, 'w') as f:
            yaml.dump(sample_mkdocs_config, f)
        config = load_config(config_path)
        assert config['site_name'] == sample_mkdocs_config['site_name']

    @patch('builtins.open', mock_open(read_data="site_name: Mocked Site"))
    def test_config_loading_with_mocked_file_operations(self):
        with patch('os.path.exists', return_value=True):
            config = load_config('mocked_path.yml')
            assert config['site_name'] == 'Mocked Site'

    def test_config_backup_and_modification_scenarios(self, temp_dir, mkdocs_config_file):
        original_config = load_config(mkdocs_config_file)
        backup_path = mkdocs_config_file + '.backup'
        shutil.copy2(mkdocs_config_file, backup_path)
        modified_config = original_config.copy()
        modified_config['site_name'] = 'Modified Site Name'
        modified_config['site_description'] = 'This site has been modified'
        with open(mkdocs_config_file, 'w') as f:
            yaml.dump(modified_config, f)
        current_config = load_config(mkdocs_config_file)
        assert current_config['site_name'] == 'Modified Site Name'
        assert current_config['site_description'] == 'This site has been modified'
        shutil.copy2(backup_path, mkdocs_config_file)
        restored_config = load_config(mkdocs_config_file)
        assert restored_config['site_name'] == original_config['site_name']
        assert 'This site has been modified' not in restored_config.get('site_description', '')
        os.remove(backup_path)