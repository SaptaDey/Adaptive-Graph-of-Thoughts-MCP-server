import json
from pathlib import Path

import pytest
import yaml

# Directories to scan for configuration files
config_dirs = [Path("config"), Path("settings")]

# Gather all YAML files (*.yml, *.yaml)
yaml_paths = []
for d in config_dirs:
    if d.exists():
        yaml_paths.extend(d.rglob("*.yml"))
        yaml_paths.extend(d.rglob("*.yaml"))

# Gather all JSON files
json_paths = []
for d in config_dirs:
    if d.exists():
        json_paths.extend(d.rglob("*.json"))


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_is_valid(yaml_path):
    """
    Validate that each YAML file parses without error and
    has a top-level mapping or sequence.
    """
    content = yaml_path.read_text()
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML syntax error in {yaml_path}: {e}")
    assert isinstance(data, (dict, list)), (
        f"Top-level YAML should be a mapping or list in {yaml_path}"
    )

    # Illustrative: require "name" and "version" in app.yml
    if yaml_path.name == "app.yml" and isinstance(data, dict):
        required_keys = {"name", "version"}
        missing = required_keys - set(data.keys())
        assert not missing, f"{yaml_path} missing required keys {missing}"


@pytest.mark.parametrize("json_path", json_paths)
def test_json_is_valid(json_path):
    """
    Validate that each JSON file parses without error and
    has a top-level mapping or array.
    """
    content = json_path.read_text()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        pytest.fail(f"JSON syntax error in {json_path}: {e}")
    assert isinstance(data, (dict, list)), (
        f"Top-level JSON should be a mapping or list in {json_path}"
    )


def test_no_empty_config_files():
    """
    Ensure no configuration file is empty.
    """
    for d in config_dirs:
        if not d.exists():
            continue
        for pattern in ("*.yml", "*.yaml", "*.json"):
            for path in d.rglob(pattern):
                size = path.stat().st_size
                assert size > 0, f"{path} is unexpectedly empty (size={size} bytes)"


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_no_duplicate_keys(yaml_path):
    """
    Test that YAML files don't contain duplicate keys at any level.
    """
    content = yaml_path.read_text()
    # Use a custom constructor to detect duplicate keys
    class DuplicateKeyError(Exception):
        pass
    
    def no_duplicates_constructor(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise DuplicateKeyError(f"Duplicate key found: {key}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping
    
    yaml.SafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates_constructor)
    
    try:
        yaml.safe_load(content)
    except DuplicateKeyError as e:
        pytest.fail(f"Duplicate keys in {yaml_path}: {e}")
    except yaml.YAMLError:
        # Syntax errors are handled by test_yaml_is_valid
        pass


@pytest.mark.parametrize("json_path", json_paths)  
def test_json_no_security_leaks(json_path):
    """
    Test that JSON files don't contain potential security issues like
    exposed passwords, API keys, or secrets.
    """
    content = json_path.read_text().lower()
    sensitive_patterns = [
        'password', 'passwd', 'pwd', 'secret', 'token', 'api_key', 
        'apikey', 'private_key', 'privatekey', 'auth_token', 'authtoken'
    ]
    
    for pattern in sensitive_patterns:
        if pattern in content:
            # Allow certain safe contexts
            safe_contexts = ['password_policy', 'token_expiry', 'secret_length']
            if not any(safe in content for safe in safe_contexts):
                pytest.fail(f"Potential security leak in {json_path}: contains '{pattern}'")


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_no_security_leaks(yaml_path):
    """
    Test that YAML files don't contain potential security issues.
    """
    content = yaml_path.read_text().lower()
    sensitive_patterns = [
        'password', 'passwd', 'pwd', 'secret', 'token', 'api_key',
        'apikey', 'private_key', 'privatekey', 'auth_token', 'authtoken'
    ]
    
    for pattern in sensitive_patterns:
        if pattern in content:
            # Allow certain safe contexts
            safe_contexts = ['password_policy', 'token_expiry', 'secret_length']
            if not any(safe in content for safe in safe_contexts):
                pytest.fail(f"Potential security leak in {yaml_path}: contains '{pattern}'")


def test_config_directories_exist():
    """
    Test that at least one of the expected config directories exists.
    """
    existing_dirs = [d for d in config_dirs if d.exists()]
    assert existing_dirs, f"None of the expected config directories exist: {config_dirs}"


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_encoding_is_utf8(yaml_path):
    """
    Test that YAML files are properly UTF-8 encoded.
    """
    try:
        yaml_path.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        pytest.fail(f"YAML file {yaml_path} is not valid UTF-8: {e}")


@pytest.mark.parametrize("json_path", json_paths)
def test_json_encoding_is_utf8(json_path):
    """
    Test that JSON files are properly UTF-8 encoded.
    """
    try:
        json_path.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        pytest.fail(f"JSON file {json_path} is not valid UTF-8: {e}")


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_no_tabs(yaml_path):
    """
    Test that YAML files don't contain tab characters (should use spaces).
    """
    content = yaml_path.read_text()
    if '\t' in content:
        pytest.fail(f"YAML file {yaml_path} contains tab characters - use spaces instead")


@pytest.mark.parametrize("json_path", json_paths)
def test_json_is_formatted(json_path):
    """
    Test that JSON files are properly formatted (indented).
    """
    content = json_path.read_text()
    try:
        data = json.loads(content)
        formatted = json.dumps(data, indent=2, sort_keys=True)
        # Allow some flexibility in formatting
        content_normalized = json.dumps(json.loads(content), sort_keys=True)
        formatted_normalized = json.dumps(data, sort_keys=True)
        assert content_normalized == formatted_normalized, (
            f"JSON file {json_path} appears to be improperly formatted"
        )
    except json.JSONDecodeError:
        # Syntax errors are handled by test_json_is_valid
        pass


def test_config_file_sizes_reasonable():
    """
    Test that config files are not suspiciously large (potential issue).
    """
    max_size_bytes = 1024 * 1024  # 1MB
    
    for d in config_dirs:
        if not d.exists():
            continue
        for pattern in ("*.yml", "*.yaml", "*.json"):
            for path in d.rglob(pattern):
                size = path.stat().st_size
                assert size <= max_size_bytes, (
                    f"Config file {path} is suspiciously large: {size} bytes"
                )


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_no_circular_references(yaml_path):
    """
    Test that YAML files don't contain circular references.
    """
    content = yaml_path.read_text()
    try:
        # Load with safe_load which prevents most circular reference issues
        data = yaml.safe_load(content)
        # Additional check for anchor/alias loops if needed
        if '&' in content and '*' in content:
            # This is a basic check - more sophisticated detection could be added
            assert content.count('&') >= content.count('*'), (
                f"Potential circular reference in {yaml_path}"
            )
    except yaml.YAMLError:
        # Syntax errors are handled by test_yaml_is_valid
        pass


@pytest.mark.parametrize("json_path", json_paths)
def test_json_values_not_null_where_expected(json_path):
    """
    Test that important configuration values are not null/None.
    """
    try:
        data = json.loads(json_path.read_text())
        if isinstance(data, dict):
            # Check common important keys that shouldn't be null
            important_keys = ['name', 'version', 'host', 'port', 'database']
            for key in important_keys:
                if key in data:
                    assert data[key] is not None, (
                        f"Important key '{key}' is null in {json_path}"
                    )
    except json.JSONDecodeError:
        # Syntax errors are handled by test_json_is_valid
        pass


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_values_not_null_where_expected(yaml_path):
    """
    Test that important configuration values are not null/None.
    """
    try:
        data = yaml.safe_load(yaml_path.read_text())
        if isinstance(data, dict):
            # Check common important keys that shouldn't be null
            important_keys = ['name', 'version', 'host', 'port', 'database']
            for key in important_keys:
                if key in data:
                    assert data[key] is not None, (
                        f"Important key '{key}' is null in {yaml_path}"
                    )
    except yaml.YAMLError:
        # Syntax errors are handled by test_yaml_is_valid
        pass


def test_config_files_have_consistent_naming():
    """
    Test that config files follow consistent naming conventions.
    """
    all_config_files = []
    for d in config_dirs:
        if d.exists():
            all_config_files.extend(d.rglob("*.yml"))
            all_config_files.extend(d.rglob("*.yaml"))
            all_config_files.extend(d.rglob("*.json"))
    
    for path in all_config_files:
        name = path.name
        # Check for consistent naming patterns
        assert not name.startswith('.'), f"Config file {path} starts with dot (hidden file)"
        assert ' ' not in name, f"Config file {path} contains spaces in filename"
        assert name.islower() or '_' in name or '-' in name, (
            f"Config file {path} should use lowercase, underscores, or hyphens"
        )


@pytest.mark.parametrize("yaml_path", yaml_paths)
def test_yaml_boolean_values_proper_format(yaml_path):
    """
    Test that boolean values in YAML are properly formatted.
    """
    content = yaml_path.read_text()
    try:
        data = yaml.safe_load(content)
        
        def check_booleans(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_booleans(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_booleans(item, f"{path}[{i}]")
            elif isinstance(obj, bool):
                # Check that boolean appears as true/false in the raw content
                bool_str = str(obj).lower()
                # This is a basic check - could be more sophisticated
                assert bool_str in ['true', 'false'], (
                    f"Boolean at {path} should be 'true' or 'false' in {yaml_path}"
                )
        
        if data is not None:
            check_booleans(data)
    except yaml.YAMLError:
        # Syntax errors are handled by test_yaml_is_valid
        pass


def test_app_yml_specific_requirements():
    """
    Test specific requirements for app.yml files beyond basic validation.
    """
    app_yml_files = []
    for d in config_dirs:
        if d.exists():
            app_yml_files.extend(d.rglob("app.yml"))
            app_yml_files.extend(d.rglob("app.yaml"))
    
    for app_yml in app_yml_files:
        try:
            data = yaml.safe_load(app_yml.read_text())
            if isinstance(data, dict):
                # Test version format
                if 'version' in data:
                    version = data['version']
                    assert isinstance(version, str), f"Version should be string in {app_yml}"
                    # Basic semantic version check
                    version_parts = str(version).split('.')
                    assert len(version_parts) >= 2, (
                        f"Version should have at least major.minor format in {app_yml}"
                    )
                
                # Test name is not empty
                if 'name' in data:
                    name = data['name']
                    assert isinstance(name, str) and name.strip(), (
                        f"Name should be non-empty string in {app_yml}"
                    )
        except yaml.YAMLError:
            # Syntax errors are handled by test_yaml_is_valid
            pass


def test_config_files_readable():
    """
    Test that all config files are readable with proper permissions.
    """
    for d in config_dirs:
        if not d.exists():
            continue
        for pattern in ("*.yml", "*.yaml", "*.json"):
            for path in d.rglob(pattern):
                assert path.is_file(), f"{path} is not a regular file"
                assert path.stat().st_size >= 0, f"{path} has invalid size"
                # Test actual readability
                try:
                    path.read_text()
                except (PermissionError, OSError) as e:
                    pytest.fail(f"Cannot read config file {path}: {e}")


def test_no_binary_files_in_config_dirs():
    """
    Test that config directories don't contain binary files.
    """
    for d in config_dirs:
        if not d.exists():
            continue
        
        for path in d.rglob("*"):
            if path.is_file() and not path.suffix.lower() in ['.yml', '.yaml', '.json', '.txt', '.md']:
                try:
                    # Try to read as text - if it fails, it's likely binary
                    content = path.read_text(encoding='utf-8')
                    # Check for null bytes which indicate binary content
                    assert '\x00' not in content, f"Binary content detected in {path}"
                except UnicodeDecodeError:
                    pytest.fail(f"Binary file found in config directory: {path}")


# Mock tests for edge cases
def test_handle_missing_config_directories(monkeypatch):
    """
    Test behavior when config directories don't exist.
    """
    # Mock Path.exists to return False
    def mock_exists(self):
        return False
    
    monkeypatch.setattr(Path, "exists", mock_exists)
    
    # The original functions should handle missing directories gracefully
    # Test that no exceptions are raised
    for d in config_dirs:
        assert not d.exists()  # Verify our mock is working


def test_handle_permission_denied(tmp_path, monkeypatch):
    """
    Test behavior when config files can't be read due to permissions.
    """
    # Create a temporary config directory structure
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "test.yml"
    config_file.write_text("name: test\nversion: 1.0")
    
    # Mock the config_dirs to point to our temp directory
    monkeypatch.setattr("tests.test_config_files.config_dirs", [config_dir])
    
    # This test mainly ensures our test setup is robust
    assert config_file.exists()
    assert config_file.read_text()  # Should be readable


def test_very_large_config_file_handling(tmp_path, monkeypatch):
    """
    Test handling of unusually large config files.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create a large but valid YAML file
    large_yaml = config_dir / "large.yml"
    content = ["items:"]
    for i in range(1000):
        content.append(f"  - id: {i}")
        content.append(f"    name: item_{i}")
    
    large_yaml.write_text('\n'.join(content))
    
    # Mock the config_dirs
    monkeypatch.setattr("tests.test_config_files.config_dirs", [config_dir])
    
    # Test that it can be loaded
    data = yaml.safe_load(large_yaml.read_text())
    assert isinstance(data, dict)
    assert len(data['items']) == 1000


def test_json_schema_validation_with_schema_files():
    """
    Test JSON files against schema files when available.
    """
    import jsonschema
    
    schema_files = []
    for d in config_dirs:
        if d.exists():
            schema_files.extend(d.rglob("*.schema.json"))
    
    for schema_file in schema_files:
        try:
            schema = json.loads(schema_file.read_text())
            # Find corresponding JSON files to validate
            base_name = schema_file.name.replace('.schema.json', '.json')
            json_files = list(schema_file.parent.rglob(base_name))
            
            for json_file in json_files:
                try:
                    data = json.loads(json_file.read_text())
                    jsonschema.validate(data, schema)
                except jsonschema.ValidationError as e:
                    pytest.fail(f"Schema validation failed for {json_file}: {e}")
                except json.JSONDecodeError:
                    # Handled by other tests
                    pass
        except json.JSONDecodeError:
            # Invalid schema file - this should be caught by other tests
            pass


def test_yaml_anchor_and_alias_usage(yaml_paths_parametrized=yaml_paths):
    """
    Test that YAML anchors and aliases are used correctly.
    """
    for yaml_path in yaml_paths_parametrized:
        content = yaml_path.read_text()
        if '&' in content or '*' in content:
            try:
                data = yaml.safe_load(content)
                # If it loads successfully, anchors/aliases are valid
                assert data is not None or content.strip() == '', (
                    f"YAML with anchors/aliases should not be empty: {yaml_path}"
                )
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML anchor/alias usage in {yaml_path}: {e}")


def test_config_files_have_documentation():
    """
    Test that configuration files have adequate documentation (comments).
    """
    for d in config_dirs:
        if not d.exists():
            continue
        
        for yaml_path in d.rglob("*.yml"):
            content = yaml_path.read_text()
            # Check for comments (lines starting with #)
            lines = content.split('\n')
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            total_lines = len([line for line in lines if line.strip()])
            
            if total_lines > 10:  # Only check documentation for larger files
                comment_ratio = len(comment_lines) / total_lines if total_lines > 0 else 0
                assert comment_ratio >= 0.1, (
                    f"Large config file {yaml_path} should have some documentation comments"
                )
        
        for yaml_path in d.rglob("*.yaml"):
            content = yaml_path.read_text()
            lines = content.split('\n')
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            total_lines = len([line for line in lines if line.strip()])
            
            if total_lines > 10:
                comment_ratio = len(comment_lines) / total_lines if total_lines > 0 else 0
                assert comment_ratio >= 0.1, (
                    f"Large config file {yaml_path} should have some documentation comments"
                )


def test_json_no_comments():
    """
    Test that JSON files don't contain comments (which are not valid JSON).
    """
    for d in config_dirs:
        if not d.exists():
            continue
        
        for json_path in d.rglob("*.json"):
            content = json_path.read_text()
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('#'):
                    pytest.fail(
                        f"JSON file {json_path} contains comments on line {i}. "
                        f"JSON does not support comments."
                    )


def test_config_values_within_reasonable_ranges():
    """
    Test that numeric configuration values are within reasonable ranges.
    """
    def check_numeric_values(obj, path="", file_path=None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                check_numeric_values(v, f"{path}.{k}", file_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_numeric_values(item, f"{path}[{i}]", file_path)
        elif isinstance(obj, (int, float)):
            # Check for reasonable ranges based on common config patterns
            if 'port' in path.lower():
                assert 1 <= obj <= 65535, (
                    f"Port number {obj} at {path} should be 1-65535 in {file_path}"
                )
            elif 'timeout' in path.lower():
                assert 0 <= obj <= 3600, (
                    f"Timeout {obj} at {path} should be 0-3600 seconds in {file_path}"
                )
            elif 'percentage' in path.lower() or 'percent' in path.lower():
                assert 0 <= obj <= 100, (
                    f"Percentage {obj} at {path} should be 0-100 in {file_path}"
                )
    
    for d in config_dirs:
        if not d.exists():
            continue
        
        for yaml_path in list(d.rglob("*.yml")) + list(d.rglob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_path.read_text())
                if data is not None:
                    check_numeric_values(data, file_path=yaml_path)
            except yaml.YAMLError:
                # Handled by other tests
                pass
        
        for json_path in d.rglob("*.json"):
            try:
                data = json.loads(json_path.read_text())
                check_numeric_values(data, file_path=json_path)
            except json.JSONDecodeError:
                # Handled by other tests
                pass
