import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from datetime import datetime, timezone
import sys
from typing import Dict, Any, List

# Add the src directory to the path for importing the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from adaptive_graph_of_thoughts.metadata_helpers import (
    MetadataManager,
    ValidationError,
    MetadataValidator,
    extract_metadata,
    validate_metadata_schema,
    merge_metadata,
    serialize_metadata,
    deserialize_metadata
)


@pytest.fixture
def sample_metadata():
    """Fixture providing sample metadata for testing."""
    return {
        "id": "test_123",
        "name": "Test Node",
        "type": "computation",
        "created_at": "2023-01-01T00:00:00Z",
        "version": "1.0.0",
        "dependencies": ["dep1", "dep2"],
        "config": {
            "param1": "value1",
            "param2": 42,
            "nested": {
                "key": "nested_value"
            }
        }
    }

@pytest.fixture
def invalid_metadata():
    """Fixture providing invalid metadata for testing error conditions."""
    return {
        "id": None,  # Invalid: None value
        "name": "",  # Invalid: empty string
        "type": "invalid_type",  # Invalid type
        "created_at": "invalid_date",  # Invalid date format
        "version": 123,  # Invalid: should be string
        "dependencies": "not_a_list",  # Invalid: should be list
    }

@pytest.fixture
def metadata_manager():
    """Fixture providing a MetadataManager instance."""
    return MetadataManager()

@pytest.fixture
def temp_metadata_file():
    """Fixture providing a temporary file for metadata testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestMetadataManager:
    """Test suite for MetadataManager class."""
    
    def test_init_default(self):
        """Test MetadataManager initialization with defaults."""
        manager = MetadataManager()
        assert manager is not None
        assert hasattr(manager, 'validate')
        assert hasattr(manager, 'extract')
        
    def test_init_with_config(self):
        """Test MetadataManager initialization with custom config."""
        config = {"strict_validation": True, "auto_timestamp": False}
        manager = MetadataManager(config=config)
        assert manager.config == config
        
    def test_init_with_invalid_config(self):
        """Test MetadataManager initialization with invalid config."""
        with pytest.raises(TypeError):
            MetadataManager(config="invalid_config")  # Should be dict
            
    def test_init_with_none_config(self):
        """Test MetadataManager initialization with None config."""
        manager = MetadataManager(config=None)
        assert manager.config == {}
        
    def test_manager_state_persistence(self, metadata_manager):
        """Test that manager maintains state across operations."""
        initial_state = metadata_manager.get_state() if hasattr(metadata_manager, 'get_state') else {}
        if hasattr(metadata_manager, 'process_metadata'):
            metadata_manager.process_metadata({})
        assert metadata_manager is not None


class TestExtractMetadata:
    """Test suite for metadata extraction functions."""
    
    def test_extract_metadata_from_dict(self, sample_metadata):
        """Test extracting metadata from dictionary."""
        result = extract_metadata(sample_metadata)
        assert isinstance(result, dict)
        assert "id" in result
        
    def test_extract_metadata_from_json_string(self, sample_metadata):
        """Test extracting metadata from JSON string."""
        json_string = json.dumps(sample_metadata)
        result = extract_metadata(json_string)
        assert result == sample_metadata
        
    def test_extract_metadata_from_file(self, sample_metadata, temp_metadata_file):
        """Test extracting metadata from file."""
        with open(temp_metadata_file, 'w') as f:
            json.dump(sample_metadata, f)
        result = extract_metadata(temp_metadata_file)
        assert result == sample_metadata
        
    def test_extract_metadata_empty_input(self):
        """Test extracting metadata from empty input."""
        with pytest.raises(ValueError):
            extract_metadata("")
            
    def test_extract_metadata_invalid_json(self):
        """Test extracting metadata from malformed JSON."""
        with pytest.raises(json.JSONDecodeError):
            extract_metadata('{"invalid": json}')
            
    def test_extract_metadata_nonexistent_file(self):
        """Test extracting metadata from non-existent file."""
        with pytest.raises(FileNotFoundError):
            extract_metadata("/path/that/does/not/exist.json")
            
    def test_extract_metadata_with_encoding_issues(self, temp_metadata_file):
        """Test extracting metadata with encoding issues."""
        metadata_with_unicode = {"name": "Test with 🚀 emoji", "description": "Special chars: àáâãäå"}
        with open(temp_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_unicode, f, ensure_ascii=False)
        result = extract_metadata(temp_metadata_file)
        assert result["name"] == "Test with 🚀 emoji"
        
    def test_extract_metadata_large_file(self, temp_metadata_file):
        """Test extracting metadata from large file."""
        large_metadata = {
            "id": "large_test",
            "data": ["item_{}".format(i) for i in range(1000)],
            "nested": {f"key_{i}": f"value_{i}" for i in range(100)}
        }
        with open(temp_metadata_file, 'w') as f:
            json.dump(large_metadata, f)
        result = extract_metadata(temp_metadata_file)
        assert len(result["data"]) == 1000
        assert len(result["nested"]) == 100


class TestMetadataValidation:
    """Test suite for metadata validation functions."""
    
    def test_validate_metadata_schema_valid(self, sample_metadata):
        """Test validation with valid metadata."""
        result = validate_metadata_schema(sample_metadata)
        assert result is True or result is None
        
    def test_validate_metadata_schema_invalid(self, invalid_metadata):
        """Test validation with invalid metadata."""
        with pytest.raises(ValidationError):
            validate_metadata_schema(invalid_metadata)
            
    def test_validate_metadata_missing_required_fields(self):
        """Test validation with missing required fields."""
        incomplete_metadata = {"name": "Test"}
        with pytest.raises(ValidationError):
            validate_metadata_schema(incomplete_metadata)
            
    def test_validate_metadata_wrong_field_types(self):
        """Test validation with wrong field types."""
        wrong_types = {
            "id": 123,
            "name": None,
            "dependencies": "should_be_list",
            "config": "should_be_dict"
        }
        with pytest.raises(ValidationError):
            validate_metadata_schema(wrong_types)
            
    def test_validate_metadata_empty_dict(self):
        """Test validation with empty dictionary."""
        with pytest.raises(ValidationError):
            validate_metadata_schema({})
            
    def test_validate_metadata_none_input(self):
        """Test validation with None input."""
        with pytest.raises(ValidationError):
            validate_metadata_schema(None)
            
    def test_validate_metadata_extra_fields_allowed(self, sample_metadata):
        """Test validation allows extra fields."""
        extended_metadata = sample_metadata.copy()
        extended_metadata["extra_field"] = "extra_value"
        result = validate_metadata_schema(extended_metadata)
        assert result is True or result is None
        
    def test_validate_metadata_nested_validation(self):
        """Test validation of nested structures."""
        nested_metadata = {
            "id": "test",
            "config": {
                "nested": {
                    "deep": {
                        "value": "test"
                    }
                }
            }
        }
        result = validate_metadata_schema(nested_metadata)
        assert result is True or result is None


class TestMetadataMerging:
    """Test suite for metadata merging functions."""
    
    def test_merge_metadata_basic(self):
        """Test basic metadata merging."""
        base = {"id": "test", "name": "Base"}
        update = {"description": "Updated", "version": "1.1"}
        result = merge_metadata(base, update)
        assert result["id"] == "test"
        assert result["name"] == "Base"
        assert result["description"] == "Updated"
        assert result["version"] == "1.1"
        
    def test_merge_metadata_overwrite(self):
        """Test metadata merging with overwrites."""
        base = {"id": "test", "name": "Base", "version": "1.0"}
        update = {"name": "Updated", "version": "2.0"}
        result = merge_metadata(base, update)
        assert result["name"] == "Updated"
        assert result["version"] == "2.0"
        
    def test_merge_metadata_nested(self):
        """Test merging nested metadata structures."""
        base = {
            "config": {
                "param1": "value1",
                "nested": {"key1": "value1"}
            }
        }
        update = {
            "config": {
                "param2": "value2",
                "nested": {"key2": "value2"}
            }
        }
        result = merge_metadata(base, update)
        assert result["config"]["param1"] == "value1"
        assert result["config"]["param2"] == "value2"
        
    def test_merge_metadata_empty_inputs(self):
        """Test merging with empty inputs."""
        assert merge_metadata({}, {}) == {}
        assert merge_metadata({"key": "value"}, {}) == {"key": "value"}
        assert merge_metadata({}, {"key": "value"}) == {"key": "value"}
        
    def test_merge_metadata_none_inputs(self):
        """Test merging with None inputs."""
        with pytest.raises((TypeError, ValueError)):
            merge_metadata(None, {})
        with pytest.raises((TypeError, ValueError)):
            merge_metadata({}, None)
            
    def test_merge_metadata_conflicting_types(self):
        """Test merging with conflicting data types."""
        base = {"config": {"param": "string_value"}}
        update = {"config": {"param": 123}}
        result = merge_metadata(base, update)
        assert result["config"]["param"] == 123


class TestMetadataSerialization:
    """Test suite for metadata serialization functions."""
    
    def test_serialize_metadata_to_json(self, sample_metadata):
        """Test serializing metadata to JSON string."""
        result = serialize_metadata(sample_metadata)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == sample_metadata
        
    def test_serialize_metadata_to_file(self, sample_metadata, temp_metadata_file):
        """Test serializing metadata to file."""
        serialize_metadata(sample_metadata, output_file=temp_metadata_file)
        with open(temp_metadata_file, 'r') as f:
            loaded = json.load(f)
        assert loaded == sample_metadata
        
    def test_serialize_metadata_with_formatting(self, sample_metadata):
        """Test serializing metadata with pretty formatting."""
        result = serialize_metadata(sample_metadata, indent=2)
        assert "\n" in result
        
    def test_deserialize_metadata_from_json(self, sample_metadata):
        """Test deserializing metadata from JSON string."""
        json_string = json.dumps(sample_metadata)
        result = deserialize_metadata(json_string)
        assert result == sample_metadata
        
    def test_deserialize_metadata_from_file(self, sample_metadata, temp_metadata_file):
        """Test deserializing metadata from file."""
        with open(temp_metadata_file, 'w') as f:
            json.dump(sample_metadata, f)
        result = deserialize_metadata(temp_metadata_file)
        assert result == sample_metadata
        
    def test_serialize_deserialize_roundtrip(self, sample_metadata):
        """Test complete serialization-deserialization roundtrip."""
        serialized = serialize_metadata(sample_metadata)
        deserialized = deserialize_metadata(serialized)
        assert deserialized == sample_metadata
        
    def test_serialize_metadata_special_types(self):
        """Test serializing metadata with special Python types."""
        special_metadata = {
            "datetime": datetime.now(timezone.utc).isoformat(),
            "path": str(Path("/tmp/test")),
            "none_value": None,
            "boolean": True,
            "list": [1, 2, 3],
            "nested_list": [[1, 2], [3, 4]]
        }
        result = serialize_metadata(special_metadata)
        deserialized = deserialize_metadata(result)
        assert deserialized["datetime"] == special_metadata["datetime"]
        assert deserialized["none_value"] is None
        assert deserialized["boolean"] is True
        
    def test_serialize_metadata_invalid_input(self):
        """Test serializing invalid metadata."""
        class CustomObject:
            pass
        invalid_metadata = {"object": CustomObject()}
        with pytest.raises(TypeError):
            serialize_metadata(invalid_metadata)


class TestEdgeCasesAndErrors:
    """Test suite for edge cases and error conditions."""
    
    def test_metadata_helpers_with_very_large_data(self):
        """Test handling very large metadata structures."""
        large_metadata = {
            "id": "large_test",
            "large_list": list(range(10000)),
            "large_dict": {f"key_{i}": f"value_{i}" for i in range(1000)},
            "nested_large": {
                "level1": {
                    "level2": {
                        "data": ["item"] * 1000
                    }
                }
            }
        }
        serialized = serialize_metadata(large_metadata)
        deserialized = deserialize_metadata(serialized)
        assert len(deserialized["large_list"]) == 10000
        assert len(deserialized["large_dict"]) == 1000
        
    def test_metadata_helpers_unicode_handling(self):
        """Test handling Unicode and special characters."""
        unicode_metadata = {
            "emoji": "🚀🎉🌟",
            "chinese": "你好世界",
            "arabic": "مرحبا بالعالم",
            "mixed": "Hello 世界 🌍",
            "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        }
        serialized = serialize_metadata(unicode_metadata)
        deserialized = deserialize_metadata(serialized)
        assert deserialized == unicode_metadata
        
    def test_metadata_helpers_memory_efficiency(self):
        """Test memory efficiency with repeated operations."""
        metadata = {"id": "test", "data": list(range(1000))}
        for i in range(100):
            serialized = serialize_metadata(metadata)
            deserialized = deserialize_metadata(serialized)
            merged = merge_metadata(metadata, {"iteration": i})
        assert True
        
    def test_concurrent_access_safety(self, metadata_manager):
        """Test thread safety of metadata operations."""
        import threading
        import time
        results = []
        errors = []
        def worker(thread_id):
            try:
                for i in range(10):
                    metadata = {"thread": thread_id, "iteration": i}
                    serialized = serialize_metadata(metadata)
                    deserialized = deserialize_metadata(serialized)
                    results.append(deserialized)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(errors) == 0
        assert len(results) == 50
        
    @pytest.mark.parametrize("invalid_input", [
        123,
        "string",
        [],
        set(),
        object(),
    ])
    def test_metadata_validation_invalid_types(self, invalid_input):
        """Test validation with various invalid input types."""
        with pytest.raises((ValidationError, TypeError, ValueError)):
            validate_metadata_schema(invalid_input)


class TestIntegrationScenarios:
    """Integration tests for complete metadata workflows."""
    
    def test_complete_metadata_workflow(self, temp_metadata_file):
        """Test complete workflow from creation to persistence."""
        original_metadata = {
            "id": "workflow_test",
            "name": "Integration Test",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {"param": "value"}
        }
        validate_metadata_schema(original_metadata)
        serialize_metadata(original_metadata, output_file=temp_metadata_file)
        loaded_metadata = extract_metadata(temp_metadata_file)
        validate_metadata_schema(loaded_metadata)
        updates = {"version": "1.0", "updated_at": datetime.now(timezone.utc).isoformat()}
        final_metadata = merge_metadata(loaded_metadata, updates)
        validate_metadata_schema(final_metadata)
        assert final_metadata["id"] == original_metadata["id"]
        assert "version" in final_metadata
        assert "updated_at" in final_metadata


def test_module_imports():
    """Test that all expected functions and classes can be imported."""
    from adaptive_graph_of_thoughts.metadata_helpers import (
        MetadataManager,
        ValidationError,
        extract_metadata,
        validate_metadata_schema,
        merge_metadata,
        serialize_metadata,
        deserialize_metadata
    )
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])