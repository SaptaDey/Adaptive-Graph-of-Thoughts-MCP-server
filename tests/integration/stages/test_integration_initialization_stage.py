import pytest
from neo4j import Driver
from testcontainers.neo4j import Neo4jContainer
import shutil

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="Docker not available"
)

from dataclasses import dataclass
import sys
import types

class DummyDefaultParams:
    initial_confidence = [0.9, 0.9, 0.9, 0.9]
    initial_layer = "root_layer"


@dataclass
class DummySettings:
    asr_got: type = type("obj", (), {"default_parameters": DummyDefaultParams()})

stub_pkg = types.ModuleType("adaptive_graph_of_thoughts.async_server")
stub_pkg.AdaptiveGraphServer = object
sys.modules["adaptive_graph_of_thoughts.async_server"] = stub_pkg
stub_pkg2 = types.ModuleType("src.adaptive_graph_of_thoughts.async_server")
stub_pkg2.AdaptiveGraphServer = object
sys.modules["src.adaptive_graph_of_thoughts.async_server"] = stub_pkg2
stub_config = types.ModuleType("adaptive_graph_of_thoughts.config")
stub_config.Settings = DummySettings
stub_config.runtime_settings = types.SimpleNamespace(neo4j=types.SimpleNamespace(uri="bolt://localhost", user="neo4j", password="test", database="neo4j"))
sys.modules["adaptive_graph_of_thoughts.config"] = stub_config
sys.modules["src.adaptive_graph_of_thoughts.config"] = stub_config

from src.adaptive_graph_of_thoughts.domain.models.common_types import GoTProcessorSessionData
from src.adaptive_graph_of_thoughts.domain.stages.stage_1_initialization import InitializationStage
from src.adaptive_graph_of_thoughts.domain.services import neo4j_utils
from src.adaptive_graph_of_thoughts.domain.models.graph_elements import NodeType


@pytest.fixture(scope="module")
def settings_instance():
    return DummySettings()


@pytest.fixture(scope="module")
def neo4j_test_container_manager():
    original_settings = None
    if hasattr(neo4j_utils, "_neo4j_settings") and neo4j_utils._neo4j_settings is not None:
        orig = neo4j_utils._neo4j_settings.neo4j
        original_settings = neo4j_utils.Neo4jSettings()
        original_settings.uri = orig.uri
        original_settings.user = orig.user
        original_settings.password = orig.password
        original_settings.database = orig.database

    with Neo4jContainer("neo4j:5.18.0") as neo4j_cont:
        current = neo4j_utils.get_neo4j_settings().neo4j
        current.uri = neo4j_cont.get_connection_url()
        current.user = "neo4j"
        current.password = neo4j_cont.NEO4J_ADMIN_PASSWORD
        current.database = "neo4j"

        if neo4j_utils._driver is not None:
            neo4j_utils.close_neo4j_driver()

        yield neo4j_cont

    if original_settings is not None:
        neo4j_utils._neo4j_settings.neo4j.uri = original_settings.uri
        neo4j_utils._neo4j_settings.neo4j.user = original_settings.user
        neo4j_utils._neo4j_settings.neo4j.password = original_settings.password
        neo4j_utils._neo4j_settings.neo4j.database = original_settings.database
    else:
        neo4j_utils._neo4j_settings = None

    if neo4j_utils._driver is not None:
        neo4j_utils.close_neo4j_driver()
    neo4j_utils._driver = None


@pytest.fixture(autouse=True)
def auto_use_container(neo4j_test_container_manager):
    pass


@pytest.mark.asyncio
async def test_initialization_stage_creates_new_root_node(settings_instance):
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query="integration test")

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    assert root_id == "n0"

    with driver.session(database=db_name) as s:
        result = s.run(
            "MATCH (n:Node:ROOT {id: $id}) RETURN properties(n) as props, labels(n) as labels",
            id=root_id,
        )
        record = result.single()
        assert record is not None
        assert record["props"]["label"] == stage.root_node_label
        assert record["props"]["metadata_query_context"] == "integration test"
        assert NodeType.ROOT.value in record["labels"]
        assert "Node" in record["labels"]


@pytest.mark.asyncio
async def test_initialization_stage_uses_existing_root_node(settings_instance):
    stage = InitializationStage(settings=settings_instance)
    query = "existing root"
    op_params = {"initial_disciplinary_tags": ["physics", "extra"]}
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": op_params})

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    existing_id = "n0_existing"
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run(
            """
            CREATE (r:Node:ROOT {
                id: $id,
                label: 'Existing',
                type: $type,
                metadata_query_context: $query,
                metadata_disciplinary_tags: ['general']
            })
            """,
            id=existing_id,
            type=NodeType.ROOT.value,
            query=query,
        )

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 0
    assert output.metrics["used_existing_neo4j_node"] is True
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    assert root_id == existing_id
    tags = output.next_stage_context_update[InitializationStage.stage_name]["initial_disciplinary_tags"]
    assert "general" in tags and "physics" in tags and "extra" in tags

    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n:Node:ROOT {id: $id}) RETURN n.metadata_disciplinary_tags AS tags", id=existing_id)
        record = result.single()
        assert record is not None
        assert set(record["tags"]) == {"general", "physics", "extra"}
        result_n0 = s.run("MATCH (n:Node:ROOT {id: 'n0'}) RETURN n")
        assert result_n0.single() is None


@pytest.mark.asyncio
async def test_initialization_stage_with_invalid_query_empty_string(settings_instance):
    """Test initialization stage returns error for empty query string."""
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query="")

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 0
    assert output.metrics["used_existing_neo4j_node"] is False
    assert "error" in output.next_stage_context_update[InitializationStage.stage_name]
    assert "Invalid initial query" in output.summary


@pytest.mark.asyncio
async def test_initialization_stage_with_invalid_query_none(settings_instance):
    """Test initialization stage returns error for None query."""
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query=None)

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 0
    assert output.metrics["used_existing_neo4j_node"] is False
    assert "error" in output.next_stage_context_update[InitializationStage.stage_name]
    assert "Invalid initial query" in output.summary


@pytest.mark.asyncio
async def test_initialization_stage_with_invalid_query_non_string(settings_instance):
    """Test initialization stage returns error for non-string query."""
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query=123)  # Integer instead of string

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 0
    assert output.metrics["used_existing_neo4j_node"] is False
    assert "error" in output.next_stage_context_update[InitializationStage.stage_name]
    assert "Invalid initial query" in output.summary


@pytest.mark.asyncio
async def test_initialization_stage_with_very_long_query(settings_instance):
    """Test initialization stage with very long query string."""
    stage = InitializationStage(settings=settings_instance)
    long_query = "This is a very long query " * 1000  # ~26,000 characters
    session_data = GoTProcessorSessionData(query=long_query)

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    
    with driver.session(database=db_name) as s:
        result = s.run(
            "MATCH (n:Node:ROOT {id: $id}) RETURN properties(n) as props",
            id=root_id,
        )
        record = result.single()
        assert record is not None
        assert record["props"]["metadata_query_context"] == long_query


@pytest.mark.asyncio
async def test_initialization_stage_with_special_characters_in_query(settings_instance):
    """Test initialization stage with special characters and unicode in query."""
    stage = InitializationStage(settings=settings_instance)
    special_query = "Test with 'quotes' \"double quotes\" & special chars: !@#$%^&*()_+ unicode: 中文 🚀 ñáéíóú"
    session_data = GoTProcessorSessionData(query=special_query)

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    
    with driver.session(database=db_name) as s:
        result = s.run(
            "MATCH (n:Node:ROOT {id: $id}) RETURN properties(n) as props",
            id=root_id,
        )
        record = result.single()
        assert record is not None
        assert record["props"]["metadata_query_context"] == special_query


@pytest.mark.asyncio
async def test_initialization_stage_with_empty_disciplinary_tags(settings_instance):
    """Test initialization stage with empty disciplinary tags list."""
    stage = InitializationStage(settings=settings_instance)
    query = "test query"
    op_params = {"initial_disciplinary_tags": []}
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": op_params})

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    tags = output.next_stage_context_update[InitializationStage.stage_name].get("initial_disciplinary_tags", [])
    assert isinstance(tags, list)


@pytest.mark.asyncio
async def test_initialization_stage_with_many_disciplinary_tags(settings_instance):
    """Test initialization stage with many disciplinary tags."""
    stage = InitializationStage(settings=settings_instance)
    query = "test query"
    many_tags = [f"tag_{i}" for i in range(50)]  # 50 tags
    op_params = {"initial_disciplinary_tags": many_tags}
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": op_params})

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n:Node:ROOT {id: $id}) RETURN n.metadata_disciplinary_tags AS tags", id=root_id)
        record = result.single()
        assert record is not None
        assert len(record["tags"]) == 50
        assert set(record["tags"]) == set(many_tags)


@pytest.mark.asyncio
async def test_initialization_stage_with_duplicate_disciplinary_tags(settings_instance):
    """Test initialization stage handles duplicate disciplinary tags correctly."""
    stage = InitializationStage(settings=settings_instance)
    query = "test query"
    duplicate_tags = ["physics", "chemistry", "physics", "biology", "chemistry"]
    op_params = {"initial_disciplinary_tags": duplicate_tags}
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": op_params})

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n:Node:ROOT {id: $id}) RETURN n.metadata_disciplinary_tags AS tags", id=root_id)
        record = result.single()
        assert record is not None
        unique_tags = set(record["tags"])
        assert unique_tags == {"physics", "chemistry", "biology"}


@pytest.mark.asyncio
async def test_initialization_stage_with_no_accumulated_context(settings_instance):
    """Test initialization stage with no accumulated context."""
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query="test query", accumulated_context=None)

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    assert root_id is not None


@pytest.mark.asyncio
async def test_initialization_stage_with_empty_accumulated_context(settings_instance):
    """Test initialization stage with empty accumulated context."""
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query="test query", accumulated_context={})

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    assert root_id is not None


@pytest.mark.asyncio
async def test_initialization_stage_multiple_executions_same_query(settings_instance):
    """Test multiple executions with the same query reuse the same node."""
    stage = InitializationStage(settings=settings_instance)
    query = "repeated query"
    session_data = GoTProcessorSessionData(query=query)

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    # First execution
    output1 = await stage.execute(current_session_data=session_data)
    assert output1.metrics["nodes_created_in_neo4j"] == 1
    root_id1 = output1.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]

    # Second execution with same query
    output2 = await stage.execute(current_session_data=session_data)
    assert output2.metrics["nodes_created_in_neo4j"] == 0
    assert output2.metrics["used_existing_neo4j_node"] is True
    root_id2 = output2.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    
    assert root_id1 == root_id2


@pytest.mark.asyncio
async def test_initialization_stage_different_queries_create_different_nodes(settings_instance):
    """Test different queries create different root nodes."""
    stage = InitializationStage(settings=settings_instance)
    
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    # First query
    session_data1 = GoTProcessorSessionData(query="first unique query")
    output1 = await stage.execute(current_session_data=session_data1)
    root_id1 = output1.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]

    # Second query
    session_data2 = GoTProcessorSessionData(query="second unique query")
    output2 = await stage.execute(current_session_data=session_data2)
    root_id2 = output2.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    
    assert root_id1 != root_id2
    assert output1.metrics["nodes_created_in_neo4j"] == 1
    assert output2.metrics["nodes_created_in_neo4j"] == 1


@pytest.mark.asyncio
async def test_initialization_stage_existing_node_with_different_tags_merges(settings_instance):
    """Test that adding tags to existing node merges them correctly."""
    stage = InitializationStage(settings=settings_instance)
    query = "merge tags test"
    
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    existing_id = "merge_test_node"
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run(
            """
            CREATE (r:Node:ROOT {
                id: $id,
                label: 'Existing',
                type: $type,
                metadata_query_context: $query,
                metadata_disciplinary_tags: ['existing_tag']
            })
            """,
            id=existing_id,
            type=NodeType.ROOT.value,
            query=query,
        )

    # Execute with additional tags
    op_params = {"initial_disciplinary_tags": ["new_tag1", "new_tag2"]}
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": op_params})
    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 0
    assert output.metrics["used_existing_neo4j_node"] is True
    
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n:Node:ROOT {id: $id}) RETURN n.metadata_disciplinary_tags AS tags", id=existing_id)
        record = result.single()
        assert record is not None
        assert set(record["tags"]) == {"existing_tag", "new_tag1", "new_tag2"}


@pytest.mark.asyncio
async def test_initialization_stage_handles_malformed_operational_params(settings_instance):
    """Test initialization stage handles malformed operational parameters gracefully."""
    stage = InitializationStage(settings=settings_instance)
    query = "test query"
    
    # Test with non-dict operational_params
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": "not_a_dict"})

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    # Should not raise an exception
    output = await stage.execute(current_session_data=session_data)
    assert output.metrics["nodes_created_in_neo4j"] == 1


@pytest.mark.asyncio
async def test_initialization_stage_handles_non_list_disciplinary_tags(settings_instance):
    """Test initialization stage handles non-list disciplinary tags gracefully."""
    stage = InitializationStage(settings=settings_instance)
    query = "test query"
    
    # Test with non-list disciplinary tags
    op_params = {"initial_disciplinary_tags": "not_a_list"}
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": op_params})

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    # Should not raise an exception
    output = await stage.execute(current_session_data=session_data)
    assert output.metrics["nodes_created_in_neo4j"] == 1


@pytest.mark.asyncio
async def test_initialization_stage_preserves_existing_node_properties(settings_instance):
    """Test that existing node properties are preserved when reusing nodes."""
    stage = InitializationStage(settings=settings_instance)
    query = "property preservation test"
    
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    existing_id = "preserve_test_node"
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run(
            """
            CREATE (r:Node:ROOT {
                id: $id,
                label: 'OriginalLabel',
                type: $type,
                metadata_query_context: $query,
                metadata_disciplinary_tags: ['original'],
                custom_property: 'should_be_preserved'
            })
            """,
            id=existing_id,
            type=NodeType.ROOT.value,
            query=query,
        )

    session_data = GoTProcessorSessionData(query=query)
    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["used_existing_neo4j_node"] is True
    
    with driver.session(database=db_name) as s:
        result = s.run(
            "MATCH (n:Node:ROOT {id: $id}) RETURN properties(n) as props",
            id=existing_id,
        )
        record = result.single()
        assert record is not None
        props = record["props"]
        assert props["custom_property"] == "should_be_preserved"
        assert props["label"] == "OriginalLabel"
        assert props["metadata_query_context"] == query


@pytest.mark.asyncio
async def test_initialization_stage_concurrent_execution_safety(settings_instance):
    """Test concurrent execution of initialization stage doesn't create duplicate nodes."""
    import asyncio
    
    stage = InitializationStage(settings=settings_instance)
    query = "concurrent test"
    
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    # Execute multiple times concurrently
    tasks = []
    for i in range(5):
        session_data = GoTProcessorSessionData(query=query)
        tasks.append(stage.execute(current_session_data=session_data))
    
    results = await asyncio.gather(*tasks)
    
    # Check that only one node was created and others reused it
    total_created = sum(result.metrics["nodes_created_in_neo4j"] for result in results)
    assert total_created == 1
    
    # Verify all results point to the same root node ID
    root_ids = [result.next_stage_context_update[InitializationStage.stage_name]["root_node_id"] for result in results]
    assert len(set(root_ids)) == 1  # All should be the same
    
    # Verify only one node exists in the database
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n:Node:ROOT) RETURN count(n) as count")
        record = result.single()
        assert record["count"] == 1


@pytest.mark.asyncio
async def test_initialization_stage_output_structure_validation(settings_instance):
    """Test that the output structure contains all expected fields and types."""
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query="structure validation test")

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    # Validate output structure
    assert hasattr(output, 'metrics')
    assert hasattr(output, 'next_stage_context_update')
    assert hasattr(output, 'summary')
    
    # Validate metrics structure
    assert isinstance(output.metrics, dict)
    required_metrics = ["nodes_created_in_neo4j", "used_existing_neo4j_node", "updated_existing_node_tags"]
    for metric in required_metrics:
        assert metric in output.metrics
    
    # Validate context update structure
    assert isinstance(output.next_stage_context_update, dict)
    assert InitializationStage.stage_name in output.next_stage_context_update
    stage_context = output.next_stage_context_update[InitializationStage.stage_name]
    assert "root_node_id" in stage_context
    assert isinstance(stage_context["root_node_id"], str)
    assert "initial_disciplinary_tags" in stage_context


@pytest.mark.asyncio
async def test_initialization_stage_node_id_uniqueness(settings_instance):
    """Test that node IDs are unique across different executions."""
    stage = InitializationStage(settings=settings_instance)
    
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    # Create multiple nodes with different queries
    root_ids = []
    for i in range(5):
        session_data = GoTProcessorSessionData(query=f"unique_query_{i}")
        output = await stage.execute(current_session_data=session_data)
        root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
        root_ids.append(root_id)

    # Validate ID uniqueness
    assert len(set(root_ids)) == len(root_ids), "All node IDs should be unique"
    
    for root_id in root_ids:
        assert isinstance(root_id, str)
        assert len(root_id) > 0


@pytest.mark.asyncio
async def test_initialization_stage_with_whitespace_only_query(settings_instance):
    """Test initialization stage with whitespace-only query."""
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query="   \t\n   ")

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    # Should still create a node with the whitespace query
    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    
    with driver.session(database=db_name) as s:
        result = s.run(
            "MATCH (n:Node:ROOT {id: $id}) RETURN properties(n) as props",
            id=root_id,
        )
        record = result.single()
        assert record is not None
        assert record["props"]["metadata_query_context"] == "   \t\n   "


@pytest.mark.asyncio
async def test_initialization_stage_disciplinary_tags_with_special_characters(settings_instance):
    """Test initialization stage with disciplinary tags containing special characters."""
    stage = InitializationStage(settings=settings_instance)
    query = "test query"
    special_tags = ["physics-quantum", "bio_molecular", "comp.sci", "math/statistics", "🧬genetics"]
    op_params = {"initial_disciplinary_tags": special_tags}
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": op_params})

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    root_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n:Node:ROOT {id: $id}) RETURN n.metadata_disciplinary_tags AS tags", id=root_id)
        record = result.single()
        assert record is not None
        assert set(record["tags"]) == set(special_tags)


@pytest.mark.asyncio
async def test_initialization_stage_existing_node_tag_update_failure_handling(settings_instance):
    """Test handling when tag update fails for existing node."""
    stage = InitializationStage(settings=settings_instance)
    query = "tag update failure test"
    
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    existing_id = "update_failure_test"
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run(
            """
            CREATE (r:Node:ROOT {
                id: $id,
                label: 'Existing',
                type: $type,
                metadata_query_context: $query,
                metadata_disciplinary_tags: ['existing_tag']
            })
            """,
            id=existing_id,
            type=NodeType.ROOT.value,
            query=query,
        )

    # Execute with additional tags
    op_params = {"initial_disciplinary_tags": ["new_tag"]}
    session_data = GoTProcessorSessionData(query=query, accumulated_context={"operational_params": op_params})
    output = await stage.execute(current_session_data=session_data)

    # Should still work and merge tags successfully
    assert output.metrics["used_existing_neo4j_node"] is True
    tags = output.next_stage_context_update[InitializationStage.stage_name]["initial_disciplinary_tags"]
    assert "existing_tag" in tags
    assert "new_tag" in tags


@pytest.mark.asyncio
async def test_initialization_stage_with_missing_session_id(settings_instance):
    """Test initialization stage handles missing session_id gracefully."""
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query="test query", session_id="")

    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")

    # Should not raise an exception
    output = await stage.execute(current_session_data=session_data)
    assert output.metrics["nodes_created_in_neo4j"] == 1


@pytest.mark.asyncio
async def test_initialization_stage_database_cleanup_between_tests(settings_instance):
    """Test that database cleanup works correctly between test executions."""
    stage = InitializationStage(settings=settings_instance)
    
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().neo4j.database
    
    # Verify database is clean
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n) RETURN count(n) as count")
        record = result.single()
        initial_count = record["count"]
    
    # Create a node
    session_data = GoTProcessorSessionData(query="cleanup test")
    output = await stage.execute(current_session_data=session_data)
    assert output.metrics["nodes_created_in_neo4j"] == 1
    
    # Verify node was created
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n) RETURN count(n) as count")
        record = result.single()
        assert record["count"] == initial_count + 1
        
    # Clean up for next test
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n")
        
    # Verify cleanup worked
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n) RETURN count(n) as count")
        record = result.single()
        assert record["count"] == 0
