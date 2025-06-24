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
