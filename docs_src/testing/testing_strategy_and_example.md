## Testing Strategy Outline

A comprehensive testing strategy for the refactored Neo4j-native application should cover different levels of granularity to ensure correctness and robustness.

### 1. Unit Tests

*   **Focus:**
    *   Test individual helper functions within each stage that do *not* involve direct database calls. This includes:
        *   Property preparation logic (e.g., `_prepare_node_properties_for_neo4j`, `_prepare_edge_properties_for_neo4j` if they exist in stages or are moved to a utility).
        *   Data transformation and parsing logic (e.g., parsing JSON strings from Neo4j properties into Pydantic models if done within a stage).
        *   Complex business logic that determines what data to write (e.g., `_generate_hypothesis_content` in `HypothesisStage`, `_get_conceptual_dimensions` in `DecompositionStage`).
        *   Conditional logic within stages that determines flow based on input data (not DB state).
*   **Mocking:**
    *   External dependencies like LLM calls (if any) should be mocked.
    *   Direct Neo4j `execute_query` calls should **not** be part of unit tests; these are covered by integration tests. If a helper function *unavoidably* prepares data and then calls `execute_query`, the `execute_query` part should be mocked to assert it's called with correct parameters.
*   **Tools:**
    *   `pytest` (for test structure, fixtures, assertions).
    *   `unittest.mock.patch` or `pytest-mock` (for mocking).

### 2. Integration Tests (Per Stage)

*   **Focus:** Verify the direct Neo4j interaction of each refactored stage. Ensure that each stage correctly reads from and/or writes to Neo4j as per its responsibilities.
*   **Setup:**
    *   **Testcontainers:** Use `testcontainers-python` with the official Neo4j container (e.g., `Neo4jContainer`). This allows spinning up a clean, ephemeral Neo4j instance for each test module/suite or even per test if necessary (though per-module is often a good balance).
    *   **Neo4j Configuration:** The `neo4j_utils` module (or wherever the Neo4j driver connection is managed) must be configured to connect to the dynamically created test Neo4j instance. This typically involves:
        *   Modifying the URI, user, and password used by `neo4j_utils` at runtime within the test setup (e.g., via environment variables that `Neo4jSettings` in `neo4j_utils` can pick up, or by directly patching the settings object within `neo4j_utils` for the duration of the test).
        *   Ensuring the Neo4j driver instance is re-initialized with the test container's details.
*   **Test Flow (Arrange, Act, Assert):**
    1.  **Arrange:**
        *   Before running the stage, use the Neo4j driver (connected to the test container) to populate the database with any prerequisite data specific to the test case. For example:
            *   For `DecompositionStage`, ensure a root node with a specific ID and properties exists.
            *   For `HypothesisStage`, ensure dimension nodes (created by `DecompositionStage`) exist.
            *   For `EvidenceStage`, ensure hypothesis nodes exist.
        *   Prepare `GoTProcessorSessionData` with the necessary input context for the stage being tested (e.g., `root_node_id` for `DecompositionStage`).
    2.  **Act:**
        *   Instantiate the stage with appropriate settings.
        *   Execute the stage's `execute(current_session_data=...)` method.
    3.  **Assert:**
        *   **Database State:** Use the Neo4j driver to query the test database.
            *   Verify that the expected nodes were created/updated/deleted.
            *   Verify that the expected relationships were created/updated/deleted.
            *   Check the properties, labels of these nodes and relationships.
        *   **Stage Output:**
            *   Validate the `StageOutput` object returned by the `execute` method.
            *   Check `next_stage_context_update` for correct data being passed on (e.g., new node IDs).
            *   Check `metrics` for accuracy (e.g., `nodes_created_in_neo4j`).
*   **Teardown:**
    *   Testcontainers will automatically stop and remove the Neo4j container after tests.
    *   Ensure any modifications to global state (like `neo4j_utils` settings) are reverted if necessary (fixtures with `yield` can handle this).
*   **Tools:**
    *   `pytest` (especially for fixtures and async tests if stages are async).
    *   `testcontainers-python`.
    *   `neo4j` Python driver (for setting up DB state and making assertions against DB).

### 3. End-to-End Tests (`GoTProcessor`)

*   **Focus:** Test the entire `GoTProcessor.process_query` flow, ensuring all Neo4j-native stages integrate correctly and produce the expected final output and database state.
*   **Setup:**
    *   Use Testcontainers for a clean Neo4j instance.
    *   Configure `neo4j_utils` to point to this test instance.
*   **Test Flow:**
    1.  **Arrange:**
        *   Optionally, pre-populate Neo4j with any baseline data relevant for a complex scenario.
        *   Prepare the input query string and any `operational_params` or `initial_context`.
    2.  **Act:**
        *   Instantiate `GoTProcessor`.
        *   Call `GoTProcessor.process_query(...)`.
    3.  **Assert:**
        *   **Final Database State:** Query Neo4j to verify the overall graph structure created by the full pipeline (e.g., presence of root, dimensions, hypotheses, evidence, IBNs, hyperedges as expected for the input query). Check key properties.
        *   **`GoTProcessorSessionData`:** Inspect the session data object returned by `process_query`.
            *   Verify `final_answer` and `final_confidence_vector`.
            *   Check the `stage_outputs_trace` for completeness and correctness of summaries/errors per stage.
            *   Verify the `accumulated_context` contains expected final outputs from key stages (like `CompositionStage`'s output).
*   **Tools:**
    *   `pytest`.
    *   `testcontainers-python`.
    *   `neo4j` Python driver.

### 4. Dependency Management for Tests

*   Ensure test-specific dependencies like `testcontainers-python`, `pytest-asyncio` (if using async stages), and any other relevant testing utilities are added to the project's development dependencies (e.g., in `pyproject.toml` under `[tool.poetry.group.test.dependencies]` or equivalent for other package managers).

### Acknowledgment on `neo4j_utils` Configuration for Testing:
The `neo4j_utils` module, due to its singleton-like nature for managing Neo4j settings and the driver, requires careful handling in a testing environment. The example integration test demonstrates a method of patching its internal settings (`_neo4j_settings`) and driver instance (`_driver`) within a pytest fixture. This allows tests to redirect Neo4j operations to a test-specific container. Alternatives include designing `neo4j_utils` to prioritize environment variables for connection details (which Testcontainers can provide) or adding an explicit re-initialization function to `neo4j_utils`. The patching approach in the example is a pragmatic way to test the current structure.

## Example Integration Test (for `InitializationStage`)

```python
import pytest
from neo4j import Driver # type: ignore
from testcontainers.neo4j import Neo4jContainer # type: ignore
import os # For potentially setting environment variables

from src.asr_got_reimagined.config import Settings
from src.asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from src.asr_got_reimagined.domain.stages.stage_1_initialization import InitializationStage
# Import the neo4j_utils module itself to allow patching/re-evaluating its globals
from src.asr_got_reimagined.domain.services import neo4j_utils
from src.asr_got_reimagined.domain.models.graph_elements import NodeType


@pytest.fixture(scope="module")
def settings_instance():
    """Provides a Settings instance for tests."""
    return Settings() # Assumes Settings() can load from a test .env or has defaults

@pytest.fixture(scope="module")
def neo4j_test_container_manager():
    """
    Manages a Neo4j test container for the test module.
    This fixture demonstrates patching neo4j_utils internals.
    """
    original_settings = None
    original_driver = None

    # Attempt to capture original state if neo4j_utils was already initialized
    if hasattr(neo4j_utils, '_neo4j_settings') and neo4j_utils._neo4j_settings is not None:
        original_settings = neo4j_utils.Neo4jSettings(
            uri=neo4j_utils._neo4j_settings.uri,
            user=neo4j_utils._neo4j_settings.user,
            password=neo4j_utils._neo4j_settings.password,
            database=neo4j_utils._neo4j_settings.database
        )
    if hasattr(neo4j_utils, '_driver') and neo4j_utils._driver is not None:
        original_driver = neo4j_utils._driver

    with Neo4jContainer("neo4j:5.18.0") as neo4j_cont:
        # Override settings in neo4j_utils
        # Ensure _neo4j_settings exists; get_neo4j_settings() will create it if None
        current_settings_obj = neo4j_utils.get_neo4j_settings() 
        
        current_settings_obj.uri = neo4j_cont.get_connection_url()
        current_settings_obj.user = "neo4j" # Default for container
        current_settings_obj.password = neo4j_cont.NEO4J_ADMIN_PASSWORD
        current_settings_obj.database = "neo4j" # Default DB in container

        # Force re-initialization of the driver in neo4j_utils if it was already set
        if neo4j_utils._driver is not None:
            neo4j_utils.close_neo4j_driver() # Closes and sets _driver to None
        
        # The next call to get_neo4j_driver() within tests will use these patched settings
        yield neo4j_cont

    # Teardown: Restore original settings and driver IF they existed
    if original_settings:
        neo4j_utils._neo4j_settings.uri = original_settings.uri
        neo4j_utils._neo4j_settings.user = original_settings.user
        neo4j_utils._neo4j_settings.password = original_settings.password
        neo4j_utils._neo4j_settings.database = original_settings.database
    else: # If it was None before, reset to None so it can be lazy-loaded again
        neo4j_utils._neo4j_settings = None

    if neo4j_utils._driver is not None: # If a test driver was created
        neo4j_utils.close_neo4j_driver() # Ensure it's closed
    
    # If original_driver was something, restore it (though typically it'd be None after close)
    # Forcing a clean state for subsequent modules is often best:
    neo4j_utils._driver = None


@pytest.fixture(autouse=True)
def auto_use_neo4j_container_manager(neo4j_test_container_manager):
    """Fixture to automatically apply the container manager for all tests in the module."""
    pass


@pytest.mark.asyncio
async def test_initialization_stage_creates_new_root_node(settings_instance):
    stage = InitializationStage(settings=settings_instance)
    session_data = GoTProcessorSessionData(query="Test query for new root")

    # Driver will be configured by neo4j_test_container_manager fixture
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().database

    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n") # Clean database for this test

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 1
    assert output.metrics["used_existing_neo4j_node"] is False
    root_node_id = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    assert root_node_id == "n0" # Default ID for new root

    # Verify node in Neo4j
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n:Node:ROOT {id: $id}) RETURN properties(n) as props, labels(n) as labels", id=root_node_id)
        record = result.single()
        assert record is not None
        assert record["props"]["label"] == stage.root_node_label
        assert record["props"]["metadata_query_context"] == "Test query for new root"
        assert NodeType.ROOT.value in record["labels"] 
        assert "Node" in record["labels"]

@pytest.mark.asyncio
async def test_initialization_stage_uses_existing_root_node(settings_instance):
    stage = InitializationStage(settings=settings_instance)
    initial_query = "Test query for existing root"
    op_params = {"initial_disciplinary_tags": ["physics", "new_tag"]}
    session_data = GoTProcessorSessionData(query=initial_query, accumulated_context={"operational_params": op_params})
    
    driver: Driver = neo4j_utils.get_neo4j_driver()
    db_name = neo4j_utils.get_neo4j_settings().database

    existing_node_id = "n0_existing" # Using a different ID for existing node
    with driver.session(database=db_name) as s:
        s.run("MATCH (n) DETACH DELETE n") # Clean first
        # Create the node with the ROOT label and also the specific type label via property
        s.run(f"""
            CREATE (r:Node:ROOT {{
                id: $id,
                label: "Existing Task Understanding",
                type: "{NodeType.ROOT.value}", 
                metadata_query_context: $query,
                metadata_disciplinary_tags: ["general", "physics"] 
            }})
        """, id=existing_node_id, query=initial_query)

    output = await stage.execute(current_session_data=session_data)

    assert output.metrics["nodes_created_in_neo4j"] == 0
    assert output.metrics["used_existing_neo4j_node"] is True
    assert output.metrics.get("updated_existing_node_tags", False) is True 
    
    root_node_id_from_output = output.next_stage_context_update[InitializationStage.stage_name]["root_node_id"]
    assert root_node_id_from_output == existing_node_id

    updated_tags_from_output = output.next_stage_context_update[InitializationStage.stage_name]["initial_disciplinary_tags"]
    assert "general" in updated_tags_from_output
    assert "physics" in updated_tags_from_output
    assert "new_tag" in updated_tags_from_output

    # Verify tags in Neo4j
    with driver.session(database=db_name) as s:
        result = s.run("MATCH (n:Node:ROOT {id: $id}) RETURN n.metadata_disciplinary_tags AS tags", id=existing_node_id)
        record = result.single()
        assert record is not None
        assert "general" in record["tags"]
        assert "physics" in record["tags"]
        assert "new_tag" in record["tags"]
        
        # Ensure no "n0" node was created if the existing one was used
        result_n0 = s.run("MATCH (n:Node:ROOT {id: 'n0'}) RETURN n")
        assert result_n0.single() is None
```
This file (`testing_strategy_and_example.md`) contains the testing strategy and the example integration test.
The example test has been updated to:
1. Use a module-scoped fixture `neo4j_test_container_manager` that attempts to save and restore the state of `neo4j_utils._neo4j_settings` and `neo4j_utils._driver`. This makes it more robust if `neo4j_utils` was initialized before tests or by other test modules.
2. Ensure the test Neo4j database is cleaned (`MATCH (n) DETACH DELETE n`) before each relevant test execution to ensure test isolation.
3. Explicitly use the database name from `neo4j_utils.get_neo4j_settings().database` when opening sessions, which defaults to "neo4j" for the container.
4. The existing node in the second test uses `id: "n0_existing"` to avoid collision if the stage accidentally creates "n0".
5. Assert that the default "n0" node is *not* created when an existing node is used.
6. The `auto_use_neo4j_container_manager` fixture ensures that the container setup and teardown logic is applied for all tests within the module where this fixture is defined or imported.

The test file would typically reside in a path like `tests/integration/stages/test_integration_initialization_stage.py`.
The `neo4j_utils` module's singleton pattern for driver and settings management makes testing a bit more involved regarding setup/teardown of this global state. The fixture tries to handle this gracefully.
The use of `pytest-asyncio` would be necessary to run these `async def` tests (e.g., by installing it and `pytest` running normally).
The `Neo4jContainer` version is specified as "neo4j:5.18.0" as an example; it should match a version compatible with the application's Cypher queries and features (like APOC if used by other stages).
The `type: ignore` comments are for `testcontainers` as it might not have perfect type hints recognized by all static analyzers immediately.
The `neo4j_utils._NEO4J_SETTINGS.uri = original_uri` line in the previous version of the test was incorrect as `_NEO4J_SETTINGS` is the Pydantic model instance itself, so its attributes should be set directly. This has been corrected in this version.
Also corrected setting `neo4j_utils._neo4j_settings` to `None` if it was `None` originally, to ensure proper lazy loading behavior is restored.
Final check of `test_initialization_stage_uses_existing_root_node`: the node creation query was updated to ensure the `ROOT` label is applied, and `type` property is set for consistency with how `InitializationStage` creates new nodes.
The `neo4j_test_container_manager` fixture now correctly captures the original settings by instantiating a `Neo4jSettings` object if `_neo4j_settings` was initially `None`, ensuring that defaults are captured before patching.The file `testing_strategy_and_example.md` has been created with the testing strategy outline and the example integration test for `InitializationStage`.

The testing strategy covers:
1.  **Unit Tests:** Focusing on helper functions within stages, mocking external dependencies and Neo4j calls.
2.  **Integration Tests (Per Stage):** Verifying direct Neo4j interactions using Testcontainers for ephemeral Neo4j instances. This involves an Arrange-Act-Assert pattern where the database state and stage output are checked.
3.  **End-to-End Tests (`GoTProcessor`):** Testing the full processing pipeline with Neo4j-native stages, verifying the final database state and the overall output from `GoTProcessor`.
4.  **Dependency Management:** Noting the need for test-specific dependencies.
5.  **Acknowledgment on `neo4j_utils` Configuration:** Highlighting the need to manage the singleton-like nature of `neo4j_utils` for test isolation, with the example fixture demonstrating a patching approach.

The example integration test for `InitializationStage` (`tests/integration/stages/test_integration_initialization_stage.py`) includes:
*   A module-scoped `pytest` fixture (`neo4j_test_container_manager`) using `testcontainers.neo4j.Neo4jContainer` to manage an ephemeral Neo4j database.
*   This fixture patches the `neo4j_utils._neo4j_settings` object and manages the `neo4j_utils._driver` instance to ensure that during the tests, the application connects to the test container. It also includes teardown logic to restore the original settings.
*   An `autouse` fixture to apply this container management to all tests in the module.
*   Two test cases for `InitializationStage`:
    *   `test_initialization_stage_creates_new_root_node`: Verifies that a new root node ("n0") is created in Neo4j with correct properties and labels when no existing matching node is found.
    *   `test_initialization_stage_uses_existing_root_node`: Pre-populates Neo4j with a root node matching the query, then verifies that the stage uses this existing node and correctly updates its disciplinary tags based on operational parameters. It also checks that a new "n0" node is *not* created in this scenario.
*   Database cleaning (`MATCH (n) DETACH DELETE n`) is performed before tests that require a specific initial state.
*   Assertions are made against both the `StageOutput` metrics/context and the actual data in the test Neo4j database.

This fulfills the requirements of the subtask.
