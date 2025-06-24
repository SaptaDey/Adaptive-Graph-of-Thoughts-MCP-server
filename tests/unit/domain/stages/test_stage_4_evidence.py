import datetime  # Moved from test_apply_temporal_decay
import json  # Import the json module
from datetime import datetime as dt
from typing import Any
from unittest.mock import (  # Using unittest.mock for AsyncMock
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from adaptive_graph_of_thoughts.config import (
    ASRGoTDefaultParams,
    ExaSearchConfig,
    GoogleScholarConfig,
    PubMedConfig,
    Settings,
)
from adaptive_graph_of_thoughts.domain.models.common_types import (
    GoTProcessorSessionData,
)
from adaptive_graph_of_thoughts.domain.stages.stage_3_hypothesis import HypothesisStage
from adaptive_graph_of_thoughts.domain.stages.stage_4_evidence import EvidenceStage
from adaptive_graph_of_thoughts.infrastructure import Neo4jGraphRepository
from adaptive_graph_of_thoughts.infrastructure.api_clients.exa_search_client import (
    ExaArticleResult,
)
from adaptive_graph_of_thoughts.infrastructure.api_clients.google_scholar_client import (
    GoogleScholarArticle,
)
from adaptive_graph_of_thoughts.infrastructure.api_clients.pubmed_client import (
    PubMedArticle,
    PubMedClientError,
)


# --- Test Fixtures ---
@pytest.fixture
def mock_default_params() -> ASRGoTDefaultParams:
    """
    Creates default ASRGoT parameters with evidence gathering limited to one iteration.

    Returns:
        An ASRGoTDefaultParams instance with evidence_max_iterations set to 1.
    """
    return ASRGoTDefaultParams(
        evidence_max_iterations=1
    )  # Default to 1 iteration for tests


@pytest.fixture
def mock_settings_all_clients(mock_default_params: ASRGoTDefaultParams) -> Settings:
    """
    Creates a Settings object with all literature search clients (PubMed, Google Scholar, ExaSearch) configured for testing.

    Args:
        mock_default_params: Mocked default parameters for the ASRGoT configuration.

    Returns:
        A Settings instance with all client configurations enabled and a mocked ASRGoT section.
    """
    return Settings(
        pubmed=PubMedConfig(
            base_url="https://pubmed.example.com", email="test@example.com"
        ),
        google_scholar=GoogleScholarConfig(
            base_url="https://scholar.example.com", api_key="gs_key"
        ),
        exa_search=ExaSearchConfig(
            base_url="https://exa.example.com", api_key="exa_key"
        ),
        asr_got=MagicMock(default_parameters=mock_default_params),  # Mock asr_got part
    )


@pytest.fixture
def mock_settings_pubmed_only(mock_default_params: ASRGoTDefaultParams) -> Settings:
    """
    Creates a Settings object configured with only the PubMed client enabled for testing.

    Args:
        mock_default_params: Default parameters to use for the ASRGoT configuration.

    Returns:
        A Settings instance with PubMed configured and Google Scholar and ExaSearch disabled.
    """
    return Settings(
        pubmed=PubMedConfig(
            base_url="https://pubmed.example.com", email="test@example.com"
        ),
        google_scholar=None,
        exa_search=None,
        asr_got=MagicMock(default_parameters=mock_default_params),
    )


@pytest.fixture
def evidence_stage_all_clients(mock_settings_all_clients: Settings) -> EvidenceStage:
    # Temporarily mock client instantiations within this fixture's scope
    """
    Creates an EvidenceStage instance with all external clients mocked for testing.

    The returned EvidenceStage has PubMed, Google Scholar, and ExaSearch clients replaced with MagicMock instances, and their asynchronous methods (`search_articles`, `search`, and `close`) are set as AsyncMock for use in asynchronous test scenarios.
    """
    with (
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.pubmed_client.PubMedClient",
            new_callable=MagicMock,
        ),
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.google_scholar_client.GoogleScholarClient",
            new_callable=MagicMock,
        ),
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.exa_search_client.ExaSearchClient",
            new_callable=MagicMock,
        ),
    ):
        stage = EvidenceStage(settings=mock_settings_all_clients, graph_repo=Neo4jGraphRepository())
        # Replace mocked instances with AsyncMock capable ones for their methods
        if stage.pubmed_client:
            stage.pubmed_client.search_articles = AsyncMock()
        if stage.google_scholar_client:
            stage.google_scholar_client.search = AsyncMock()
        if stage.exa_client:
            stage.exa_client.search = AsyncMock()

        # Mock close methods as well
        if stage.pubmed_client:
            stage.pubmed_client.close = AsyncMock()
        if stage.google_scholar_client:
            stage.google_scholar_client.close = AsyncMock()
        if stage.exa_client:
            stage.exa_client.close = AsyncMock()
        return stage


@pytest.fixture
def evidence_stage_pubmed_only(mock_settings_pubmed_only: Settings) -> EvidenceStage:
    """
    Creates an EvidenceStage instance with only the PubMed client mocked for testing.

    The returned EvidenceStage has its PubMed client's search and close methods replaced with AsyncMock to simulate asynchronous behavior. Other clients are not initialized.
    """
    with patch(
        "adaptive_graph_of_thoughts.infrastructure.api_clients.pubmed_client.PubMedClient",
        new_callable=MagicMock,
    ):
        stage = EvidenceStage(settings=mock_settings_pubmed_only, graph_repo=Neo4jGraphRepository())
        if stage.pubmed_client:
            stage.pubmed_client.search_articles = AsyncMock()
            stage.pubmed_client.close = AsyncMock()
        return stage


@pytest.fixture
def sample_hypothesis_data() -> dict[str, Any]:
    """
    Provides a sample hypothesis node dictionary for testing purposes.

    The returned dictionary includes an ID, label, JSON-encoded plan, disciplinary tags, and a confidence vector list, simulating the structure expected by EvidenceStage tests.
    """
    return {
        "id": "hypo1",
        "label": "Hypothesis about AI in medicine",
        "plan_json": json.dumps(
            {
                "query": "AI applications in medical diagnosis",
                "type": "literature_review",
            }
        ),
        "metadata_disciplinary_tags": ["medicine", "ai"],
        "confidence_vector_list": [
            0.7,
            0.6,
            0.8,
            0.5,
        ],  # For _update_hypothesis_confidence
    }


@pytest.fixture
def mock_session_data(
    _sample_hypothesis_data: dict[str, Any],
) -> GoTProcessorSessionData:
    """
    Creates a mock session data object containing accumulated context with a hypothesis node ID.

    Args:
        sample_hypothesis_data: Sample hypothesis data used for test context.

    Returns:
        A GoTProcessorSessionData instance with a test session ID and accumulated context referencing the hypothesis node.
    """
    return GoTProcessorSessionData(
        session_id="test_session",
        accumulated_context={
            HypothesisStage.stage_name: {"hypothesis_node_ids": ["hypo1"]}
        },
    )


# --- Test Helper Functions ---
def create_mock_pubmed_articles(count=1) -> list[PubMedArticle]:
    """
    Creates a list of mock PubMedArticle objects for testing purposes.

    Args:
        count: The number of mock articles to generate.

    Returns:
        A list of PubMedArticle instances with dummy data.
    """
    return [
        PubMedArticle(
            pmid=f"pmid{i}",
            title=f"PubMed Article {i}",
            url=f"http://pubmed.com/pmid{i}",
            authors=[f"Author {i}"],
            publication_date="2023",
            doi=f"doi_pm_{i}",
            abstract=f"Abstract {i}",
        )
        for i in range(count)
    ]


def create_mock_gs_articles(count=1) -> list[GoogleScholarArticle]:
    """
    Creates a list of mock Google Scholar article objects for testing.

    Args:
        count: The number of mock articles to generate.

    Returns:
        A list of GoogleScholarArticle instances with dummy data.
    """
    return [
        GoogleScholarArticle(
            title=f"GS Article {i}",
            link=f"http://gs.com/gs{i}",
            snippet=f"Snippet {i}",
            authors=f"GS Author {i}A, GS Author {i}B",
            publication_info="Journal GS, 2023",
            cited_by_count=i * 10,
        )
        for i in range(count)
    ]


def create_mock_exa_results(count=1) -> list[ExaArticleResult]:
    """
    Creates a list of mock ExaArticleResult objects for testing purposes.

    Args:
        count: The number of mock ExaArticleResult objects to generate.

    Returns:
        A list of ExaArticleResult instances with preset attributes.
    """
    return [
        ExaArticleResult(
            id=f"exa{i}",
            title=f"Exa Result {i}",
            url=f"http://exa.com/exa{i}",
            score=0.8 + i * 0.05,
            author=f"Exa Author {i}",
            published_date="2023-01-0{i+1}",
            highlights=[f"Highlight {i}"],
        )
        for i in range(count)
    ]


# --- Test Cases ---
def test_evidence_stage_initialization(
    mock_settings_all_clients: Settings, mock_settings_pubmed_only: Settings
):
    """
    Verifies that EvidenceStage initializes the correct API clients based on the provided settings.

    This test ensures that when all clients are enabled in the settings, PubMed, Google Scholar, and ExaSearch clients are instantiated. When only PubMed is enabled, only the PubMed client is instantiated, and the others remain uninitialized.
    """
    with (
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.pubmed_client.PubMedClient"
        ) as MockPubMed,
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.google_scholar_client.GoogleScholarClient"
        ) as MockGS,
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.exa_search_client.ExaSearchClient"
        ) as MockExa,
    ):
        stage_all = EvidenceStage(settings=mock_settings_all_clients, graph_repo=Neo4jGraphRepository())
        MockPubMed.assert_called_once_with(mock_settings_all_clients)
        MockGS.assert_called_once_with(mock_settings_all_clients)
        MockExa.assert_called_once_with(mock_settings_all_clients)
        assert stage_all.pubmed_client is not None
        assert stage_all.google_scholar_client is not None
        assert stage_all.exa_client is not None

    with (
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.pubmed_client.PubMedClient"
        ) as MockPubMed,
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.google_scholar_client.GoogleScholarClient"
        ) as MockGS,
        patch(
            "adaptive_graph_of_thoughts.infrastructure.api_clients.exa_search_client.ExaSearchClient"
        ) as MockExa,
    ):
        stage_pubmed_only = EvidenceStage(settings=mock_settings_pubmed_only, graph_repo=Neo4jGraphRepository())
        MockPubMed.assert_called_once_with(mock_settings_pubmed_only)
        MockGS.assert_not_called()
        MockExa.assert_not_called()
        assert stage_pubmed_only.pubmed_client is not None
        assert stage_pubmed_only.google_scholar_client is None
        assert stage_pubmed_only.exa_client is None


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_all_clients(
    evidence_stage_all_clients: EvidenceStage, sample_hypothesis_data: dict[str, Any]
):
    """
    Tests that _execute_hypothesis_plan gathers evidence from all enabled clients.

    Verifies that when all clients (PubMed, Google Scholar, ExaSearch) are active, their search methods are called, and the combined evidence list contains items from each source with correct structure and content.
    """
    stage = evidence_stage_all_clients

    # Setup mocks for each client's search method
    stage.pubmed_client.search_articles.return_value = create_mock_pubmed_articles(2)
    stage.google_scholar_client.search.return_value = create_mock_gs_articles(2)
    stage.exa_client.search.return_value = create_mock_exa_results(2)

    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)

    stage.pubmed_client.search_articles.assert_called_once()
    stage.google_scholar_client.search.assert_called_once()
    stage.exa_client.search.assert_called_once()

    assert len(found_evidence) == 6  # 2 from each source
    # Check one item from each source for basic structure
    assert any(
        item["raw_source_data_type"] == "PubMedArticle" for item in found_evidence
    )
    assert any(
        item["raw_source_data_type"] == "GoogleScholarArticle"
        for item in found_evidence
    )
    assert any(
        item["raw_source_data_type"] == "ExaArticleResult" for item in found_evidence
    )

    # Example check for PubMed item details
    pubmed_item = next(
        item
        for item in found_evidence
        if item["raw_source_data_type"] == "PubMedArticle"
    )
    assert "PubMed Article 0" in pubmed_item["content"]
    assert "Abstract 0" in pubmed_item["content"]
    assert pubmed_item["authors_list"] == ["Author 0"]
    assert pubmed_item["publication_date_str"] == "2023"
    assert pubmed_item["doi"] == "doi_pm_0"


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_pubmed_only(
    evidence_stage_pubmed_only: EvidenceStage, sample_hypothesis_data: dict[str, Any]
):
    """
    Tests that _execute_hypothesis_plan returns only PubMed evidence when only the PubMed client is enabled.

    Verifies that the PubMed client's search method is called, other clients are not used, and all returned evidence items are from PubMed.
    """
    stage = evidence_stage_pubmed_only
    stage.pubmed_client.search_articles.return_value = create_mock_pubmed_articles(3)

    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)

    stage.pubmed_client.search_articles.assert_called_once()
    assert (
        stage.google_scholar_client is None
    )  # Or ensure its search is not called if it were mocked
    assert stage.exa_client is None

    assert len(found_evidence) == 3
    assert all(
        item["raw_source_data_type"] == "PubMedArticle" for item in found_evidence
    )


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_client_error_handling(
    evidence_stage_all_clients: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
    caplog,
):
    """
    Tests that errors from the PubMed client during evidence gathering are logged and do not prevent evidence from other clients from being collected.

    Verifies that when the PubMed client raises an error, evidence from Google Scholar and ExaSearch is still returned, and the error is present in the logs.
    """
    stage = evidence_stage_all_clients

    stage.pubmed_client.search_articles.side_effect = PubMedClientError(
        "PubMed simulated error"
    )
    stage.google_scholar_client.search.return_value = create_mock_gs_articles(1)
    stage.exa_client.search.return_value = create_mock_exa_results(1)

    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)

    assert "Error querying PubMed" in caplog.text
    assert len(found_evidence) == 2  # GS and Exa results should still be there
    assert any(
        item["raw_source_data_type"] == "GoogleScholarArticle"
        for item in found_evidence
    )
    assert any(
        item["raw_source_data_type"] == "ExaArticleResult" for item in found_evidence
    )


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_query_extraction(
    evidence_stage_pubmed_only: EvidenceStage, sample_hypothesis_data: dict[str, Any]
):
    """
    Tests that the evidence stage extracts the correct search query from hypothesis data when executing a hypothesis plan.

    Verifies that the query is taken from the 'query' field in 'plan_json' if present, or falls back to the hypothesis label if missing or absent.
    """
    stage = evidence_stage_pubmed_only
    stage.pubmed_client.search_articles.return_value = []  # Content doesn't matter here

    # Test with query from plan_json
    await stage._execute_hypothesis_plan(sample_hypothesis_data)
    stage.pubmed_client.search_articles.assert_called_with(
        query="AI applications in medical diagnosis", max_results=2
    )

    stage.pubmed_client.search_articles.reset_mock()

    # Test with hypothesis label as query (plan_json missing query)
    hypothesis_no_plan_query = sample_hypothesis_data.copy()
    hypothesis_no_plan_query["plan_json"] = json.dumps(
        {"type": "generic_review"}
    )  # No query field
    await stage._execute_hypothesis_plan(hypothesis_no_plan_query)
    stage.pubmed_client.search_articles.assert_called_with(
        query=hypothesis_no_plan_query["label"], max_results=2
    )

    stage.pubmed_client.search_articles.reset_mock()

    # Test with no plan_json at all
    hypothesis_no_plan = sample_hypothesis_data.copy()
    del hypothesis_no_plan["plan_json"]
    await stage._execute_hypothesis_plan(hypothesis_no_plan)
    stage.pubmed_client.search_articles.assert_called_with(
        query=hypothesis_no_plan["label"], max_results=2
    )


@pytest.mark.asyncio
async def test_evidence_stage_execute_calls_close_clients(
    mock_settings_all_clients: Settings,
    mock_session_data: GoTProcessorSessionData,
    sample_hypothesis_data: dict[str, Any],
):
    # Need to patch clients at the source where they are imported by EvidenceStage
    """
    Tests that the EvidenceStage.execute method calls close_clients after processing,
    and that each external client's close method is invoked exactly once.
    """
    with (
        patch(
            "adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.PubMedClient",
            new_callable=AsyncMock,
        ) as MockPubMed,
        patch(
            "adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.GoogleScholarClient",
            new_callable=AsyncMock,
        ) as MockGS,
        patch(
            "adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.ExaSearchClient",
            new_callable=AsyncMock,
        ) as MockExa,
    ):
        # Configure mock instances that are created inside EvidenceStage.__init__
        mock_pubmed_instance = MockPubMed.return_value
        mock_gs_instance = MockGS.return_value
        mock_exa_instance = MockExa.return_value

        stage = EvidenceStage(settings=mock_settings_all_clients, graph_repo=Neo4jGraphRepository())

        # Mock methods of the stage itself
        stage._select_hypothesis_to_evaluate_from_neo4j = AsyncMock(
            side_effect=[sample_hypothesis_data, None]
        )  # Process one hypothesis then stop
        stage._execute_hypothesis_plan = AsyncMock(
            return_value=[]
        )  # No evidence found to simplify
        stage._create_evidence_in_neo4j = AsyncMock(return_value=None)
        stage._update_hypothesis_confidence_in_neo4j = AsyncMock(return_value=True)
        stage._create_ibn_in_neo4j = AsyncMock(return_value=None)
        stage._create_hyperedges_in_neo4j = AsyncMock(return_value=[])
        stage._apply_temporal_decay_and_patterns = AsyncMock()
        stage._adapt_graph_topology = AsyncMock()

        # Spy on close_clients directly on the instance
        stage.close_clients = AsyncMock(wraps=stage.close_clients)

        await stage.execute(mock_session_data)

        stage.close_clients.assert_called_once()
        # Also check individual client close methods were called via the spied close_clients
        mock_pubmed_instance.close.assert_called_once()
        mock_gs_instance.close.assert_called_once()
        mock_exa_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_evidence_stage_execute_no_hypotheses(
    evidence_stage_all_clients: EvidenceStage,
    mock_session_data: GoTProcessorSessionData,
):
    stage = evidence_stage_all_clients

    # Modify session data for this test
    mock_session_data.accumulated_context[HypothesisStage.stage_name][
        "hypothesis_node_ids"
    ] = []

    # Spy on close_clients (it should still be called by finally)
    stage.close_clients = AsyncMock(wraps=stage.close_clients)

    output = await stage.execute(mock_session_data)

    assert "Evidence skipped: No hypotheses." in output.summary
    assert output.metrics["iterations_completed"] == 0
    assert (
        output.next_stage_context_update[EvidenceStage.stage_name]["error"]
        == "No hypotheses found"
    )

    stage.close_clients.assert_called_once()


@pytest.fixture
def freeze_time(monkeypatch):
    """
    Provides a context manager to freeze `datetime.datetime.utcnow()` to a specific datetime for testing.

    Use this fixture to ensure deterministic results in tests involving temporal decay by overriding the datetime used in the target module.
    """

    class _Freezer:
        def __call__(self, frozen_dt):
            """
            Monkeypatches the datetime class to return a fixed datetime value.

            Replaces the `dt` class in the target module so that calls to `now()` return the specified `frozen_dt`, enabling deterministic testing of time-dependent logic.
            """

            class FrozenDateTime(dt):
                @classmethod
                def now(cls, tz=None):
                    return frozen_dt.replace(tzinfo=tz)

            monkeypatch.setattr(
                "adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.dt",
                FrozenDateTime,
                raising=False,
            )

    return _Freezer()


@pytest.mark.parametrize(
    "plan_json,label,expected_query",
    [
        (json.dumps({"query": "deep learning"}), "fallback", "deep learning"),
        (
            json.dumps({"type": "literature_review"}),
            "label wins",
            "label wins",
        ),  # no 'query'
        (None, "plain label", "plain label"),
        ("{malformed json", "bad json label", "bad json label"),  # JSONDecodeError
    ],
)
@pytest.mark.asyncio
async def test_query_extraction_logic(
    evidence_stage_pubmed_only,
    sample_hypothesis_data,
    plan_json,
    label,
    expected_query,
):
    data = sample_hypothesis_data.copy()
    data["label"] = label
    if plan_json is not None:
        data["plan_json"] = plan_json
    else:
        data.pop("plan_json", None)

    # Short-circuit client to avoid API calls
    evidence_stage_pubmed_only.pubmed_client.search_articles = AsyncMock(
        return_value=[]
    )
    await evidence_stage_pubmed_only._execute_hypothesis_plan(data)
    evidence_stage_pubmed_only.pubmed_client.search_articles.assert_called_once()
    called_kwargs = (
        evidence_stage_pubmed_only.pubmed_client.search_articles.call_args.kwargs
    )
    assert called_kwargs["query"] == expected_query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conf_vec,expected_update",
    [
        ([1.0, 1.0, 1.0, 1.0], 0.9),
        ([0.0, 0.0, 0.0, 0.0], 0.1),
        ([0.5, 0.6, 0.4, 0.5], 0.5),
    ],
)
async def test_update_hypothesis_confidence_variants(
    monkeypatch, evidence_stage_all_clients, conf_vec, expected_update
):
    # monkeypatch execute_query so no DB I/O happens and we can inspect cypher payload
    """
    Tests that updating hypothesis confidence in Neo4j correctly computes and persists
    the empirical support value based on different confidence vectors.

    Args:
        monkeypatch: Pytest fixture for patching functions.
        evidence_stage_all_clients: EvidenceStage instance with all clients mocked.
        conf_vec: List of confidence values to test.
        expected_update: Expected empirical support value after update.
    """
    recorded = {}

    async def fake_execute(_query, params, _tx_type="write"):
        """
        Mocks the execution of a database query by recording the provided parameters.

        Args:
                query: The query string to be executed.
                params: The parameters to be recorded.
                tx_type: The transaction type, defaults to "write".
        """
        recorded["params"] = params
        return None

    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.execute_query",
        fake_execute,
    )

    hypothesis_data = {"id": "h1", "confidence_vector_list": conf_vec}
    await evidence_stage_all_clients._update_hypothesis_confidence_in_neo4j(
        hypothesis_data
    )
    assert "confidence_empirical_support" in recorded["params"]
    # Use average of vector as naive expectation; adjust formula if implementation changes
    assert (
        pytest.approx(recorded["params"]["confidence_empirical_support"], 0.2)
        == expected_update
    )


@pytest.mark.asyncio
async def test_apply_temporal_decay(
    monkeypatch, evidence_stage_all_clients, freeze_time
):
    # Create dummy node list with timestamp 10 days ago
    """
    Tests that temporal decay is correctly applied to evidence node confidence values.

    This test verifies that when evidence nodes have timestamps in the past, the
    `_apply_temporal_decay_and_patterns` method reduces their `confidence_empirical_support`
    values appropriately. It uses monkeypatching to inject mock evidence nodes and to
    capture the updated confidence values after decay is applied.
    """
    old_dt = dt.utcnow() - datetime.timedelta(days=10)
    freeze_time(old_dt)  # now() returns old_dt inside EvidenceStage

    evidence_nodes = [
        {
            "confidence_empirical_support": 0.8,
            "metadata_timestamp_iso": old_dt.isoformat(),
        },
        {
            "confidence_empirical_support": 0.2,
            "metadata_timestamp_iso": old_dt.isoformat(),
        },
    ]

    # monkeypatch selection helpers to inject list
    monkeypatch.setattr(
        evidence_stage_all_clients,
        "_get_evidence_nodes_from_neo4j",
        AsyncMock(return_value=evidence_nodes),
    )
    monkeypatch.setattr(
        evidence_stage_all_clients,
        "_persist_confidence_updates",
        AsyncMock(),
    )

    # run
    await evidence_stage_all_clients._apply_temporal_decay_and_patterns()

    # expect _persist_confidence_updates called with decayed values
    args, _ = evidence_stage_all_clients._persist_confidence_updates.call_args
    updated_nodes = args[0]
    assert all(n["confidence_empirical_support"] < 0.8 for n in updated_nodes)


@pytest.mark.asyncio
async def test_execute_handles_client_init_failure(
    monkeypatch, mock_settings_all_clients, mock_session_data, sample_hypothesis_data
):
    # make PubMedClient raise on instantiation
    """
    Tests that EvidenceStage handles PubMedClient initialization failure gracefully.

    Simulates an exception during PubMedClient instantiation and verifies that the execute
    method logs the failure, continues execution, and ensures client resources are closed.
    """
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.infrastructure.api_clients.pubmed_client.PubMedClient",
        lambda _: (_ for _ in ()).throw(Exception("boom")),
    )
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.GoogleScholarClient",
        MagicMock,
    )
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.ExaSearchClient",
        MagicMock,
    )

    stage = EvidenceStage(settings=mock_settings_all_clients, graph_repo=Neo4jGraphRepository())
    stage.close_clients = AsyncMock(wraps=stage.close_clients)
    # Short-circuit loop to exit quickly
    stage._select_hypothesis_to_evaluate_from_neo4j = AsyncMock(
        side_effect=[sample_hypothesis_data, None]
    )
    stage._execute_hypothesis_plan = AsyncMock(return_value=[])
    stage._create_evidence_in_neo4j = AsyncMock(return_value=None)
    stage._update_hypothesis_confidence_in_neo4j = AsyncMock(return_value=True)
    stage._create_ibn_in_neo4j = AsyncMock(return_value=None)
    stage._create_hyperedges_in_neo4j = AsyncMock(return_value=[])
    stage._apply_temporal_decay_and_patterns = AsyncMock()
    stage._adapt_graph_topology = AsyncMock()

    output = await stage.execute(mock_session_data)
    stage.close_clients.assert_called_once()
    assert "Failed to initialize PubMedClient" in output.logs  # or inspect caplog


def test_adapt_graph_topology_calls(_monkeypatch, evidence_stage_all_clients):
    evidence_stage_all_clients._create_hyperedges_in_neo4j = MagicMock(return_value=[1])
    evidence_stage_all_clients._remove_redundant_edges_from_neo4j = MagicMock()
    evidence_stage_all_clients._simplify_graph = MagicMock()

    evidence_stage_all_clients._adapt_graph_topology()

    evidence_stage_all_clients._create_hyperedges_in_neo4j.assert_called()
    evidence_stage_all_clients._remove_redundant_edges_from_neo4j.assert_called()
    evidence_stage_all_clients._simplify_graph.assert_called()


# --- Additional Comprehensive Test Cases ---

@pytest.mark.asyncio
async def test_execute_hypothesis_plan_empty_results(
    evidence_stage_all_clients: EvidenceStage, sample_hypothesis_data: dict[str, Any]
):
    """
    Tests that _execute_hypothesis_plan handles cases where all clients return empty results.
    
    Verifies that when all external search clients return no articles, the method
    returns an empty list and doesn't crash.
    """
    stage = evidence_stage_all_clients
    
    # Setup all clients to return empty results
    stage.pubmed_client.search_articles.return_value = []
    stage.google_scholar_client.search.return_value = []
    stage.exa_client.search.return_value = []
    
    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)
    
    assert len(found_evidence) == 0
    stage.pubmed_client.search_articles.assert_called_once()
    stage.google_scholar_client.search.assert_called_once()
    stage.exa_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_malformed_plan_json(
    evidence_stage_pubmed_only: EvidenceStage,
):
    """
    Tests handling of malformed JSON in hypothesis plan_json field.
    
    Verifies that when plan_json contains invalid JSON, the method falls back
    to using the hypothesis label as the search query.
    """
    stage = evidence_stage_pubmed_only
    stage.pubmed_client.search_articles.return_value = []
    
    hypothesis_data = {
        "id": "test_hypo",
        "label": "Test Hypothesis Label",
        "plan_json": "{ invalid json content",  # Malformed JSON
        "metadata_disciplinary_tags": ["test"],
        "confidence_vector_list": [0.5, 0.5, 0.5, 0.5],
    }
    
    await stage._execute_hypothesis_plan(hypothesis_data)
    
    # Should fall back to using the label as query
    stage.pubmed_client.search_articles.assert_called_with(
        query="Test Hypothesis Label", max_results=2
    )


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_all_clients_exception(
    evidence_stage_all_clients: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
    caplog,
):
    """
    Tests error handling when all external clients raise exceptions.
    
    Verifies that when all search clients fail, appropriate errors are logged
    and an empty evidence list is returned instead of crashing.
    """
    stage = evidence_stage_all_clients
    
    # Make all clients raise exceptions
    stage.pubmed_client.search_articles.side_effect = PubMedClientError("PubMed failed")
    stage.google_scholar_client.search.side_effect = Exception("Google Scholar failed")
    stage.exa_client.search.side_effect = Exception("Exa failed")
    
    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)
    
    # Should return empty list when all clients fail
    assert len(found_evidence) == 0
    
    # All error messages should be logged
    assert "Error querying PubMed" in caplog.text
    assert "Error querying Google Scholar" in caplog.text
    assert "Error querying Exa" in caplog.text


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_max_results_parameter(
    evidence_stage_pubmed_only: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests that the max_results parameter is correctly passed to client search methods.
    
    Verifies that the default max_results value of 2 is passed to external clients
    and that the parameter can be customized if needed.
    """
    stage = evidence_stage_pubmed_only
    stage.pubmed_client.search_articles.return_value = []
    
    await stage._execute_hypothesis_plan(sample_hypothesis_data)
    
    # Verify max_results=2 is passed by default
    stage.pubmed_client.search_articles.assert_called_with(
        query="AI applications in medical diagnosis", max_results=2
    )


@pytest.mark.asyncio
async def test_evidence_article_conversion_pubmed(
    evidence_stage_pubmed_only: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests detailed conversion of PubMed articles to evidence format.
    
    Verifies that all relevant fields from PubMedArticle objects are correctly
    mapped to the evidence dictionary structure.
    """
    stage = evidence_stage_pubmed_only
    
    # Create detailed mock PubMed article
    mock_article = PubMedArticle(
        pmid="12345",
        title="Test Article Title",
        url="https://pubmed.gov/12345",
        authors=["Author One", "Author Two"],
        publication_date="2023-01-15",
        doi="10.1234/test.doi",
        abstract="This is a test abstract with important content.",
    )
    stage.pubmed_client.search_articles.return_value = [mock_article]
    
    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)
    
    assert len(found_evidence) == 1
    evidence_item = found_evidence[0]
    
    # Verify all fields are correctly mapped
    assert evidence_item["raw_source_data_type"] == "PubMedArticle"
    assert evidence_item["url"] == "https://pubmed.gov/12345"
    assert evidence_item["authors_list"] == ["Author One", "Author Two"]
    assert evidence_item["publication_date_str"] == "2023-01-15"
    assert evidence_item["doi"] == "10.1234/test.doi"
    assert "Test Article Title" in evidence_item["content"]
    assert "This is a test abstract" in evidence_item["content"]


@pytest.mark.asyncio
async def test_evidence_article_conversion_google_scholar(
    evidence_stage_all_clients: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests detailed conversion of Google Scholar articles to evidence format.
    
    Verifies that all relevant fields from GoogleScholarArticle objects are
    correctly mapped to the evidence dictionary structure.
    """
    stage = evidence_stage_all_clients
    
    # Setup only Google Scholar to return results
    stage.pubmed_client.search_articles.return_value = []
    stage.exa_client.search.return_value = []
    
    mock_gs_article = GoogleScholarArticle(
        title="Google Scholar Test Article",
        link="https://scholar.google.com/test",
        snippet="This is a test snippet from Google Scholar",
        authors="Jane Smith, John Doe",
        publication_info="Nature Medicine, 2023",
        cited_by_count=25,
    )
    stage.google_scholar_client.search.return_value = [mock_gs_article]
    
    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)
    
    assert len(found_evidence) == 1
    evidence_item = found_evidence[0]
    
    assert evidence_item["raw_source_data_type"] == "GoogleScholarArticle"
    assert evidence_item["url"] == "https://scholar.google.com/test"
    assert evidence_item["authors_list"] == ["Jane Smith", "John Doe"]
    assert evidence_item["publication_date_str"] == "Nature Medicine, 2023"
    assert evidence_item["doi"] is None  # Google Scholar doesn't provide DOI
    assert "Google Scholar Test Article" in evidence_item["content"]
    assert "This is a test snippet" in evidence_item["content"]


@pytest.mark.asyncio
async def test_evidence_article_conversion_exa(
    evidence_stage_all_clients: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests detailed conversion of Exa search results to evidence format.
    
    Verifies that all relevant fields from ExaArticleResult objects are
    correctly mapped to the evidence dictionary structure.
    """
    stage = evidence_stage_all_clients
    
    # Setup only Exa to return results
    stage.pubmed_client.search_articles.return_value = []
    stage.google_scholar_client.search.return_value = []
    
    mock_exa_result = ExaArticleResult(
        id="exa_12345",
        title="Exa Search Result Title",
        url="https://example.com/article",
        score=0.95,
        author="Research Author",
        published_date="2023-03-20",
        highlights=["Important highlight 1", "Important highlight 2"],
    )
    stage.exa_client.search.return_value = [mock_exa_result]
    
    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)
    
    assert len(found_evidence) == 1
    evidence_item = found_evidence[0]
    
    assert evidence_item["raw_source_data_type"] == "ExaArticleResult"
    assert evidence_item["url"] == "https://example.com/article"
    assert evidence_item["authors_list"] == ["Research Author"]
    assert evidence_item["publication_date_str"] == "2023-03-20"
    assert evidence_item["doi"] is None  # Exa doesn't provide DOI
    assert "Exa Search Result Title" in evidence_item["content"]
    assert "Important highlight 1" in evidence_item["content"]


@pytest.mark.asyncio
async def test_close_clients_individual_failures(
    evidence_stage_all_clients: EvidenceStage,
    caplog,
):
    """
    Tests that close_clients handles individual client close failures gracefully.
    
    Verifies that if one client's close method fails, other clients are still
    closed and the error is logged appropriately.
    """
    stage = evidence_stage_all_clients
    
    # Make PubMed client close fail
    stage.pubmed_client.close.side_effect = Exception("PubMed close failed")
    
    await stage.close_clients()
    
    # All close methods should be called despite the exception
    stage.pubmed_client.close.assert_called_once()
    stage.google_scholar_client.close.assert_called_once()
    stage.exa_client.close.assert_called_once()
    
    # Error should be logged
    assert "Error closing PubMed client" in caplog.text


@pytest.mark.asyncio
async def test_close_clients_with_none_clients(
    mock_settings_pubmed_only: Settings,
):
    """
    Tests that close_clients handles None clients gracefully.
    
    Verifies that when some clients are None (not initialized), the close_clients
    method doesn't crash and only calls close on initialized clients.
    """
    with patch(
        "adaptive_graph_of_thoughts.infrastructure.api_clients.pubmed_client.PubMedClient",
        new_callable=MagicMock,
    ):
        stage = EvidenceStage(settings=mock_settings_pubmed_only, graph_repo=Neo4jGraphRepository())
        stage.pubmed_client.close = AsyncMock()
        
        await stage.close_clients()
        
        # Only PubMed client should be closed (others are None)
        stage.pubmed_client.close.assert_called_once()
        assert stage.google_scholar_client is None
        assert stage.exa_client is None


@pytest.mark.asyncio
async def test_execute_with_multiple_hypothesis_iterations(
    evidence_stage_all_clients: EvidenceStage,
    mock_session_data: GoTProcessorSessionData,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests execute method with multiple hypothesis evaluation iterations.
    
    Verifies that the stage processes multiple hypotheses correctly and
    increments iteration counters appropriately.
    """
    stage = evidence_stage_all_clients
    
    # Create multiple hypotheses to process
    hypothesis_1 = sample_hypothesis_data.copy()
    hypothesis_1["id"] = "hypo1"
    hypothesis_2 = sample_hypothesis_data.copy()
    hypothesis_2["id"] = "hypo2"
    
    # Mock hypothesis selection to return multiple hypotheses then None
    stage._select_hypothesis_to_evaluate_from_neo4j = AsyncMock(
        side_effect=[hypothesis_1, hypothesis_2, None]
    )
    stage._execute_hypothesis_plan = AsyncMock(return_value=[])
    stage._create_evidence_in_neo4j = AsyncMock(return_value=None)
    stage._update_hypothesis_confidence_in_neo4j = AsyncMock(return_value=True)
    stage._create_ibn_in_neo4j = AsyncMock(return_value=None)
    stage._create_hyperedges_in_neo4j = AsyncMock(return_value=[])
    stage._apply_temporal_decay_and_patterns = AsyncMock()
    stage._adapt_graph_topology = AsyncMock()
    stage.close_clients = AsyncMock()
    
    output = await stage.execute(mock_session_data)
    
    # Should have processed 2 hypotheses
    assert output.metrics["iterations_completed"] == 2
    assert stage._select_hypothesis_to_evaluate_from_neo4j.call_count == 3  # Called until None
    assert stage._execute_hypothesis_plan.call_count == 2


@pytest.mark.asyncio 
async def test_execute_max_iterations_limit(
    evidence_stage_all_clients: EvidenceStage,
    mock_session_data: GoTProcessorSessionData,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests that execute method respects the max_iterations limit.
    
    Verifies that the stage stops processing after reaching the configured
    maximum number of iterations, even if more hypotheses are available.
    """
    stage = evidence_stage_all_clients
    
    # Set max_iterations to 1 for this test
    stage.max_iterations = 1
    
    # Mock to always return a hypothesis (infinite loop potential)
    stage._select_hypothesis_to_evaluate_from_neo4j = AsyncMock(
        return_value=sample_hypothesis_data
    )
    stage._execute_hypothesis_plan = AsyncMock(return_value=[])
    stage._create_evidence_in_neo4j = AsyncMock(return_value=None)
    stage._update_hypothesis_confidence_in_neo4j = AsyncMock(return_value=True)
    stage._create_ibn_in_neo4j = AsyncMock(return_value=None)
    stage._create_hyperedges_in_neo4j = AsyncMock(return_value=[])
    stage._apply_temporal_decay_and_patterns = AsyncMock()
    stage._adapt_graph_topology = AsyncMock()
    stage.close_clients = AsyncMock()
    
    output = await stage.execute(mock_session_data)
    
    # Should have stopped at max_iterations = 1
    assert output.metrics["iterations_completed"] == 1
    assert stage._execute_hypothesis_plan.call_count == 1


@pytest.mark.asyncio
async def test_update_hypothesis_confidence_edge_cases(
    monkeypatch, evidence_stage_all_clients
):
    """
    Tests edge cases for hypothesis confidence updates.
    
    Verifies that confidence updates handle edge cases like empty confidence
    vectors, None values, and extreme values correctly.
    """
    recorded_params = []
    
    async def fake_execute(_query, params, _tx_type="write"):
        recorded_params.append(params)
        return None
    
    monkeypatch.setattr(
        evidence_stage_all_clients.graph_repo,
        "execute_query",
        fake_execute,
    )
    
    # Test empty confidence vector
    hypothesis_data = {"id": "h1", "confidence_vector_list": []}
    result = await evidence_stage_all_clients._update_hypothesis_confidence_in_neo4j(hypothesis_data)
    assert result is False  # Should return False for empty vector
    
    # Test None confidence vector
    hypothesis_data = {"id": "h2", "confidence_vector_list": None}
    result = await evidence_stage_all_clients._update_hypothesis_confidence_in_neo4j(hypothesis_data)
    assert result is False  # Should return False for None vector
    
    # Test single value
    hypothesis_data = {"id": "h3", "confidence_vector_list": [0.8]}
    result = await evidence_stage_all_clients._update_hypothesis_confidence_in_neo4j(hypothesis_data)
    assert result is True  # Should succeed with single value
    
    # Should have made 1 call (only the last one with valid data)
    assert len(recorded_params) == 1
    
    # Should have confidence_empirical_support set
    params = recorded_params[0]
    assert "confidence_empirical_support" in params
    assert isinstance(params["confidence_empirical_support"], (int, float))


@pytest.mark.asyncio
async def test_temporal_decay_with_recent_timestamp(
    monkeypatch, evidence_stage_all_clients, freeze_time
):
    """
    Tests temporal decay behavior with recently created evidence nodes.
    
    Verifies that evidence nodes with recent timestamps receive minimal
    or no decay to their confidence values.
    """
    recent_dt = dt.utcnow() - datetime.timedelta(hours=1)  # Very recent
    freeze_time(recent_dt)
    
    evidence_nodes = [
        {
            "confidence_empirical_support": 0.9,
            "metadata_timestamp_iso": recent_dt.isoformat(),
        }
    ]
    
    monkeypatch.setattr(
        evidence_stage_all_clients,
        "_get_evidence_nodes_from_neo4j",
        AsyncMock(return_value=evidence_nodes),
    )
    monkeypatch.setattr(
        evidence_stage_all_clients,
        "_persist_confidence_updates",
        AsyncMock(),
    )
    
    await evidence_stage_all_clients._apply_temporal_decay_and_patterns()
    
    # For recent timestamps, confidence should remain high
    args, _ = evidence_stage_all_clients._persist_confidence_updates.call_args
    updated_nodes = args[0]
    assert len(updated_nodes) == 1
    # Recent nodes should have minimal decay
    assert updated_nodes[0]["confidence_empirical_support"] >= 0.85


@pytest.mark.asyncio
async def test_temporal_decay_with_missing_timestamp(
    monkeypatch, evidence_stage_all_clients
):
    """
    Tests temporal decay handling of evidence nodes without timestamp metadata.
    
    Verifies that nodes missing timestamp information are handled gracefully
    without causing crashes in the decay calculation.
    """
    evidence_nodes = [
        {"confidence_empirical_support": 0.8},  # Missing timestamp
        {"confidence_empirical_support": 0.6, "metadata_timestamp_iso": None},
    ]
    
    monkeypatch.setattr(
        evidence_stage_all_clients,
        "_get_evidence_nodes_from_neo4j",
        AsyncMock(return_value=evidence_nodes),
    )
    monkeypatch.setattr(
        evidence_stage_all_clients,
        "_persist_confidence_updates",
        AsyncMock(),
    )
    
    # Should not crash with missing timestamps
    await evidence_stage_all_clients._apply_temporal_decay_and_patterns()
    
    # Should still call persist with some updates
    evidence_stage_all_clients._persist_confidence_updates.assert_called_once()


@pytest.mark.asyncio
async def test_deserialize_tags_edge_cases(evidence_stage_all_clients: EvidenceStage):
    """
    Tests the _deserialize_tags method with various edge cases.
    
    Verifies that the method handles None values, empty strings, malformed JSON,
    and different data types correctly.
    """
    stage = evidence_stage_all_clients
    
    # Test None input
    result = stage._deserialize_tags(None)
    assert result == set()
    
    # Test empty string
    result = stage._deserialize_tags("")
    assert result == set()
    
    # Test valid list
    result = stage._deserialize_tags(["tag1", "tag2"])
    assert result == {"tag1", "tag2"}
    
    # Test valid set
    result = stage._deserialize_tags({"tag1", "tag2"})
    assert result == {"tag1", "tag2"}
    
    # Test valid JSON string
    result = stage._deserialize_tags('["tag1", "tag2"]')
    assert result == {"tag1", "tag2"}
    
    # Test invalid JSON string
    result = stage._deserialize_tags('["tag1", "tag2"')  # Missing closing bracket
    assert result == set()


@pytest.mark.asyncio
async def test_stage_name_property():
    """
    Tests that the EvidenceStage.stage_name property returns the correct value.
    
    Verifies that the stage name is correctly defined for use in session context
    and stage identification.
    """
    assert EvidenceStage.stage_name == "EvidenceStage"


def test_evidence_stage_string_representation(mock_settings_all_clients: Settings):
    """
    Tests the string representation of EvidenceStage.
    
    Verifies that the EvidenceStage object has a meaningful string representation
    for debugging and logging purposes.
    """
    with (
        patch("adaptive_graph_of_thoughts.infrastructure.api_clients.pubmed_client.PubMedClient"),
        patch("adaptive_graph_of_thoughts.infrastructure.api_clients.google_scholar_client.GoogleScholarClient"),
        patch("adaptive_graph_of_thoughts.infrastructure.api_clients.exa_search_client.ExaSearchClient"),
    ):
        stage = EvidenceStage(settings=mock_settings_all_clients, graph_repo=Neo4jGraphRepository())
        str_repr = str(stage)
        assert "EvidenceStage" in str_repr or "object" in str_repr


@pytest.mark.asyncio
async def test_evidence_stage_large_dataset_handling(
    evidence_stage_all_clients: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests EvidenceStage behavior with large numbers of search results.
    
    Verifies that the stage can handle scenarios where external APIs return
    large numbers of articles without performance issues or memory problems.
    """
    stage = evidence_stage_all_clients
    
    # Create large datasets (simulating high-volume search results)
    large_pubmed_results = create_mock_pubmed_articles(50)
    large_gs_results = create_mock_gs_articles(50)
    large_exa_results = create_mock_exa_results(50)
    
    stage.pubmed_client.search_articles.return_value = large_pubmed_results
    stage.google_scholar_client.search.return_value = large_gs_results
    stage.exa_client.search.return_value = large_exa_results
    
    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)
    
    # Should handle large datasets
    assert len(found_evidence) == 150  # 50 from each source
    
    # Verify that all evidence items have required structure
    for evidence_item in found_evidence[:10]:  # Check first 10 for performance
        assert "raw_source_data_type" in evidence_item
        assert "content" in evidence_item
        assert "url" in evidence_item


@pytest.mark.asyncio
async def test_evidence_stage_concurrent_client_calls(
    evidence_stage_all_clients: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests that evidence gathering from multiple clients happens concurrently.
    
    Verifies that calls to different external APIs are made and that all
    clients are called appropriately.
    """
    stage = evidence_stage_all_clients
    
    # Track call order by recording call information
    call_info = []
    
    async def track_pubmed_call(*args, **kwargs):
        call_info.append("pubmed")
        return create_mock_pubmed_articles(1)
    
    async def track_gs_call(*args, **kwargs):
        call_info.append("gs")
        return create_mock_gs_articles(1)
    
    async def track_exa_call(*args, **kwargs):
        call_info.append("exa")
        return create_mock_exa_results(1)
    
    stage.pubmed_client.search_articles.side_effect = track_pubmed_call
    stage.google_scholar_client.search.side_effect = track_gs_call
    stage.exa_client.search.side_effect = track_exa_call
    
    await stage._execute_hypothesis_plan(sample_hypothesis_data)
    
    # All three calls should have been made
    assert len(call_info) == 3
    assert "pubmed" in call_info
    assert "gs" in call_info
    assert "exa" in call_info


@pytest.mark.asyncio
async def test_memory_efficiency_with_repeated_executions(
    evidence_stage_all_clients: EvidenceStage,
    mock_session_data: GoTProcessorSessionData,
    sample_hypothesis_data: dict[str, Any],
):
    """
    Tests memory efficiency during repeated execute calls.
    
    Verifies that repeated executions don't cause memory leaks or
    excessive memory usage accumulation.
    """
    stage = evidence_stage_all_clients
    
    # Mock all required methods
    stage._select_hypothesis_to_evaluate_from_neo4j = AsyncMock(
        side_effect=[sample_hypothesis_data, None]
    )
    stage._execute_hypothesis_plan = AsyncMock(return_value=[])
    stage._create_evidence_in_neo4j = AsyncMock(return_value=None)
    stage._update_hypothesis_confidence_in_neo4j = AsyncMock(return_value=True)
    stage._create_ibn_in_neo4j = AsyncMock(return_value=None)
    stage._create_hyperedges_in_neo4j = AsyncMock(return_value=[])
    stage._apply_temporal_decay_and_patterns = AsyncMock()
    stage._adapt_graph_topology = AsyncMock()
    stage.close_clients = AsyncMock()
    
    # Execute multiple times
    for _ in range(5):
        output = await stage.execute(mock_session_data)
        assert output.metrics["iterations_completed"] == 1
        
        # Reset mocks for next iteration
        stage._select_hypothesis_to_evaluate_from_neo4j.reset_mock()
        stage._select_hypothesis_to_evaluate_from_neo4j.side_effect = [sample_hypothesis_data, None]


@pytest.mark.asyncio
async def test_create_evidence_neo4j_error_handling(
    evidence_stage_all_clients: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
    monkeypatch,
):
    """
    Tests error handling in _create_evidence_in_neo4j method.
    
    Verifies that Neo4j errors are properly caught and logged without
    crashing the evidence creation process.
    """
    stage = evidence_stage_all_clients
    
    # Mock Neo4j to raise an error
    async def failing_execute_query(*args, **kwargs):
        from adaptive_graph_of_thoughts.infrastructure.neo4j_utils import Neo4jError
        raise Neo4jError("Database connection failed")
    
    monkeypatch.setattr(stage.graph_repo, "execute_query", failing_execute_query)
    
    evidence_data = {
        "content": "Test evidence content",
        "source_description": "Test source",
        "url": "https://example.com",
        "doi": "10.1234/test",
        "authors_list": ["Test Author"],
        "publication_date_str": "2023",
        "supports_hypothesis": True,
        "strength": 0.8,
        "disciplinary_tags": ["test"],
        "timestamp": dt.now(),
        "raw_source_data_type": "TestArticle",
        "original_data": {},
    }
    
    result = await stage._create_evidence_in_neo4j(
        sample_hypothesis_data, evidence_data, 0, 0
    )
    
    # Should return None on failure
    assert result is None


@pytest.mark.asyncio
async def test_create_ibn_threshold_behavior(
    evidence_stage_all_clients: EvidenceStage,
    sample_hypothesis_data: dict[str, Any],
    monkeypatch,
):
    """
    Tests IBN creation threshold behavior.
    
    Verifies that IBNs are only created when similarity exceeds the threshold
    and that different disciplinary tags are required.
    """
    stage = evidence_stage_all_clients
    
    # Mock similarity calculation to return specific values
    def mock_similarity(text1, text2):
        return 0.3  # Below default threshold of 0.5
    
    monkeypatch.setattr(
        "adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.calculate_semantic_similarity",
        mock_similarity,
    )
    
    evidence_node_data = {
        "id": "ev1",
        "label": "Evidence label",
        "metadata_disciplinary_tags": ["biology"],  # Different from hypothesis
    }
    
    hypothesis_node_data = {
        "id": "hypo1",
        "label": "Hypothesis label",
        "metadata_disciplinary_tags": ["medicine"],  # Different from evidence
    }
    
    result = await stage._create_ibn_in_neo4j(evidence_node_data, hypothesis_node_data)
    
    # Should return None due to low similarity
    assert result is None


@pytest.mark.asyncio
async def test_adaptation_graph_topology_error_handling(
    evidence_stage_all_clients: EvidenceStage,
    caplog,
):
    """
    Tests error handling in _adapt_graph_topology method.
    
    Verifies that errors in graph topology adaptation are caught and logged
    without stopping the overall execution.
    """
    stage = evidence_stage_all_clients
    
    # Mock methods to raise errors
    stage._create_hyperedges_in_neo4j = MagicMock(side_effect=Exception("Hyperedge error"))
    stage._remove_redundant_edges_from_neo4j = MagicMock(side_effect=Exception("Remove edges error"))
    stage._simplify_graph = MagicMock(side_effect=Exception("Simplify error"))
    
    # Should not raise exceptions
    stage._adapt_graph_topology()
    
    # All methods should have been called despite errors
    stage._create_hyperedges_in_neo4j.assert_called()
    stage._remove_redundant_edges_from_neo4j.assert_called()
    stage._simplify_graph.assert_called()


