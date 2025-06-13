import pytest
from unittest.mock import AsyncMock, MagicMock, patch # Using unittest.mock for AsyncMock
from typing import List, Dict, Any
from datetime import datetime as dt

from adaptive_graph_of_thoughts.config import Settings, PubMedConfig, GoogleScholarConfig, ExaSearchConfig, ASRGoTDefaultParams
from adaptive_graph_of_thoughts.domain.stages.stage_4_evidence import EvidenceStage
from adaptive_graph_of_thoughts.services.api_clients.pubmed_client import PubMedArticle, PubMedClientError, PubMedClient
from adaptive_graph_of_thoughts.services.api_clients.google_scholar_client import GoogleScholarArticle, GoogleScholarClientError, GoogleScholarClient
from adaptive_graph_of_thoughts.services.api_clients.exa_search_client import ExaArticleResult, ExaSearchClientError, ExaSearchClient
from adaptive_graph_of_thoughts.domain.models.graph_elements import StatisticalPower, NodeType
from adaptive_graph_of_thoughts.domain.models.common_types import GoTProcessorSessionData
from adaptive_graph_of_thoughts.domain.stages.stage_3_hypothesis import HypothesisStage


# --- Test Fixtures ---
@pytest.fixture
def mock_default_params() -> ASRGoTDefaultParams:
    """
    Creates default parameters for testing with evidence_max_iterations set to 1.
    
    Returns:
        An ASRGoTDefaultParams instance configured for a single evidence iteration.
    """
    return ASRGoTDefaultParams(evidence_max_iterations=1) # Default to 1 iteration for tests

@pytest.fixture
def mock_settings_all_clients(mock_default_params: ASRGoTDefaultParams) -> Settings:
    """
    Creates a Settings object with PubMed, Google Scholar, and ExaSearch clients configured for testing.
    
    Args:
        mock_default_params: Mocked default parameters for the asr_got configuration.
    
    Returns:
        A Settings instance with all external article search clients enabled and configured with test values.
    """
    return Settings(
        pubmed=PubMedConfig(base_url="https://pubmed.example.com", email="test@example.com"),
        google_scholar=GoogleScholarConfig(base_url="https://scholar.example.com", api_key="gs_key"),
        exa_search=ExaSearchConfig(base_url="https://exa.example.com", api_key="exa_key"),
        asr_got=MagicMock(default_parameters=mock_default_params) # Mock asr_got part
    )

@pytest.fixture
def mock_settings_pubmed_only(mock_default_params: ASRGoTDefaultParams) -> Settings:
    """
    Creates a Settings object configured with only the PubMed client enabled for testing.
    
    Args:
        mock_default_params: Mocked default parameters for the ASRGoT configuration.
    
    Returns:
        A Settings instance with PubMed configured and Google Scholar and ExaSearch disabled.
    """
    return Settings(
        pubmed=PubMedConfig(base_url="https://pubmed.example.com", email="test@example.com"),
        google_scholar=None,
        exa_search=None,
        asr_got=MagicMock(default_parameters=mock_default_params)
    )

@pytest.fixture
def evidence_stage_all_clients(mock_settings_all_clients: Settings) -> EvidenceStage:
    # Temporarily mock client instantiations within this fixture's scope
    """
    Creates an EvidenceStage instance with all external article search clients mocked for testing.
    
    The returned EvidenceStage has its PubMed, Google Scholar, and ExaSearch clients replaced with MagicMock instances, and their asynchronous search and close methods set to AsyncMock for use in asynchronous test scenarios.
    """
    with patch('adaptive_graph_of_thoughts.services.api_clients.pubmed_client.PubMedClient', new_callable=MagicMock) as MockPubMed, \
         patch('adaptive_graph_of_thoughts.services.api_clients.google_scholar_client.GoogleScholarClient', new_callable=MagicMock) as MockGS, \
         patch('adaptive_graph_of_thoughts.services.api_clients.exa_search_client.ExaSearchClient', new_callable=MagicMock) as MockExa:

        stage = EvidenceStage(settings=mock_settings_all_clients)
        # Replace mocked instances with AsyncMock capable ones for their methods
        if stage.pubmed_client: stage.pubmed_client.search_articles = AsyncMock()
        if stage.google_scholar_client: stage.google_scholar_client.search = AsyncMock()
        if stage.exa_client: stage.exa_client.search = AsyncMock()

        # Mock close methods as well
        if stage.pubmed_client: stage.pubmed_client.close = AsyncMock()
        if stage.google_scholar_client: stage.google_scholar_client.close = AsyncMock()
        if stage.exa_client: stage.exa_client.close = AsyncMock()
        return stage

@pytest.fixture
def evidence_stage_pubmed_only(mock_settings_pubmed_only: Settings) -> EvidenceStage:
    """
    Creates an EvidenceStage instance with only the PubMed client enabled and its methods mocked for asynchronous testing.
    
    The PubMed client's search and close methods are replaced with AsyncMock to simulate async behavior.
    """
    with patch('adaptive_graph_of_thoughts.services.api_clients.pubmed_client.PubMedClient', new_callable=MagicMock) as MockPubMed:
        stage = EvidenceStage(settings=mock_settings_pubmed_only)
        if stage.pubmed_client:
            stage.pubmed_client.search_articles = AsyncMock()
            stage.pubmed_client.close = AsyncMock()
        return stage


@pytest.fixture
def sample_hypothesis_data() -> Dict[str, Any]:
    """
    Provides a sample hypothesis dictionary for testing purposes.
    
    The returned dictionary includes an ID, label, JSON-encoded plan, disciplinary tags, and a confidence vector list.
    """
    return {
        "id": "hypo1",
        "label": "Hypothesis about AI in medicine",
        "plan_json": json.dumps({"query": "AI applications in medical diagnosis", "type": "literature_review"}),
        "metadata_disciplinary_tags": ["medicine", "ai"],
        "confidence_vector_list": [0.7, 0.6, 0.8, 0.5] # For _update_hypothesis_confidence
    }

@pytest.fixture
def mock_session_data(sample_hypothesis_data: Dict[str, Any]) -> GoTProcessorSessionData:
    """
    Creates a mock GoTProcessorSessionData instance with a test session ID and accumulated context referencing a hypothesis node.
    
    Args:
        sample_hypothesis_data: Sample hypothesis data (unused in this function).
    
    Returns:
        A GoTProcessorSessionData object with preset session ID and context for testing.
    """
    return GoTProcessorSessionData(
        session_id="test_session",
        accumulated_context={
            HypothesisStage.stage_name: {"hypothesis_node_ids": ["hypo1"]}
        }
    )

# --- Test Helper Functions ---
def create_mock_pubmed_articles(count=1) -> List[PubMedArticle]:
    """
    Creates a list of mock PubMedArticle instances for testing.
    
    Args:
        count: The number of mock articles to generate.
    
    Returns:
        A list of PubMedArticle objects with sample metadata.
    """
    return [
        PubMedArticle(pmid=f"pmid{i}", title=f"PubMed Article {i}", url=f"http://pubmed.com/pmid{i}",
                      authors=[f"Author {i}"], publication_date="2023", doi=f"doi_pm_{i}", abstract=f"Abstract {i}")
        for i in range(count)
    ]

def create_mock_gs_articles(count=1) -> List[GoogleScholarArticle]:
    """
    Creates a list of mock Google Scholar article objects for testing.
    
    Args:
        count: The number of mock articles to generate.
    
    Returns:
        A list of GoogleScholarArticle instances with sample data.
    """
    return [
        GoogleScholarArticle(title=f"GS Article {i}", link=f"http://gs.com/gs{i}", snippet=f"Snippet {i}",
                             authors=f"GS Author {i}A, GS Author {i}B", publication_info="Journal GS, 2023", cited_by_count=i*10)
        for i in range(count)
    ]

def create_mock_exa_results(count=1) -> List[ExaArticleResult]:
    """
    Creates a list of mock ExaArticleResult objects for testing.
    
    Args:
        count: The number of mock results to generate.
    
    Returns:
        A list of ExaArticleResult instances with sample data.
    """
    return [
        ExaArticleResult(id=f"exa{i}", title=f"Exa Result {i}", url=f"http://exa.com/exa{i}", score=0.8 + i*0.05,
                         author=f"Exa Author {i}", published_date="2023-01-0{i+1}", highlights=[f"Highlight {i}"])
        for i in range(count)
    ]


# --- Test Cases ---
def test_evidence_stage_initialization(mock_settings_all_clients: Settings, mock_settings_pubmed_only: Settings):
    """
    Verifies that EvidenceStage initializes the correct API clients based on provided settings.
    
    This test checks that all clients (PubMed, Google Scholar, ExaSearch) are instantiated when enabled in the settings, and only the PubMed client is instantiated when the others are not configured.
    """
    with patch('adaptive_graph_of_thoughts.services.api_clients.pubmed_client.PubMedClient') as MockPubMed, \
         patch('adaptive_graph_of_thoughts.services.api_clients.google_scholar_client.GoogleScholarClient') as MockGS, \
         patch('adaptive_graph_of_thoughts.services.api_clients.exa_search_client.ExaSearchClient') as MockExa:

        stage_all = EvidenceStage(settings=mock_settings_all_clients)
        MockPubMed.assert_called_once_with(mock_settings_all_clients)
        MockGS.assert_called_once_with(mock_settings_all_clients)
        MockExa.assert_called_once_with(mock_settings_all_clients)
        assert stage_all.pubmed_client is not None
        assert stage_all.google_scholar_client is not None
        assert stage_all.exa_client is not None

    with patch('adaptive_graph_of_thoughts.services.api_clients.pubmed_client.PubMedClient') as MockPubMed, \
         patch('adaptive_graph_of_thoughts.services.api_clients.google_scholar_client.GoogleScholarClient') as MockGS, \
         patch('adaptive_graph_of_thoughts.services.api_clients.exa_search_client.ExaSearchClient') as MockExa:

        stage_pubmed_only = EvidenceStage(settings=mock_settings_pubmed_only)
        MockPubMed.assert_called_once_with(mock_settings_pubmed_only)
        MockGS.assert_not_called()
        MockExa.assert_not_called()
        assert stage_pubmed_only.pubmed_client is not None
        assert stage_pubmed_only.google_scholar_client is None
        assert stage_pubmed_only.exa_client is None


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_all_clients(evidence_stage_all_clients: EvidenceStage, sample_hypothesis_data: Dict[str, Any]):
    """
    Tests that _execute_hypothesis_plan aggregates evidence from all enabled clients.
    
    Verifies that when all clients are enabled and return results, the method queries each client once, combines their outputs, and the resulting evidence list contains items from each source with expected fields.
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

    assert len(found_evidence) == 6 # 2 from each source
    # Check one item from each source for basic structure
    assert any(item["raw_source_data_type"] == "PubMedArticle" for item in found_evidence)
    assert any(item["raw_source_data_type"] == "GoogleScholarArticle" for item in found_evidence)
    assert any(item["raw_source_data_type"] == "ExaArticleResult" for item in found_evidence)

    # Example check for PubMed item details
    pubmed_item = next(item for item in found_evidence if item["raw_source_data_type"] == "PubMedArticle")
    assert "PubMed Article 0" in pubmed_item["content"]
    assert "Abstract 0" in pubmed_item["content"]
    assert pubmed_item["authors_list"] == ["Author 0"]
    assert pubmed_item["publication_date_str"] == "2023"
    assert pubmed_item["doi"] == "doi_pm_0"


@pytest.mark.asyncio
async def test_execute_hypothesis_plan_pubmed_only(evidence_stage_pubmed_only: EvidenceStage, sample_hypothesis_data: Dict[str, Any]):
    """
    Tests that _execute_hypothesis_plan returns only PubMed articles when only the PubMed client is enabled.
    
    Verifies that the PubMed client's search method is called, other clients are not used, and all returned evidence items are from PubMed.
    """
    stage = evidence_stage_pubmed_only
    stage.pubmed_client.search_articles.return_value = create_mock_pubmed_articles(3)

    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)

    stage.pubmed_client.search_articles.assert_called_once()
    assert stage.google_scholar_client is None # Or ensure its search is not called if it were mocked
    assert stage.exa_client is None

    assert len(found_evidence) == 3
    assert all(item["raw_source_data_type"] == "PubMedArticle" for item in found_evidence)

@pytest.mark.asyncio
async def test_execute_hypothesis_plan_client_error_handling(evidence_stage_all_clients: EvidenceStage, sample_hypothesis_data: Dict[str, Any], caplog):
    """
    Tests that _execute_hypothesis_plan handles PubMed client errors gracefully.
    
    Simulates a PubMed client error during evidence search and verifies that evidence from other clients is still collected, and the error is logged.
    """
    stage = evidence_stage_all_clients

    stage.pubmed_client.search_articles.side_effect = PubMedClientError("PubMed simulated error")
    stage.google_scholar_client.search.return_value = create_mock_gs_articles(1)
    stage.exa_client.search.return_value = create_mock_exa_results(1)

    found_evidence = await stage._execute_hypothesis_plan(sample_hypothesis_data)

    assert "Error querying PubMed" in caplog.text
    assert len(found_evidence) == 2 # GS and Exa results should still be there
    assert any(item["raw_source_data_type"] == "GoogleScholarArticle" for item in found_evidence)
    assert any(item["raw_source_data_type"] == "ExaArticleResult" for item in found_evidence)

@pytest.mark.asyncio
async def test_execute_hypothesis_plan_query_extraction(evidence_stage_pubmed_only: EvidenceStage, sample_hypothesis_data: Dict[str, Any]):
    """
    Tests that the hypothesis plan query extraction logic correctly prioritizes the query field in plan_json, falling back to the hypothesis label when necessary.
    
    Verifies that the PubMed client's search_articles method is called with the appropriate query string depending on the presence or absence of a query in plan_json or the plan_json field itself.
    """
    stage = evidence_stage_pubmed_only
    stage.pubmed_client.search_articles.return_value = [] # Content doesn't matter here

    # Test with query from plan_json
    await stage._execute_hypothesis_plan(sample_hypothesis_data)
    stage.pubmed_client.search_articles.assert_called_with(query="AI applications in medical diagnosis", max_results=2)

    stage.pubmed_client.search_articles.reset_mock()

    # Test with hypothesis label as query (plan_json missing query)
    hypothesis_no_plan_query = sample_hypothesis_data.copy()
    hypothesis_no_plan_query["plan_json"] = json.dumps({"type": "generic_review"}) # No query field
    await stage._execute_hypothesis_plan(hypothesis_no_plan_query)
    stage.pubmed_client.search_articles.assert_called_with(query=hypothesis_no_plan_query["label"], max_results=2)

    stage.pubmed_client.search_articles.reset_mock()

    # Test with no plan_json at all
    hypothesis_no_plan = sample_hypothesis_data.copy()
    del hypothesis_no_plan["plan_json"]
    await stage._execute_hypothesis_plan(hypothesis_no_plan)
    stage.pubmed_client.search_articles.assert_called_with(query=hypothesis_no_plan["label"], max_results=2)


@pytest.mark.asyncio
async def test_evidence_stage_execute_calls_close_clients(mock_settings_all_clients: Settings, mock_session_data: GoTProcessorSessionData, sample_hypothesis_data: Dict[str, Any]):

    # Need to patch clients at the source where they are imported by EvidenceStage
    with patch('adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.PubMedClient', new_callable=AsyncMock) as MockPubMed, \
         patch('adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.GoogleScholarClient', new_callable=AsyncMock) as MockGS, \
         patch('adaptive_graph_of_thoughts.domain.stages.stage_4_evidence.ExaSearchClient', new_callable=AsyncMock) as MockExa:

        # Configure mock instances that are created inside EvidenceStage.__init__
        mock_pubmed_instance = MockPubMed.return_value
        mock_gs_instance = MockGS.return_value
        mock_exa_instance = MockExa.return_value

        stage = EvidenceStage(settings=mock_settings_all_clients)

        # Mock methods of the stage itself
        stage._select_hypothesis_to_evaluate_from_neo4j = AsyncMock(side_effect=[sample_hypothesis_data, None]) # Process one hypothesis then stop
        stage._execute_hypothesis_plan = AsyncMock(return_value=[]) # No evidence found to simplify
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
async def test_evidence_stage_execute_no_hypotheses(evidence_stage_all_clients: EvidenceStage, mock_session_data: GoTProcessorSessionData):
    stage = evidence_stage_all_clients

    # Modify session data for this test
    mock_session_data.accumulated_context[HypothesisStage.stage_name]["hypothesis_node_ids"] = []

    # Spy on close_clients (it should still be called by finally)
    stage.close_clients = AsyncMock(wraps=stage.close_clients)

    output = await stage.execute(mock_session_data)

    assert "Evidence skipped: No hypotheses." in output.summary
    assert output.metrics["iterations_completed"] == 0
    assert output.next_stage_context_update[EvidenceStage.stage_name]["error"] == "No hypotheses found"

    stage.close_clients.assert_called_once()
```
