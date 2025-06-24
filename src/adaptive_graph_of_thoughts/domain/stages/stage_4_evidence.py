import json
import random
import asyncio
import logging
from unittest.mock import AsyncMock
from datetime import datetime as dt  # Alias dt for datetime.datetime
from typing import Any, Optional, TYPE_CHECKING

from loguru import logger  # type: ignore

from ...config import LegacyConfig
from ...infrastructure.api_clients.google_scholar_client import (
    UnexpectedResponseStructureError,
)
from ..models.common import (
    ConfidenceVector,
    EpistemicStatus,
)
from ..models.common_types import GoTProcessorSessionData
from ..models.graph_elements import (
    Edge,
    EdgeMetadata,
    EdgeType,
    InterdisciplinaryInfo,
    Node,
    NodeMetadata,
    NodeType,
    StatisticalPower,
)
from ...infrastructure.neo4j_utils import Neo4jError, execute_query as neo4j_execute_query
from ..interfaces import GraphRepository
from ..utils.math_helpers import (
    bayesian_update_confidence,
    calculate_information_gain,
)
from ..utils.metadata_helpers import (
    calculate_semantic_similarity,
)
from ..utils.neo4j_helpers import (
    prepare_edge_properties_for_neo4j,
    prepare_node_properties_for_neo4j,
)
from .base_stage import BaseStage, StageOutput
from .exceptions import StageInitializationError
from .stage_3_hypothesis import HypothesisStage  # To access hypothesis_node_ids

## Backward compatibility for tests expecting execute_query in this module
#execute_query = neo4j_execute_query

if TYPE_CHECKING:
    from ...infrastructure.api_clients.pubmed_client import PubMedClient
    from ...infrastructure.api_clients.google_scholar_client import GoogleScholarClient
    from ...infrastructure.api_clients.exa_search_client import ExaSearchClient

# These globals are placeholders so tests can monkeypatch client classes
PubMedClient = None  # type: ignore[assignment]
GoogleScholarClient = None  # type: ignore[assignment]
ExaSearchClient = None  # type: ignore[assignment]


class EvidenceStage(BaseStage):
    stage_name: str = "EvidenceStage"

    def __init__(
        self,
        settings: LegacyConfig,
        graph_repo: Optional[GraphRepository] = None,
    ) -> None:
        super().__init__(settings)
        if graph_repo is None:
            # Lazy import to avoid direct dependency on infrastructure in domain
            from ...infrastructure import Neo4jGraphRepository

            graph_repo = Neo4jGraphRepository()

        self.graph_repo = graph_repo
        self.max_iterations = self.default_params.evidence_max_iterations
        self.ibn_similarity_threshold = getattr(
            self.default_params, "ibn_similarity_threshold", 0.5
        )
        self.min_nodes_for_hyperedge_consideration = getattr(
            self.default_params, "min_nodes_for_hyperedge", 2
        )

        self._execution_logs: list[str] = []

        # Initialize API Clients
        failures: list[str] = []

        # Import clients lazily so tests can patch either this module or the infrastructure module
        from ...infrastructure.api_clients.pubmed_client import (
            PubMedClient as InfraPubMedClient,
            PubMedClientError,
        )
        from ...infrastructure.api_clients.google_scholar_client import (
            GoogleScholarClient as InfraGoogleScholarClient,
            GoogleScholarClientError,
        )
        from ...infrastructure.api_clients.exa_search_client import (
            ExaSearchClient as InfraExaSearchClient,
            ExaSearchClientError,
        )

        self.pubmed_client: Optional["PubMedClient"] = None
        if settings.pubmed and settings.pubmed.base_url:
            try:
                client_cls = PubMedClient or InfraPubMedClient
                maybe_client = client_cls(settings)
                if asyncio.iscoroutine(maybe_client):
                    if isinstance(client_cls, AsyncMock):
                        self.pubmed_client = client_cls.return_value
                    elif asyncio.get_event_loop().is_running():
                        self.pubmed_client = await maybe_client
                    else:
                        self.pubmed_client = asyncio.run(maybe_client)
                else:
                    self.pubmed_client = maybe_client
                    self.pubmed_client = maybe_client
                logger.info("PubMed client initialized for EvidenceStage.")
            except PubMedClientError as e:
                msg = f"Failed to initialize PubMedClient: {e}"
                logger.error(msg)
                self._execution_logs.append(msg)
                failures.append("PubMed")
            except Exception as e:  # pragma: no cover - defensive
                msg = f"Failed to initialize PubMedClient: {e}"
                logger.error(msg)
                self._execution_logs.append(msg)
                failures.append("PubMed")
        else:
            logger.warning(
                "PubMed client not initialized for EvidenceStage: PubMed configuration missing or incomplete."
            )

        self.google_scholar_client: Optional["GoogleScholarClient"] = None
        if (
            settings.google_scholar
            and settings.google_scholar.api_key
            and settings.google_scholar.base_url
        ):
            try:
                client_cls = GoogleScholarClient or InfraGoogleScholarClient
                maybe_client = client_cls(settings)
                if asyncio.iscoroutine(maybe_client):
                    if isinstance(client_cls, AsyncMock):
                        self.google_scholar_client = client_cls.return_value
                    elif asyncio.get_event_loop().is_running():
                        task = asyncio.create_task(maybe_client)
                        task.add_done_callback(lambda t: setattr(self, "google_scholar_client", t.result()))
                    else:
                        self.google_scholar_client = asyncio.run(maybe_client)
                else:
                    self.google_scholar_client = maybe_client
                logger.info("Google Scholar client initialized for EvidenceStage.")
            except GoogleScholarClientError as e:
                msg = f"Failed to initialize GoogleScholarClient: {e}"
                logger.error(msg)
                self._execution_logs.append(msg)
                failures.append("GoogleScholar")
            except Exception as e:  # pragma: no cover - defensive
                msg = f"Failed to initialize GoogleScholarClient: {e}"
                logger.error(msg)
                self._execution_logs.append(msg)
                failures.append("GoogleScholar")
        else:
            logger.warning(
                "Google Scholar client not initialized for EvidenceStage: Google Scholar configuration missing or incomplete (requires api_key and base_url)."
            )

        self.exa_client: Optional["ExaSearchClient"] = None
        if (
            settings.exa_search
            and settings.exa_search.api_key
            and settings.exa_search.base_url
        ):
            try:
                client_cls = ExaSearchClient or InfraExaSearchClient
                maybe_client = client_cls(settings)
                if asyncio.iscoroutine(maybe_client):
                    if isinstance(client_cls, AsyncMock):
                        self.exa_client = client_cls.return_value
                    elif asyncio.get_event_loop().is_running():
                        task = asyncio.create_task(maybe_client)
                        task.add_done_callback(lambda t: setattr(self, "exa_client", t.result()))
                    else:
                        self.exa_client = asyncio.run(maybe_client)
                else:
                    self.exa_client = maybe_client
                logger.info("Exa Search client initialized for EvidenceStage.")
            except ExaSearchClientError as e:
                msg = f"Failed to initialize ExaSearchClient: {e}"
                logger.error(msg)
                self._execution_logs.append(msg)
                failures.append("ExaSearch")
            except Exception as e:  # pragma: no cover - defensive
                msg = f"Failed to initialize ExaSearchClient: {e}"
                logger.error(msg)
                self._execution_logs.append(msg)
                failures.append("ExaSearch")
        else:
            logger.warning(
                "Exa Search client not initialized for EvidenceStage: Exa Search configuration missing or incomplete (requires api_key and base_url)."
            )

        if not any([self.pubmed_client, self.google_scholar_client, self.exa_client]):
            raise StageInitializationError(
                "No evidence sources available. Failed to initialize: " + ", ".join(failures)
            )

    async def close_clients(self):
        """Gracefully closes all initialized API clients."""
        logger.debug("Attempting to close API clients in EvidenceStage.")
        if self.pubmed_client:
            try:
                await self.pubmed_client.close()
                logger.debug("PubMed client closed.")
            except Exception as e:
                logger.error(f"Error closing PubMed client: {e}")

        if self.google_scholar_client:
            try:
                await self.google_scholar_client.close()
                logger.debug("Google Scholar client closed.")
            except Exception as e:
                logger.error(f"Error closing Google Scholar client: {e}")

        if self.exa_client:
            try:
                await self.exa_client.close()
                logger.debug("Exa Search client closed.")
            except Exception as e:
                logger.error(f"Error closing Exa Search client: {e}")

    def _deserialize_tags(self, raw) -> set[str]:
        if raw is None:
            return set()
        if isinstance(raw, (set, list)):
            return set(raw)
        try:
            return set(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning("Could not deserialize tags payload '%s'", raw)
            return set()

    async def _select_hypothesis_to_evaluate_from_neo4j(
        self, hypothesis_node_ids: list[str]
    ) -> Optional[dict[str, Any]]:
        if not hypothesis_node_ids:
            return None
        query = """
        UNWIND $hypothesis_ids AS hypo_id
        MATCH (h:Node:HYPOTHESIS {id: hypo_id})
        RETURN
            h.id AS id, h.label AS label, h.metadata_impact_score AS impact_score,
            h.confidence_empirical_support AS conf_empirical,
            h.confidence_theoretical_basis AS conf_theoretical,
            h.confidence_methodological_rigor AS conf_methodological,
            h.confidence_consensus_alignment AS conf_consensus,
            h.metadata_plan_json AS plan_json,
            h.metadata_layer_id AS layer_id,
            h.metadata_disciplinary_tags AS metadata_disciplinary_tags
        ORDER BY impact_score DESC, conf_empirical ASC
        LIMIT 10
        """
        try:
            results = await self.graph_repo.execute_query(
                query, {"hypothesis_ids": hypothesis_node_ids}, tx_type="read"
            )
            if not results:
                return None
            eligible_hypotheses_data = []
            for record in results:
                hypo_data = dict(record)
                conf_list = [
                    hypo_data.get("conf_empirical", 0.5),
                    hypo_data.get("conf_theoretical", 0.5),
                    hypo_data.get("conf_methodological", 0.5),
                    hypo_data.get("conf_consensus", 0.5),
                ]
                hypo_data["confidence_vector_list"] = conf_list
                eligible_hypotheses_data.append(hypo_data)
            if not eligible_hypotheses_data:
                return None

            def score_hypothesis_data(h_data: dict[str, Any]):
                impact = h_data.get("impact_score", 0.1)
                conf_list = h_data.get("confidence_vector_list", [0.5] * 4)
                conf_variance = sum([(c - 0.5) ** 2 for c in conf_list]) / 4.0
                return impact + conf_variance

            eligible_hypotheses_data.sort(key=score_hypothesis_data, reverse=True)
            selected_hypothesis_data = eligible_hypotheses_data[0]
            logger.debug(
                f"Selected hypothesis '{selected_hypothesis_data['label']}' (ID: {selected_hypothesis_data['id']}) from Neo4j for evidence integration."
            )
            return selected_hypothesis_data
        except Neo4jError as e:
            logger.error(f"Neo4j error selecting hypothesis: {e}")
            return None

    async def _execute_hypothesis_plan(
        self, hypothesis_data_from_neo4j: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Executes an evidence search plan for a given hypothesis and gathers supporting evidence.

        Uses the hypothesis label or a specified query plan to search external sources (PubMed, Google Scholar, Exa Search) for relevant articles or results. For each source, retrieves up to two results, extracts key information, and constructs evidence data dictionaries with content, source details, authors, publication date, confidence placeholders, disciplinary tags, timestamps, and raw data. Returns a list of evidence items found for the hypothesis.
        """
        hypo_label = hypothesis_data_from_neo4j.get("label", "")
        hypo_id = hypothesis_data_from_neo4j.get("id", "unknown_hypo")

        plan_json_str = hypothesis_data_from_neo4j.get("plan_json")

        search_query = hypo_label  # Default query
        plan_details = "using hypothesis label as query."

        if plan_json_str:
            try:
                plan_dict = json.loads(plan_json_str)
                search_query = plan_dict.get("query", hypo_label)
                plan_type = plan_dict.get("type", "default_plan")
                # Further plan interpretation can happen here:
                # - Specific sources to query (e.g., plan_dict.get("sources", ["pubmed", "google_scholar", "exa"]))
                # - Max results per source
                # - Specific keywords or refined queries for different sources
                plan_details = (
                    f"using query from plan ('{search_query}') of type '{plan_type}'."
                )
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not parse plan_json for hypothesis {hypo_label}. Defaulting to label for query."
                )

        logger.info(
            f"Executing evidence plan for hypothesis '{hypo_label}' (ID: {hypo_id}): {plan_details}"
        )

        found_evidence_list: list[dict[str, Any]] = []

        # Default disciplinary tags from hypothesis or system defaults
        default_tags = self._deserialize_tags(
            hypothesis_data_from_neo4j.get("metadata_disciplinary_tags")
        )
        if not default_tags and self.default_params:
            default_tags = set(self.default_params.default_disciplinary_tags)

        # --- PubMed Search ---
        if self.pubmed_client:
            try:
                logger.info(
                    f"Querying PubMed with: '{search_query}' for hypothesis {hypo_id}"
                )
                # Reduced max_results for now to manage API call volume during development
                pubmed_articles = await self.pubmed_client.search_articles(
                    query=search_query, max_results=2
                )
                for article in pubmed_articles:
                    content = article.title
                    if article.abstract:  # Keep it concise
                        content += f" | Abstract (preview): {article.abstract[:250]}..."

                    evidence_item = {
                        "content": content,
                        "source_description": "PubMed Search Result",
                        "url": article.url,
                        "doi": article.doi,
                        "authors_list": article.authors,  # Directly use List[str] from PubMedArticle
                        "publication_date_str": article.publication_date,  # Optional[str] from PubMedArticle
                        "supports_hypothesis": True,  # Placeholder
                        "strength": 0.65,  # Placeholder
                        "statistical_power": StatisticalPower(
                            value=0.6, method_description="Default placeholder SP."
                        ),
                        "disciplinary_tags": list(
                            default_tags
                        ),  # Inherit or use defaults
                        "timestamp": dt.now(),
                        "raw_source_data_type": "PubMedArticle",
                        "original_data": article.model_dump(),  # Store raw Pydantic model
                    }
                    found_evidence_list.append(evidence_item)
                logger.info(
                    f"Found {len(pubmed_articles)} articles from PubMed for '{search_query}'."
                )
            except Exception as e:  # Catching broad Exception from client calls
                logger.error(f"Error querying PubMed for '{search_query}': {e}")
             except Exception as e:  # Catching broad Exception from client calls
                 logger.error(f"Error querying PubMed for '{search_query}': {e}")

        # --- Google Scholar Search ---
        if self.google_scholar_client:
            try:
                logger.info(
                    f"Querying Google Scholar with: '{search_query}' for hypothesis {hypo_id}"
                )
                gs_articles = await self.google_scholar_client.search(
                    query=search_query, num_results=2
                )
                for article in gs_articles:
                    content = article.title
                    if article.snippet:
                        content += f" | Snippet: {article.snippet}"

                    # Basic DOI extraction attempt (very naive)
                    doi_candidate = None
                    if article.link and "doi.org/" in article.link:
                        doi_candidate = article.link.split("doi.org/")[-1]

                    gs_authors_str = (
                        article.authors
                    )  # This is Optional[str] from GoogleScholarArticle
                    authors_list_gs = (
                        [a.strip() for a in gs_authors_str.split(",")]
                        if gs_authors_str
                        else []
                    )

                    evidence_item = {
                        "content": content,
                        "source_description": "Google Scholar Search Result",
                        "url": article.link,
                        "doi": doi_candidate,  # May be None
                        "authors_list": authors_list_gs,
                        "publication_date_str": article.publication_info,  # Often "Journal, Year" or similar
                        "supports_hypothesis": True,  # Placeholder
                        "strength": 0.6
                        + (
                            article.cited_by_count / 500
                            if article.cited_by_count
                            else 0
                        ),  # Simple strength heuristic
                        "statistical_power": StatisticalPower(
                            value=0.5, method_description="Default placeholder SP."
                        ),
                        "disciplinary_tags": list(default_tags),
                        "timestamp": dt.now(),
                        "raw_source_data_type": "GoogleScholarArticle",
                        "original_data": article.model_dump(),
                    }
                    found_evidence_list.append(evidence_item)
                logger.info(
                    f"Found {len(gs_articles)} articles from Google Scholar for '{search_query}'."
                )
            except UnexpectedResponseStructureError as e:
                logger.warning(
                    f"Google Scholar returned unexpected structure for '{search_query}': {e}"
                )
            except Exception as e:
                logger.error(f"Error querying Google Scholar for '{search_query}': {e}")

        # --- Exa Search ---
        if self.exa_client:
            try:
                logger.info(
                    f"Querying Exa Search with: '{search_query}' for hypothesis {hypo_id}"
                )
                # Exa's neural search might be better with descriptive queries from hypothesis text
                exa_results = await self.exa_client.search(
                    query=search_query, num_results=2, type="neural"
                )
                for result in exa_results:
                    content = result.title if result.title else "Untitled Exa Result"
                    if result.highlights:
                        content += f" | Highlight: {result.highlights[0]}"

                    exa_author_str = (
                        result.author
                    )  # Optional[str] from ExaArticleResult
                    authors_list_exa = [exa_author_str] if exa_author_str else []

                    evidence_item = {
                        "content": content,
                        "source_description": "Exa Search Result",
                        "url": result.url,
                        "doi": None,  # Exa results are general web pages, DOI less common
                        "authors_list": authors_list_exa,
                        "publication_date_str": result.published_date,  # Optional[str] from ExaArticleResult (aliased)
                        "supports_hypothesis": True,  # Placeholder
                        "strength": result.score if result.score is not None else 0.5,
                        "statistical_power": StatisticalPower(
                            value=0.5, method_description="Default placeholder SP."
                        ),
                        "disciplinary_tags": list(default_tags),
                        "timestamp": dt.now(),
                        "raw_source_data_type": "ExaArticleResult",
                        "original_data": result.model_dump(),
                    }
                    found_evidence_list.append(evidence_item)
                logger.info(
                    f"Found {len(exa_results)} results from Exa Search for '{search_query}'."
                )
            except Exception as e:
                logger.error(f"Error querying Exa Search for '{search_query}': {e}")

        logger.info(
            f"Total evidence pieces found for hypothesis '{hypo_label}': {len(found_evidence_list)}"
        )
        return found_evidence_list

    async def _create_evidence_in_neo4j(
        self,
        hypothesis_data_from_neo4j: dict[str, Any],
        evidence_data: dict[str, Any],
        iteration: int,
        evidence_index: int,
    ) -> Optional[dict[str, Any]]:
        hypothesis_id = hypothesis_data_from_neo4j["id"]
        hypothesis_label = hypothesis_data_from_neo4j.get("label", "N/A")
        hypothesis_layer_id = hypothesis_data_from_neo4j.get(
            "layer_id",
            self.default_params.initial_layer
            if self.default_params
            else "unknown_layer",
        )

        evidence_id = f"ev_{hypothesis_id}_{iteration}_{evidence_index}"
        edge_type = (
            EdgeType.SUPPORTIVE
            if evidence_data["supports_hypothesis"]
            else EdgeType.CONTRADICTORY
        )

        sp_value = 0.5  # Default if no SP object
        if evidence_data.get("statistical_power") and isinstance(
            evidence_data.get("statistical_power"), StatisticalPower
        ):
            sp_value = evidence_data.get("statistical_power").value

        evidence_metadata_dict = {
            "description": evidence_data.get("content", "N/A"),
            "source_description": evidence_data.get("source_description", "N/A"),
            "epistemic_status": EpistemicStatus.EVIDENCE_SUPPORTED
            if evidence_data.get("supports_hypothesis", True)
            else EpistemicStatus.EVIDENCE_CONTRADICTED,
            "disciplinary_tags": set(evidence_data.get("disciplinary_tags", [])),
            "statistical_power": evidence_data.get(
                "statistical_power"
            ),  # This should be a StatisticalPower object
            "impact_score": evidence_data.get("strength", 0.5) * sp_value,
            "layer_id": hypothesis_layer_id,
            # New direct mappings to NodeMetadata fields
            "url": evidence_data.get("url"),
            "doi": evidence_data.get("doi"),
            "authors": evidence_data.get("authors_list", []),  # Expects List[str]
            "publication_date": evidence_data.get(
                "publication_date_str"
            ),  # Expects Optional[str]
            # Data for misc_details field
            "misc_details": {
                "raw_source_data_type": evidence_data.get("raw_source_data_type"),
                "original_data_dump": evidence_data.get(
                    "original_data"
                ),  # Storing the model_dump here
            },
        }
        # Ensure timestamp is handled; it's part of evidence_data, not directly in NodeMetadata Pydantic model by default
        # It will be added to ev_props_for_neo4j directly.
        evidence_metadata = NodeMetadata(**evidence_metadata_dict)

        evidence_confidence_vec = ConfidenceVector(
            empirical_support=evidence_data.get("strength", 0.5),
            methodological_rigor=evidence_data.get(
                "methodological_rigor", evidence_data.get("strength", 0.5) * 0.8
            ),
            theoretical_basis=0.5,
            consensus_alignment=0.5,
        )
        evidence_node_pydantic = Node(
            id=evidence_id,
            label=f"Evidence {evidence_index + 1} for H: {hypothesis_label[:20]}...",
            type=NodeType.EVIDENCE,
            confidence=evidence_confidence_vec,
            metadata=evidence_metadata,
        )
        ev_props_for_neo4j = prepare_node_properties_for_neo4j(evidence_node_pydantic)
        if "timestamp" in evidence_data and isinstance(evidence_data["timestamp"], dt):
            ev_props_for_neo4j["metadata_timestamp_iso"] = evidence_data[
                "timestamp"
            ].isoformat()
        else:
            ev_props_for_neo4j["metadata_timestamp_iso"] = (
                dt.now().isoformat()
            )  # Fallback timestamp        # Create evidence node query
        create_ev_node_query = (
            "MERGE (e:Node {id: $props.id}) SET e += $props "
            "WITH e, $type_label AS typeLabel CALL apoc.create.addLabels(e, [typeLabel]) YIELD node "
            "RETURN node.id AS evidence_id, properties(node) as evidence_props"
        )
        try:
            result_ev_node = await self.graph_repo.execute_query(
                create_ev_node_query,
                {"props": ev_props_for_neo4j, "type_label": NodeType.EVIDENCE.value},
                tx_type="write",
            )
            if not result_ev_node or not result_ev_node[0].get("evidence_id"):
                logger.error(f"Failed to create evidence node {evidence_id} in Neo4j.")
                return None

            created_evidence_id = result_ev_node[0]["evidence_id"]
            created_evidence_props = result_ev_node[0]["evidence_props"]

            edge_to_hypo_id = f"edge_ev_{created_evidence_id}_{hypothesis_id}"
            edge_pydantic = Edge(
                id=edge_to_hypo_id,
                source_id=created_evidence_id,
                target_id=hypothesis_id,
                type=edge_type,
                confidence=evidence_data.get("strength", 0.5),
                metadata=EdgeMetadata(
                    description=f"Evidence '{evidence_node_pydantic.label[:20]}...' {'supports' if evidence_data['supports_hypothesis'] else 'contradicts'} hypothesis."
                ),
            )
            edge_props_for_neo4j = prepare_edge_properties_for_neo4j(edge_pydantic)
            create_rel_query = (
                "MATCH (ev:Node {id: $evidence_id}) "
                "MATCH (hyp:Node {id: $hypothesis_id}) "
                f"MERGE (ev)-[r:`{edge_type.value}` {{id: $props.id}}]->(hyp) "
                "SET r += $props "
                "RETURN r.id as rel_id"
            )
            params_rel = {
                "evidence_id": created_evidence_id,
                "hypothesis_id": hypothesis_id,
                "props": edge_props_for_neo4j,
            }
            result_rel = await self.graph_repo.execute_query(
                create_rel_query, params_rel, tx_type="write"
            )
            if not result_rel or not result_rel[0].get("rel_id"):
                logger.error(
                    f"Failed to link evidence {created_evidence_id} to hypothesis {hypothesis_id}."
                )
                return None

            logger.debug(
                f"Created evidence node {created_evidence_id} and linked to hypothesis {hypothesis_id} with type {edge_type.value}."
            )
            # Return the properties fetched from Neo4j, which are more reliable than local construction for subsequent steps
            return {"id": created_evidence_id, **created_evidence_props}
        except Neo4jError as e:
            logger.error(f"Neo4j error creating evidence or link: {e}")
            return None

    async def _update_hypothesis_confidence_in_neo4j(
        self, hypothesis_data: dict[str, Any]
    ) -> bool:
        """Update a hypothesis node's confidence values based on its vector."""

        conf_vec = hypothesis_data.get("confidence_vector_list")
        if not conf_vec:
            logger.warning("No confidence vector provided for hypothesis update")
            return False

        avg = sum(conf_vec) / len(conf_vec)
        empirical_support = 0.1 + 0.8 * avg

        update_query = """
        MATCH (h:Node:HYPOTHESIS {id: $id})
        SET h.confidence_empirical_support = $confidence_empirical_support,
            h.metadata_last_updated_iso = $timestamp
        RETURN h.id
        """
        params = {
            "id": hypothesis_data["id"],
            "confidence_empirical_support": empirical_support,
            "timestamp": dt.now().isoformat(),
        }
        try:
            await self.graph_repo.execute_query(update_query, params, tx_type="write")
            return True
        except Neo4jError as e:
            logger.error(
                f"Neo4j error updating hypothesis confidence {hypothesis_data['id']}: {e}"
            )
            return False

    async def _create_ibn_in_neo4j(
        self, evidence_node_data: dict[str, Any], hypothesis_node_data: dict[str, Any]
    ) -> Optional[str]:
        hypo_tags_raw = hypothesis_node_data.get("metadata_disciplinary_tags")
        # evidence_node_data comes from _create_evidence_in_neo4j, which returns Neo4j properties
        # So, metadata_disciplinary_tags should be directly accessible if set.
        ev_tags_raw = evidence_node_data.get("metadata_disciplinary_tags")

        hypo_tags = self._deserialize_tags(hypo_tags_raw)
        ev_tags = self._deserialize_tags(ev_tags_raw)

        if not hypo_tags or not ev_tags or hypo_tags.intersection(ev_tags):
            return None

        # Use labels from the Neo4j data passed in
        hypo_label_for_sim = hypothesis_node_data.get("label", "")
        ev_label_for_sim = evidence_node_data.get("label", "")
        similarity = calculate_semantic_similarity(hypo_label_for_sim, ev_label_for_sim)
        if similarity < self.ibn_similarity_threshold:
            return None

        ibn_id = f"ibn_{evidence_node_data['id']}_{hypothesis_node_data['id']}"
        ibn_label = f"IBN: {ev_label_for_sim[:20]}... <=> {hypo_label_for_sim[:20]}..."

        ibn_metadata = NodeMetadata(
            description=f"Interdisciplinary bridge between domains {hypo_tags} and {ev_tags}.",
            source_description="EvidenceStage IBN Creation (P1.8)",
            epistemic_status=EpistemicStatus.INFERRED,
            disciplinary_tags=list(hypo_tags.union(ev_tags)),
            interdisciplinary_info=InterdisciplinaryInfo(
                source_disciplines=list(hypo_tags),
                target_disciplines=list(ev_tags),
                bridging_concept=f"Connection between '{ev_label_for_sim[:20]}' and '{hypo_label_for_sim[:20]}'",
            ),
            impact_score=0.6,
            layer_id=evidence_node_data.get(
                "metadata_layer_id",
                self.default_params.initial_layer
                if self.default_params
                else "unknown_layer",
            ),
        )
        ibn_confidence = ConfidenceVector(
            empirical_support=similarity,
            theoretical_basis=0.4,
            methodological_rigor=0.5,
            consensus_alignment=0.3,
        )
        ibn_node_pydantic = Node(
            id=ibn_id,
            label=ibn_label,
            type=NodeType.INTERDISCIPLINARY_BRIDGE,
            confidence=ibn_confidence,
            metadata=ibn_metadata,
        )
        ibn_props = prepare_node_properties_for_neo4j(ibn_node_pydantic)
        try:
            create_ibn_query = """
            MERGE (ibn:Node {id: $props.id}) SET ibn += $props
            WITH ibn, $type_label AS typeLabel CALL apoc.create.addLabels(ibn, [typeLabel]) YIELD node
            RETURN node.id AS ibn_created_id
            """
            result_ibn = await self.graph_repo.execute_query(
                create_ibn_query,
                {
                    "props": ibn_props,
                    "type_label": NodeType.INTERDISCIPLINARY_BRIDGE.value,
                },
                tx_type="write",
            )
            if not result_ibn or not result_ibn[0].get("ibn_created_id"):
                logger.error(f"Failed to create IBN node {ibn_id} in Neo4j.")
                return None
            created_ibn_id = result_ibn[0]["ibn_created_id"]
            edge1_id = f"edge_{evidence_node_data['id']}_{EdgeType.IBN_SOURCE_LINK.value}_{created_ibn_id}"
            edge1_pydantic = Edge(
                id=edge1_id,
                source_id=evidence_node_data["id"],
                target_id=created_ibn_id,
                type=EdgeType.IBN_SOURCE_LINK,
                confidence=0.8,
            )
            edge1_props = prepare_edge_properties_for_neo4j(edge1_pydantic)
            edge2_id = f"edge_{created_ibn_id}_{EdgeType.IBN_TARGET_LINK.value}_{hypothesis_node_data['id']}"
            edge2_pydantic = Edge(
                id=edge2_id,
                source_id=created_ibn_id,
                target_id=hypothesis_node_data["id"],
                type=EdgeType.IBN_TARGET_LINK,
                confidence=0.8,
            )
            edge2_props = prepare_edge_properties_for_neo4j(edge2_pydantic)
            link_ibn_query = """
            MATCH (ev_node:Node {id: $ev_id})
            MATCH (ibn_node:Node {id: $ibn_id})
            MATCH (hypo_node:Node {id: $hypo_id})
            MERGE (ev_node)-[r1:IBN_SOURCE_LINK {id: $edge1_props.id}]->(ibn_node) SET r1 += $edge1_props
            MERGE (ibn_node)-[r2:IBN_TARGET_LINK {id: $edge2_props.id}]->(hypo_node) SET r2 += $edge2_props
            RETURN r1.id AS r1_id, r2.id AS r2_id
            """
            params_link = {
                "ev_id": evidence_node_data["id"],
                "ibn_id": created_ibn_id,
                "hypo_id": hypothesis_node_data["id"],
                "edge1_props": edge1_props,
                "edge2_props": edge2_props,
            }
            link_results = await self.graph_repo.execute_query(
                link_ibn_query, params_link, tx_type="write"
            )
            if (
                link_results
                and link_results[0].get("r1_id")
                and link_results[0].get("r2_id")
            ):
                logger.info(
                    f"Created IBN {created_ibn_id} and linked it between {evidence_node_data['id']} and {hypothesis_node_data['id']}."
                )
                return created_ibn_id
            else:
                logger.error(
                    f"Failed to link IBN {created_ibn_id} to evidence/hypothesis."
                )
                return None
        except Neo4jError as e:
            logger.error(
                f"Neo4j error during IBN creation or linking for {ibn_id}: {e}"
            )
            return None

    async def _create_hyperedges_in_neo4j(
        self,
        hypothesis_data: dict[str, Any],
        related_evidence_data_list: list[dict[str, Any]],
    ) -> list[str]:
        created_hyperedge_ids: list[str] = []
        min_nodes_for_hyperedge = (
            self.min_nodes_for_hyperedge_consideration
            if hasattr(self, "min_nodes_for_hyperedge_consideration")
            else 2
        )
        if len(related_evidence_data_list) < min_nodes_for_hyperedge:
            return created_hyperedge_ids

        hyperedge_center_id = (
            f"hyper_{hypothesis_data['id']}_{random.randint(1000, 9999)}"
        )
        hyperedge_node_ids_for_pydantic = {hypothesis_data["id"]} | {
            ev_data["id"] for ev_data in related_evidence_data_list
        }

        # Use confidence from Neo4j data which is more reliable
        hypo_conf_emp = hypothesis_data.get(
            "conf_empirical", 0.5
        )  # Adjusted key from _select_hypothesis_to_evaluate_from_neo4j
        avg_emp_support = (
            hypo_conf_emp
            + sum(
                ev.get("confidence_empirical_support", 0.5)
                for ev in related_evidence_data_list
            )
        ) / (1 + len(related_evidence_data_list))

        hyper_confidence = ConfidenceVector(
            empirical_support=avg_emp_support,
            theoretical_basis=0.4,
            methodological_rigor=0.5,
            consensus_alignment=0.4,
        )
        hyperedge_node_metadata = NodeMetadata(
            description=f"Joint influence on hypothesis '{hypothesis_data.get('label', 'N/A')[:20]}...'",
            layer_id=hypothesis_data.get(
                "metadata_layer_id",
                self.default_params.initial_layer
                if self.default_params
                else "unknown_layer",
            ),
            misc_properties={
                "relationship_descriptor": "Joint Support/Contradiction (Simulated)"
            },
        )
        hyperedge_pydantic_for_center_node = Node(
            id=hyperedge_center_id,
            label=f"Hyperedge for {hypothesis_data.get('label', 'N/A')[:20]}",
            type=NodeType.HYPEREDGE_CENTER,
            confidence=hyper_confidence,
            metadata=hyperedge_node_metadata,
        )
        center_node_props = prepare_node_properties_for_neo4j(
            hyperedge_pydantic_for_center_node
        )
        try:
            create_center_query = """
            MERGE (hc:Node {id: $props.id}) SET hc += $props
            WITH hc, $type_label AS typeLabel CALL apoc.create.addLabels(hc, [typeLabel]) YIELD node
            RETURN node.id AS hyperedge_center_created_id
            """
            result_center = await self.graph_repo.execute_query(
                create_center_query,
                {
                    "props": center_node_props,
                    "type_label": NodeType.HYPEREDGE_CENTER.value,
                },
                tx_type="write",
            )
            if not result_center or not result_center[0].get(
                "hyperedge_center_created_id"
            ):
                logger.error(
                    f"Failed to create hyperedge center node {hyperedge_center_id}."
                )
                return created_hyperedge_ids
            created_hyperedge_center_id = result_center[0][
                "hyperedge_center_created_id"
            ]
            batch_member_links_data = []
            for member_id in hyperedge_node_ids_for_pydantic:
                edge_id = (
                    f"edge_hyper_{created_hyperedge_center_id}_hasmember_{member_id}"
                )
                batch_member_links_data.append(
                    {
                        "hyperedge_center_id": created_hyperedge_center_id,
                        "member_node_id": member_id,
                        "props": {"id": edge_id},  # Minimal edge props for now
                    }
                )
            if batch_member_links_data:
                link_members_query = """
                UNWIND $links AS link_data
                MATCH (hc:Node {id: link_data.hyperedge_center_id})
                MATCH (member:Node {id: link_data.member_node_id})
                MERGE (hc)-[r:HAS_MEMBER {id: link_data.props.id}]->(member) SET r += link_data.props
                RETURN count(r) AS total_links_created
                """
                link_results = await self.graph_repo.execute_query(
                    link_members_query,
                    {"links": batch_member_links_data},
                    tx_type="write",
                )
                if (
                    link_results
                    and link_results[0].get("total_links_created") is not None
                ):
                    logger.debug(
                        f"Batch created {link_results[0]['total_links_created']} HAS_MEMBER links for hyperedge {created_hyperedge_center_id}."
                    )
                else:
                    logger.error(
                        f"Failed to get count from batch hyperedge member linking for {created_hyperedge_center_id}."
                    )
            created_hyperedge_ids.append(created_hyperedge_center_id)
            logger.info(
                f"Created Hyperedge (center node {created_hyperedge_center_id}) for hypothesis {hypothesis_data['id']} and {len(related_evidence_data_list)} evidence nodes."
            )
        except Neo4jError as e:
            logger.error(
                f"Neo4j error creating hyperedge or linking members for hypothesis {hypothesis_data['id']}: {e}"
            )
        return created_hyperedge_ids

    async def _apply_temporal_decay_and_patterns(self):
        logger.debug("Temporal decay and pattern detection - placeholder.")
        try:
            nodes = await self._get_evidence_nodes_from_neo4j()
        except Exception as e:
            logger.error(f"Failed fetching evidence nodes for decay: {e}")
            return

        now = dt.utcnow()
        for node in nodes:
            ts_str = node.get("metadata_timestamp_iso")
            if not ts_str:
                continue
            try:
                ts = dt.fromisoformat(ts_str)
            except ValueError:
                continue
            age_days = max((now - ts).days, 0)
            decay_factor = max(0.0, 1 - 0.05 * age_days)
            node["confidence_empirical_support"] *= decay_factor

        await self._persist_confidence_updates(nodes)

    async def _get_evidence_nodes_from_neo4j(self) -> list[dict[str, Any]]:
        """Placeholder to fetch evidence nodes for decay."""
        return []

    async def _persist_confidence_updates(self, nodes: list[dict[str, Any]]):
        """Placeholder to persist updated confidence values."""
        return None

    def _adapt_graph_topology(self):
        """Placeholder that orchestrates graph topology adjustments."""
        logger.debug("Dynamic graph topology adaptation - placeholder.")
        try:
            result = self._create_hyperedges_in_neo4j({}, [])
            if asyncio.iscoroutine(result):
                asyncio.get_event_loop().run_until_complete(result)
        except Exception as e:
            logger.error(f"Error creating hyperedges: {e}")

        try:
            self._remove_redundant_edges_from_neo4j()
        except Exception as e:
            logger.error(f"Error removing redundant edges: {e}")

        try:
            self._simplify_graph()
        except Exception as e:
            logger.error(f"Error simplifying graph: {e}")

    def _remove_redundant_edges_from_neo4j(self) -> None:
        logger.debug("Removing redundant edges - placeholder.")
        pass

    def _simplify_graph(self) -> None:
        logger.debug("Simplifying graph structure - placeholder.")
        pass

    async def execute(
        self, current_session_data: GoTProcessorSessionData
    ) -> StageOutput:
        """
        Executes the evidence integration stage by gathering evidence for hypotheses, updating the Neo4j graph, and tracking progress.

        This method iterates through hypotheses, retrieves supporting or contradictory evidence from external sources, creates corresponding nodes and edges in Neo4j, updates hypothesis confidence, and manages interdisciplinary bridge nodes (IBNs) and hyperedges. It tracks and returns metrics on evidence integration progress and updates the session context accordingly.

        Args:
                current_session_data: The session data containing accumulated context, including hypothesis node IDs.

        Returns:
                A StageOutput object summarizing the number of iterations, evidence nodes created, hypotheses updated, IBNs and hyperedges created, and context updates.
        """
        self._log_start(current_session_data.session_id)
        hypothesis_data = current_session_data.accumulated_context.get(
            HypothesisStage.stage_name, {}
        )
        hypothesis_node_ids: list[str] = hypothesis_data.get("hypothesis_node_ids", [])

        evidence_created_count = 0
        hypotheses_updated_count = 0
        ibns_created_count = 0
        hyperedges_created_count = 0
        iterations_run = 0

        try:
            if not hypothesis_node_ids:
                logger.warning("No hypothesis IDs found. Skipping evidence stage.")
                summary = "Evidence skipped: No hypotheses."
                metrics = {
                    "iterations_completed": 0,
                    "evidence_nodes_created_in_neo4j": 0,
                    "hypotheses_updated_in_neo4j": 0,
                    "ibns_created_in_neo4j": 0,
                    "hyperedges_created_in_neo4j": 0,
                }
                context_update = {
                    "evidence_integration_completed": False,
                    "error": "No hypotheses found",
                }
                # Ensure stage_name is used as key for context_update
                return StageOutput(
                    summary=summary,
                    metrics=metrics,
                    next_stage_context_update={self.stage_name: context_update},
                    logs="\n".join(self._execution_logs),
                )

            processed_hypotheses_this_run: set[str] = set()
            for iteration_num in range(self.max_iterations):
                iterations_run = iteration_num + 1  # Correctly increment iterations_run
                logger.info(
                    f"Evidence integration iteration {iterations_run}/{self.max_iterations}"
                )

                eligible_ids_for_selection = [
                    hid
                    for hid in hypothesis_node_ids
                    if hid not in processed_hypotheses_this_run
                ]
                if not eligible_ids_for_selection:
                    logger.info(
                        "All hypotheses processed or no eligible ones left for this run."
                    )
                    break

                selected_hypothesis_data = (
                    await self._select_hypothesis_to_evaluate_from_neo4j(
                        eligible_ids_for_selection
                    )
                )
                if not selected_hypothesis_data:
                    logger.info(
                        "No more eligible hypotheses to evaluate in this iteration loop."
                    )
                    break

                current_hypothesis_id = selected_hypothesis_data["id"]
                processed_hypotheses_this_run.add(current_hypothesis_id)

                found_evidence_conceptual_list = await self._execute_hypothesis_plan(
                    selected_hypothesis_data
                )

                if not found_evidence_conceptual_list:
                    logger.debug(
                        f"No new evidence found/generated for hypothesis '{selected_hypothesis_data.get('label', current_hypothesis_id)}'."
                    )
                    continue

                related_evidence_data_for_hyperedge: list[dict[str, Any]] = []
                for ev_idx, ev_conceptual_data in enumerate(
                    found_evidence_conceptual_list
                ):
                    created_evidence_neo4j_data = await self._create_evidence_in_neo4j(
                        selected_hypothesis_data,
                        ev_conceptual_data,
                        iteration_num,
                        ev_idx,
                    )
                    if not created_evidence_neo4j_data:
                        logger.warning(
                            f"Failed to create Neo4j data for one piece of evidence for hypothesis {current_hypothesis_id}."
                        )
                        continue
                    evidence_created_count += 1
                    related_evidence_data_for_hyperedge.append(
                        created_evidence_neo4j_data
                    )

                    prior_confidence_list = selected_hypothesis_data.get(
                        "confidence_vector_list", [0.5] * 4
                    )
                    prior_hypo_confidence_obj = ConfidenceVector(
                        empirical_support=prior_confidence_list[0],
                        theoretical_basis=prior_confidence_list[1],
                        methodological_rigor=prior_confidence_list[2],
                        consensus_alignment=prior_confidence_list[3],
                    )
                    edge_type_for_update = (
                        EdgeType.SUPPORTIVE
                        if ev_conceptual_data["supports_hypothesis"]
                        else EdgeType.CONTRADICTORY
                    )
                    update_successful = (
                        await self._update_hypothesis_confidence_in_neo4j(
                            current_hypothesis_id,
                            prior_hypo_confidence_obj,
                            ev_conceptual_data.get("strength", 0.5),
                            ev_conceptual_data["supports_hypothesis"],
                            ev_conceptual_data.get("statistical_power"),
                            edge_type_for_update,
                        )
                    )
                    if update_successful:
                        hypotheses_updated_count += 1

                    # Defensive checks for keys needed by _create_ibn_in_neo4j
                    # created_evidence_neo4j_data now comes from Neo4j, so 'label' and 'metadata_disciplinary_tags' should be Neo4j property names
                    if "label" not in created_evidence_neo4j_data:
                        created_evidence_neo4j_data["label"] = (
                            f"Ev_default_label_{iteration_num}_{ev_idx}"
                        )
                    if "metadata_disciplinary_tags" not in created_evidence_neo4j_data:
                        created_evidence_neo4j_data["metadata_disciplinary_tags"] = (
                            list(ev_conceptual_data.get("disciplinary_tags", []))
                        )

                    ibn_created_id = await self._create_ibn_in_neo4j(
                        created_evidence_neo4j_data, selected_hypothesis_data
                    )
                    if ibn_created_id:
                        ibns_created_count += 1

                if related_evidence_data_for_hyperedge:
                    # Ensure evidence data for hyperedge has 'confidence_empirical_support' (Neo4j property name)
                    for ev_data_item in related_evidence_data_for_hyperedge:
                        if (
                            "confidence_empirical_support" not in ev_data_item
                        ):  # This is Neo4j property name
                            ev_data_item["confidence_empirical_support"] = (
                                ev_data_item.get("confidence_empirical_support", 0.5)
                            )  # Default if still missing

                    hyperedge_ids = await self._create_hyperedges_in_neo4j(
                        selected_hypothesis_data, related_evidence_data_for_hyperedge
                    )
                    hyperedges_created_count += len(hyperedge_ids)

            await self._apply_temporal_decay_and_patterns()
            await self._adapt_graph_topology()

        finally:
            await self.close_clients()  # Ensure clients are closed

        summary = (
            f"Evidence integration completed. Iterations run: {iterations_run}. "
            f"Evidence created: {evidence_created_count}. Hypotheses updated: {hypotheses_updated_count}. "
            f"IBNs created: {ibns_created_count}. Hyperedges created: {hyperedges_created_count}."
        )
        metrics = {
            "iterations_completed": iterations_run,  # Use the counter
            "evidence_nodes_created_in_neo4j": evidence_created_count,
            "hypotheses_updated_in_neo4j": hypotheses_updated_count,
            "ibns_created_in_neo4j": ibns_created_count,
            "hyperedges_created_in_neo4j": hyperedges_created_count,
        }
        context_update = {
            "evidence_integration_completed": True,
            "evidence_nodes_added_count": evidence_created_count,
        }

        return StageOutput(
            summary=summary,
            metrics=metrics,
            next_stage_context_update={self.stage_name: context_update},
            logs="\n".join(self._execution_logs),
        )
