from loguru import logger

from adaptive_graph_of_thoughts.config import Settings
from adaptive_graph_of_thoughts.domain.models.common_types import (
    GoTProcessorSessionData,
)

# from adaptive_graph_of_thoughts.domain.models.graph_state import ASRGoTGraph # No longer used
from adaptive_graph_of_thoughts.infrastructure.neo4j_utils import (
    Neo4jError,
    execute_query,
)

# from adaptive_graph_of_thoughts.domain.utils.metadata_helpers import calculate_semantic_similarity # Placeholder, hard to use directly in Neo4j
from adaptive_graph_of_thoughts.domain.stages.base_stage import BaseStage, StageOutput


class PruningMergingStage(BaseStage):
    stage_name: str = "PruningMergingStage"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        # P1.5: Pruning and Merging thresholds from settings
        self.pruning_confidence_threshold = (
            self.default_params.pruning_confidence_threshold
        )
        self.pruning_impact_threshold = self.default_params.pruning_impact_threshold
        self.merging_semantic_overlap_threshold = (
            self.default_params.merging_semantic_overlap_threshold
        )
        # Example: Threshold for pruning low-confidence edges
        self.pruning_edge_confidence_threshold = getattr(
            self.default_params, "pruning_edge_confidence_threshold", 0.2
        )

    async def _prune_low_confidence_and_isolated_nodes_in_neo4j(self) -> int:
        """Prunes low-confidence/impact nodes and isolated nodes in a single query."""

        combined_query = """
        OPTIONAL MATCH (n:Node)
        WHERE NOT n:ROOT
          AND NOT n:DECOMPOSITION_DIMENSION
          AND n.type IN ['HYPOTHESIS', 'EVIDENCE', 'INTERDISCIPLINARY_BRIDGE']
          AND coalesce(
                apoc.coll.min([
                    coalesce(n.confidence_empirical_support, 1.0),
                    coalesce(n.confidence_theoretical_basis, 1.0),
                    coalesce(n.confidence_methodological_rigor, 1.0),
                    coalesce(n.confidence_consensus_alignment, 1.0)
                ]), 1.0
            ) < $conf_thresh
          AND coalesce(n.metadata_impact_score, 1.0) < $impact_thresh
        WITH collect(DISTINCT n) AS low_conf_nodes
        OPTIONAL MATCH (m:Node)
        WHERE NOT m:ROOT AND size((m)--()) = 0
        WITH low_conf_nodes + collect(DISTINCT m) AS nodes_to_delete
        FOREACH (nd IN nodes_to_delete | DETACH DELETE nd)
        RETURN size(nodes_to_delete) AS pruned_count
        """
        try:
            result = await execute_query(
                combined_query,
                {
                    "conf_thresh": self.pruning_confidence_threshold,
                    "impact_thresh": self.pruning_impact_threshold,
                },
                tx_type="write",
            )
            pruned_count = result[0]["pruned_count"] if result and result[0] else 0
            if pruned_count > 0:
                logger.info(
                    f"Pruned {pruned_count} nodes (low-confidence/impact or isolated) from Neo4j."
                )
            else:
                logger.info("No nodes met the criteria for combined pruning.")
            return pruned_count
        except Neo4jError as e:
            logger.error(f"Neo4j error during node pruning: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during node pruning: {e}")
            return 0

    async def _prune_low_confidence_edges_in_neo4j(self) -> int:
        """Prunes edges with confidence below a threshold."""
        query = """
        MATCH ()-[r]->()
        WHERE r.confidence IS NOT NULL AND r.confidence < $threshold
        DELETE r
        RETURN count(r) as pruned_count
        """
        try:
            result = await execute_query(
                query,
                {"threshold": self.pruning_edge_confidence_threshold},
                tx_type="write",
            )
            pruned_count = result[0]["pruned_count"] if result and result[0] else 0
            if pruned_count > 0:
                logger.info(f"Pruned {pruned_count} low-confidence edges from Neo4j.")
            return pruned_count
        except Neo4jError as e:
            logger.error(f"Neo4j error during low-confidence edge pruning: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during low-confidence edge pruning: {e}")
            return 0

    async def _merge_nodes_in_neo4j(self) -> int:
        """
        Placeholder for node merging directly in Neo4j.
        Actual merging based on semantic similarity is complex and typically requires external computation.
        This function will be a no-op for now, or implement a very simple merge if possible.
        """
        # The original merging logic is heavily reliant on Python-based semantic similarity
        # and complex Pydantic model interactions, which are difficult to translate directly
        # into efficient, generic Cypher queries without pre-calculated similarity scores
        # or a more defined, property-based merging strategy.

        # Example of a very simple merge (e.g., duplicate IDs if that were an issue, not semantic):
        # MATCH (n:Node)
        # WITH n.label AS label, n.type AS type, collect(n) AS nodes
        # WHERE size(nodes) > 1
        # CALL apoc.refactor.mergeNodes(nodes, {properties: "combine", mergeRels: true}) YIELD node
        # RETURN count(node) as merged_group_representative_count
        # This is NOT what the original code did and is just an example of APOC usage.

        logger.info(
            "Node merging in Neo4j is currently a placeholder and not fully implemented due to complexity of semantic similarity in Cypher."
        )
        # For now, this will be a no-op.
        return 0

    async def execute(
        self,
        current_session_data: GoTProcessorSessionData,  # graph: ASRGoTGraph removed
    ) -> StageOutput:
        self._log_start(current_session_data.session_id)

        total_nodes_pruned = 0
        total_edges_pruned = 0

        logger.info(
            "Starting Neo4j node pruning phase (low confidence/impact and isolated)..."
        )
        nodes_pruned_conf_impact = (
            await self._prune_low_confidence_and_isolated_nodes_in_neo4j()
        )
        total_nodes_pruned += nodes_pruned_conf_impact

        logger.info("Starting Neo4j edge pruning phase (low confidence)...")
        edges_pruned_conf = await self._prune_low_confidence_edges_in_neo4j()
        total_edges_pruned += edges_pruned_conf

        logger.info("Starting Neo4j node merging phase (currently placeholder)...")
        # Merging is complex; the direct Neo4j version is simplified/deferred.
        merged_count = await self._merge_nodes_in_neo4j()

        # Fetch current node and edge counts from Neo4j for metrics
        nodes_remaining = 0
        edges_remaining = 0
        try:
            # This specific tool might only execute one query or handle multi-statement differently.
            # For simplicity, let's assume we can get these counts, or make two calls.
            node_count_res = await execute_query(
                "MATCH (n:Node) RETURN count(n) AS node_count",
                {},
                tx_type="read",
            )
            if node_count_res:
                nodes_remaining = node_count_res[0]["node_count"]

            edge_count_res = await execute_query(
                "MATCH ()-[r]->() RETURN count(r) AS edge_count",
                {},
                tx_type="read",
            )
            if edge_count_res:
                edges_remaining = edge_count_res[0]["edge_count"]
        except Neo4jError as e:
            logger.error(f"Failed to get node/edge counts from Neo4j: {e}")

        summary = (
            f"Neo4j graph refinement completed. "
            f"Total nodes pruned: {total_nodes_pruned}. "
            f"Total edges pruned: {total_edges_pruned}. "
            f"Nodes merged (pairs): {merged_count} (merging logic is simplified/placeholder)."
        )
        metrics = {
            "nodes_pruned_in_neo4j": total_nodes_pruned,
            "edges_pruned_in_neo4j": total_edges_pruned,
            "nodes_merged_in_neo4j": merged_count,
            "nodes_remaining_in_neo4j": nodes_remaining,
            "edges_remaining_in_neo4j": edges_remaining,
        }
        context_update = {
            "pruning_merging_completed": True,
            "nodes_after_pruning_merging_in_neo4j": nodes_remaining,
        }

        return StageOutput(
            summary=summary,
            metrics=metrics,
            next_stage_context_update={self.stage_name: context_update},
        )
