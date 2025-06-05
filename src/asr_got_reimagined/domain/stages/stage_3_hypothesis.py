import random
from typing import Any

from loguru import logger

from asr_got_reimagined.config import Settings
from asr_got_reimagined.domain.models.common import (
    ConfidenceVector,
    EpistemicStatus,
)
from asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from asr_got_reimagined.domain.models.graph_elements import (
    BiasFlag,
    Edge,
    EdgeMetadata,
    EdgeType,
    FalsificationCriteria,
    Node,
    NodeMetadata,
    NodeType,
    Plan,
)
# from asr_got_reimagined.domain.models.graph_state import ASRGoTGraph # No longer used
from asr_got_reimagined.domain.services.neo4j_utils import execute_query, Neo4jError # Import Neo4j utils

import json # For property preparation
from datetime import datetime # For property preparation
from enum import Enum # For property preparation
from typing import Dict, List, Set, Optional, Union # For type hints

from asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput

# Import names of previous stages to access their output keys in accumulated_context
from .stage_2_decomposition import DecompositionStage


class HypothesisStage(BaseStage):
    stage_name: str = "HypothesisStage"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.k_min_hypotheses = self.default_params.hypotheses_per_dimension.min_hypotheses
        self.k_max_hypotheses = self.default_params.hypotheses_per_dimension.max_hypotheses
        self.hypothesis_confidence_values = self.default_params.hypothesis_confidence
        self.default_disciplinary_tags_config = self.default_params.default_disciplinary_tags
        self.default_plan_types_config = self.default_params.default_plan_types

    def _prepare_node_properties_for_neo4j(self, node_pydantic: Node) -> Dict[str, Any]:
        """Converts a Node Pydantic model into a flat dictionary for Neo4j."""
        if node_pydantic is None: return {}
        props = {"id": node_pydantic.id, "label": node_pydantic.label}
        if node_pydantic.confidence:
            for cv_field, cv_val in node_pydantic.confidence.model_dump().items():
                if cv_val is not None: props[f"confidence_{cv_field}"] = cv_val
        if node_pydantic.metadata:
            for meta_field, meta_val in node_pydantic.metadata.model_dump().items():
                if meta_val is None: continue
                if isinstance(meta_val, datetime): props[f"metadata_{meta_field}"] = meta_val.isoformat()
                elif isinstance(meta_val, Enum): props[f"metadata_{meta_field}"] = meta_val.value
                elif isinstance(meta_val, (list, set)):
                    if all(isinstance(item, (str, int, float, bool)) for item in meta_val):
                        props[f"metadata_{meta_field}"] = list(meta_val)
                    else:
                        try:
                            items_as_dicts = [item.model_dump() if hasattr(item, 'model_dump') else item for item in meta_val]
                            props[f"metadata_{meta_field}_json"] = json.dumps(items_as_dicts)
                        except TypeError as e:
                            logger.warning(f"Could not serialize list/set metadata field {meta_field} to JSON: {e}")
                            props[f"metadata_{meta_field}_str"] = str(meta_val)
                elif hasattr(meta_val, 'model_dump'): # Handles Plan, FalsificationCriteria, BiasFlag if they are direct fields
                    try: props[f"metadata_{meta_field}_json"] = json.dumps(meta_val.model_dump())
                    except TypeError as e:
                        logger.warning(f"Could not serialize Pydantic metadata field {meta_field} to JSON: {e}")
                        props[f"metadata_{meta_field}_str"] = str(meta_val)
                else: props[f"metadata_{meta_field}"] = meta_val
        return {k: v for k, v in props.items() if v is not None}

    def _prepare_edge_properties_for_neo4j(self, edge_pydantic: Edge) -> Dict[str, Any]:
        """Converts an Edge Pydantic model into a flat dictionary for Neo4j."""
        if edge_pydantic is None: return {}
        props = {"id": edge_pydantic.id}
        if hasattr(edge_pydantic, 'confidence') and edge_pydantic.confidence is not None:
             props["confidence"] = edge_pydantic.confidence
        if edge_pydantic.metadata:
            for meta_field, meta_val in edge_pydantic.metadata.model_dump().items():
                if meta_val is None: continue
                if isinstance(meta_val, datetime): props[f"metadata_{meta_field}"] = meta_val.isoformat()
                elif isinstance(meta_val, Enum): props[f"metadata_{meta_field}"] = meta_val.value
                elif isinstance(meta_val, (list,set,dict)) or hasattr(meta_val, 'model_dump'):
                    try: props[f"metadata_{meta_field}_json"] = json.dumps(meta_val.model_dump() if hasattr(meta_val, 'model_dump') else meta_val)
                    except TypeError: props[f"metadata_{meta_field}_str"] = str(meta_val)
                else: props[f"metadata_{meta_field}"] = meta_val
        return {k: v for k, v in props.items() if v is not None}

    async def _generate_hypothesis_content(
        self, dimension_label: str, dimension_tags: Set[str], hypo_index: int, initial_query: str
    ) -> dict[str, Any]:
        """
        Generates the content dictionary for a single hypothesis.
        Args:
            dimension_label: Label of the dimension node.
            dimension_tags: Disciplinary tags from the dimension node.
            hypo_index: Index of the hypothesis for this dimension.
            initial_query: The original query string.
        Returns:
            A dictionary for hypothesis metadata.
        """
        base_hypothesis_text = f"Hypothesis {hypo_index + 1} regarding '{dimension_label}' for query '{initial_query[:30]}...'"
        plan_type = random.choice(self.default_plan_types_config)
        plan_pydantic = Plan(
            type=plan_type, description=f"Plan to evaluate '{base_hypothesis_text}' via {plan_type}.",
            estimated_cost=random.uniform(0.2, 0.8), estimated_duration=random.uniform(1.0, 5.0),
            required_resources=[random.choice(["dataset_X", "computational_cluster", "expert_A"])]
        )
        fals_conditions = [f"Observe contradictory evidence from {plan_type}", f"Find statistical insignificance in {random.choice(['key_metric_A', 'key_metric_B'])}"]
        falsifiability_pydantic = FalsificationCriteria(
            description=f"This hypothesis could be falsified if {fals_conditions[0].lower()} or if {fals_conditions[1].lower()}.",
            testable_conditions=fals_conditions
        )
        bias_flags_list = []
        if random.random() < 0.15:
            bias_type = random.choice(["Confirmation Bias", "Availability Heuristic", "Anchoring Bias"])
            bias_flags_list.append(BiasFlag(
                bias_type=bias_type, description=f"Potential {bias_type} in formulating or prioritizing this hypothesis.",
                assessment_stage_id=self.stage_name, severity=random.choice(["low", "medium"])
            ))
        impact_score_float = random.uniform(0.2, 0.9)
        num_tags = random.randint(1, min(2, len(self.default_disciplinary_tags_config)))
        hypo_disciplinary_tags = set(random.sample(self.default_disciplinary_tags_config, num_tags))
        hypo_disciplinary_tags.update(dimension_tags) # Add dimension's tags

        return {
            "label": base_hypothesis_text, "plan": plan_pydantic,
            "falsification_criteria": falsifiability_pydantic, "bias_flags": bias_flags_list,
            "impact_score": impact_score_float, "disciplinary_tags": list(hypo_disciplinary_tags),
        }

    async def execute(
        self, current_session_data: GoTProcessorSessionData # graph: ASRGoTGraph removed
    ) -> StageOutput:
        self._log_start(current_session_data.session_id)

        decomposition_data = current_session_data.accumulated_context.get(DecompositionStage.stage_name, {})
        dimension_node_ids: List[str] = decomposition_data.get("dimension_node_ids", [])
        initial_query = current_session_data.query # Needed for hypothesis content generation
        operational_params = current_session_data.accumulated_context.get("operational_params", {})

        if not dimension_node_ids:
            logger.warning("No dimension node IDs found. Skipping hypothesis generation.")
            return StageOutput(summary="Hypothesis generation skipped: No dimensions.",
                               metrics={"hypotheses_created_in_neo4j": 0, "relationships_created_in_neo4j": 0},
                               next_stage_context_update={self.stage_name: {"error": "No dimensions found", "hypothesis_node_ids": []}})

        all_hypothesis_node_ids_created: List[str] = []
        nodes_created_count = 0
        edges_created_count = 0
        
        batch_hypothesis_node_data = []
        # Maps Neo4j hypothesis_id to its source dimension_id and the hypothesis label
        created_hypotheses_map: Dict[str, Dict[str, str]] = {} 

        k_min = operational_params.get("hypotheses_per_dimension_min", self.k_min_hypotheses)
        k_max = operational_params.get("hypotheses_per_dimension_max", self.k_max_hypotheses)
        
        # Step 1: Collect all hypothesis data for batch node creation
        for dim_id in dimension_node_ids:
            try:
                fetch_dim_query = "MATCH (d:Node {id: $dimension_id}) RETURN properties(d) as props"
                dim_records = await execute_query(fetch_dim_query, {"dimension_id": dim_id}, tx_type="read")
                if not dim_records or not dim_records[0].get("props"):
                    logger.warning(f"Dimension node {dim_id} not found. Skipping hypothesis generation for it.")
                    continue
                
                dim_props = dim_records[0]["props"]
                dimension_label_for_hypo = dim_props.get("label", "Unknown Dimension")
                dimension_tags_for_hypo = set(dim_props.get("metadata_disciplinary_tags", []))
                dimension_layer_for_hypo = dim_props.get("metadata_layer_id", self.default_params.initial_layer)

                k_hypotheses_to_generate = random.randint(k_min, k_max)
                logger.debug(f"Preparing {k_hypotheses_to_generate} hypotheses for dimension: '{dimension_label_for_hypo}' (ID: {dim_id})")

                for i in range(k_hypotheses_to_generate):
                    hypo_content = await self._generate_hypothesis_content(
                        dimension_label_for_hypo, dimension_tags_for_hypo, i, initial_query
                    )
                    hypo_id_neo4j = f"hypo_{dim_id}_{current_session_data.session_id}_{i}"

                    hypo_metadata = NodeMetadata(
                        description=f"A hypothesis related to dimension: '{dimension_label_for_hypo}'.",
                        source_description="HypothesisStage (P1.3)",
                        epistemic_status=EpistemicStatus.HYPOTHESIS,
                        disciplinary_tags=list(hypo_content["disciplinary_tags"]),
                        falsification_criteria=hypo_content["falsification_criteria"],
                        bias_flags=hypo_content["bias_flags"],
                        impact_score=hypo_content["impact_score"],
                        plan=hypo_content["plan"],
                        layer_id=operational_params.get("hypothesis_layer", dimension_layer_for_hypo),
                    )
                    hypothesis_node_pydantic = Node(
                        id=hypo_id_neo4j, label=hypo_content["label"], type=NodeType.HYPOTHESIS,
                        confidence=ConfidenceVector.from_list(self.hypothesis_confidence_values),
                        metadata=hypo_metadata
                    )
                    hyp_props_for_neo4j = self._prepare_node_properties_for_neo4j(hypothesis_node_pydantic)
                    
                    batch_hypothesis_node_data.append({
                        "props": hyp_props_for_neo4j,
                        "type_label_value": NodeType.HYPOTHESIS.value,
                        "dim_id_source": dim_id, # To link back for relationship creation
                        "hypo_label_original": hypo_content["label"] # For logging/mapping
                    })
            except Neo4jError as e:
                logger.error(f"Neo4j error fetching dimension {dim_id}: {e}. Skipping.")
            except Exception as e:
                logger.error(f"Unexpected error preparing hypotheses for dimension {dim_id}: {e}. Skipping.")

        # Step 2: Execute batch hypothesis node creation
        if batch_hypothesis_node_data:
            try:
                batch_node_query = """
                UNWIND $batch_data AS item
                MERGE (h:Node {id: item.props.id}) SET h += item.props
                WITH h, item.type_label_value AS typeLabelValue CALL apoc.create.addLabels(h, [typeLabelValue]) YIELD node
                RETURN node.id AS created_hyp_id, item.dim_id_source AS dim_id_source, item.hypo_label_original AS hypo_label
                """
                results_nodes = await execute_query(batch_node_query, {"batch_data": batch_hypothesis_node_data}, tx_type='write')
                
                for record in results_nodes:
                    created_hyp_id = record["created_hyp_id"]
                    dim_id_source = record["dim_id_source"]
                    hypo_label = record["hypo_label"]
                    
                    all_hypothesis_node_ids_created.append(created_hyp_id)
                    nodes_created_count += 1
                    created_hypotheses_map[created_hyp_id] = {"dim_id": dim_id_source, "label": hypo_label}
                    logger.debug(f"Batch created/merged hypothesis node '{hypo_label}' (ID: {created_hyp_id}) for dimension {dim_id_source}.")
            except Neo4jError as e:
                logger.error(f"Neo4j error during batch hypothesis node creation: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during batch hypothesis node creation: {e}")

        # Step 3: Collect and batch relationship creation
        batch_relationship_data = []
        if created_hypotheses_map:
            for created_hyp_id, hypo_data_map in created_hypotheses_map.items():
                dim_id_for_rel = hypo_data_map["dim_id"]
                hypo_label_for_rel = hypo_data_map["label"]
                
                # Need dimension label for edge description - fetch it again or store it earlier if needed
                # For simplicity, we'll use a generic description or try to retrieve it if it was stored with dim_id
                # This part might need adjustment if dim_label is not easily accessible here.
                # Assuming dim_id_for_rel can be used to fetch its label if necessary, or we use what we have.
                # For now, using a placeholder from the hypo_label_for_rel.
                dim_label_placeholder = f"Dimension for '{hypo_label_for_rel[:20]}...'"


                edge_id = f"edge_{dim_id_for_rel}_genhyp_{created_hyp_id}"
                edge_pydantic = Edge(
                    id=edge_id, source_id=dim_id_for_rel, target_id=created_hyp_id,
                    type=EdgeType.GENERATES_HYPOTHESIS, confidence=0.9,
                    metadata=EdgeMetadata(description=f"Hypothesis '{hypo_label_for_rel}' generated for dimension '{dim_label_placeholder}'.")
                )
                edge_props_for_neo4j = self._prepare_edge_properties_for_neo4j(edge_pydantic)
                batch_relationship_data.append({
                    "dim_id": dim_id_for_rel, 
                    "hyp_id": created_hyp_id, 
                    "props": edge_props_for_neo4j
                })

        # Step 4: Execute batch relationship creation
        if batch_relationship_data:
            try:
                batch_rel_query = """
                UNWIND $batch_rels AS rel_detail
                MATCH (dim:Node {id: rel_detail.dim_id})
                MATCH (hyp:Node {id: rel_detail.hyp_id})
                MERGE (dim)-[r:GENERATES_HYPOTHESIS {id: rel_detail.props.id}]->(hyp)
                SET r += rel_detail.props
                RETURN count(r) AS total_rels_created
                """
                result_rels = await execute_query(batch_rel_query, {"batch_rels": batch_relationship_data}, tx_type='write')
                if result_rels and result_rels[0].get("total_rels_created") is not None:
                    edges_created_count = result_rels[0]["total_rels_created"]
                    logger.debug(f"Batch created {edges_created_count} GENERATES_HYPOTHESIS relationships.")
                else:
                    logger.error("Batch GENERATES_HYPOTHESIS relationship creation query did not return expected count.")
            except Neo4jError as e:
                logger.error(f"Neo4j error during batch GENERATES_HYPOTHESIS relationship creation: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during batch GENERATES_HYPOTHESIS relationship creation: {e}")

        summary = f"Generated {nodes_created_count} hypotheses in Neo4j across {len(dimension_node_ids)} dimensions."
        metrics = {
            "hypotheses_created_in_neo4j": nodes_created_count,
            "relationships_created_in_neo4j": edges_created_count,
            "avg_hypotheses_per_dimension": nodes_created_count / len(dimension_node_ids) if dimension_node_ids else 0,
        }
        # Ensure hypotheses_results key is populated if other stages expect it
        hypotheses_results_for_context = [
            {"id": hyp_id, "label": data["label"], "dimension_id": data["dim_id"]} 
            for hyp_id, data in created_hypotheses_map.items()
        ]
        context_update = {
            "hypothesis_node_ids": all_hypothesis_node_ids_created,
            "hypotheses_results": hypotheses_results_for_context # Added based on got_processor expectation
        }

        return StageOutput(summary=summary, metrics=metrics, next_stage_context_update={self.stage_name: context_update})
