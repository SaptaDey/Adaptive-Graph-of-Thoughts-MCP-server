from typing import Any

from loguru import logger  # type: ignore

from asr_got_reimagined.config import Settings
from asr_got_reimagined.domain.models.common import (
    ConfidenceVector,
    EpistemicStatus,
)
from asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from asr_got_reimagined.domain.models.graph_elements import (
    Edge,
    EdgeMetadata,
    EdgeType,
    Node,
    NodeMetadata,
    NodeType,
)
# from asr_got_reimagined.domain.models.graph_state import ASRGoTGraph # No longer used
from asr_got_reimagined.domain.services.neo4j_utils import execute_query, Neo4jError
from .stage_1_initialization import InitializationStage # For context key
from asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput

import json # For property preparation
from datetime import datetime # For property preparation
from enum import Enum # For property preparation
from typing import Dict, List, Set, Optional # For type hints


class DecompositionStage(BaseStage):
    stage_name: str = "DecompositionStage"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.default_dimensions_config = (
            self.default_params.default_decomposition_dimensions
        )
        self.dimension_confidence_values = (
            self.default_params.dimension_confidence
        )

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
                elif hasattr(meta_val, 'model_dump'):
                    try: props[f"metadata_{meta_field}_json"] = json.dumps(meta_val.model_dump())
                    except TypeError as e:
                        logger.warning(f"Could not serialize Pydantic metadata field {meta_field} to JSON: {e}")
                        props[f"metadata_{meta_field}_str"] = str(meta_val)
                else: props[f"metadata_{meta_field}"] = meta_val
        return {k: v for k, v in props.items() if v is not None}

    def _prepare_edge_properties_for_neo4j(self, edge_pydantic: Edge) -> Dict[str, Any]:
        """Converts an Edge Pydantic model into a flat dictionary for Neo4j."""
        if edge_pydantic is None: return {}
        props = {"id": edge_pydantic.id} # Type is handled by relationship type in query
        # Add confidence if it exists and is not None
        if hasattr(edge_pydantic, 'confidence') and edge_pydantic.confidence is not None:
             props["confidence"] = edge_pydantic.confidence # Assuming confidence is a simple float for edges
        
        if edge_pydantic.metadata:
            for meta_field, meta_val in edge_pydantic.metadata.model_dump().items():
                if meta_val is None: continue
                if isinstance(meta_val, datetime): props[f"metadata_{meta_field}"] = meta_val.isoformat()
                elif isinstance(meta_val, Enum): props[f"metadata_{meta_field}"] = meta_val.value
                # Simplified: assume edge metadata fields are simple or JSON serializable strings
                elif isinstance(meta_val, (list,set,dict)) or hasattr(meta_val, 'model_dump'):
                    try: props[f"metadata_{meta_field}_json"] = json.dumps(meta_val.model_dump() if hasattr(meta_val, 'model_dump') else meta_val)
                    except TypeError: props[f"metadata_{meta_field}_str"] = str(meta_val)
                else: props[f"metadata_{meta_field}"] = meta_val
        return {k: v for k, v in props.items() if v is not None}


    def _get_conceptual_dimensions(
        self, 
        root_node_query_context: Optional[str], 
        custom_dimensions_input: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Determines dimensions to create based on input or defaults."""
        if custom_dimensions_input and isinstance(custom_dimensions_input, list):
            logger.info("Using custom decomposition dimensions provided in operational parameters.")
            # Basic validation: ensure items are dicts with 'label' and 'description'
            return [
                dim for dim in custom_dimensions_input 
                if isinstance(dim, dict) and "label" in dim and "description" in dim
            ]
        else:
            logger.info("Using default decomposition dimensions from configuration.")
            # Adapt default dimensions, possibly incorporating root_node_query_context
            # For now, directly using configured defaults.
            return [
                {"label": dim.label, "description": dim.description}
                for dim in self.default_dimensions_config
            ]

    async def execute(
        self, current_session_data: GoTProcessorSessionData # graph: ASRGoTGraph removed
    ) -> StageOutput:
        self._log_start(current_session_data.session_id)

        initialization_data = current_session_data.accumulated_context.get(
            InitializationStage.stage_name, {}
        )
        root_node_id = initialization_data.get("root_node_id")
        # Ensure disciplinary_tags is a list of strings
        initial_disciplinary_tags: Set[str] = set(initialization_data.get("initial_disciplinary_tags", []))


        if not root_node_id:
            err_msg = "Root node ID not found in session context. Cannot proceed."
            logger.error(err_msg)
            return StageOutput(summary=err_msg, metrics={"dimensions_created_in_neo4j": 0, "relationships_created_in_neo4j": 0},
                               next_stage_context_update={self.stage_name: {"error": err_msg, "dimension_node_ids": []}})
        
        # Fetch root node's query context or label for decomposition input
        root_node_info: Optional[Dict[str, Any]] = None
        try:
            query = "MATCH (n:Node {id: $root_node_id}) RETURN properties(n) AS props"
            results = execute_query(query, {"root_node_id": root_node_id}, tx_type="read")
            if results and results[0].get("props"):
                root_node_info = results[0]["props"]
            else:
                err_msg = f"Root node {root_node_id} not found in Neo4j."
                logger.error(err_msg)
                return StageOutput(summary=err_msg, metrics={"dimensions_created_in_neo4j": 0, "relationships_created_in_neo4j": 0},
                                   next_stage_context_update={self.stage_name: {"error": err_msg, "dimension_node_ids": []}})
        except Neo4jError as e:
            err_msg = f"Neo4j error fetching root node {root_node_id}: {e}"
            logger.error(err_msg)
            return StageOutput(summary=err_msg, metrics={"dimensions_created_in_neo4j": 0, "relationships_created_in_neo4j": 0},
                               next_stage_context_update={self.stage_name: {"error": err_msg, "dimension_node_ids": []}})

        # Use metadata_query_context if available, else label, else a default string
        decomposition_input_text = root_node_info.get("metadata_query_context") or root_node_info.get("label", "Root Task")
        root_node_layer_str = root_node_info.get("metadata_layer_id", self.default_params.initial_layer)


        operational_params = current_session_data.accumulated_context.get("operational_params", {})
        custom_dimensions_input = operational_params.get("decomposition_dimensions")
        
        dimensions_to_create_conceptual = self._get_conceptual_dimensions(decomposition_input_text, custom_dimensions_input)

        dimension_node_ids_created: List[str] = []
        nodes_created_count = 0
        edges_created_count = 0
        dimension_labels_created: List[str] = []
        
        batch_dimension_node_data = []
        created_dimensions_map: Dict[str, str] = {} # original_id to created_node_id

        for i, dim_data in enumerate(dimensions_to_create_conceptual):
            dim_label = dim_data.get("label", f"Dimension {i + 1}")
            dim_description = dim_data.get("description", f"Details for {dim_label}")
            # Make dim_id deterministic for mapping if needed, or use original_dim_data.label if unique
            original_dim_identifier = dim_data.get("id", dim_label) # Assuming label is unique enough for mapping or dim_data has a unique 'id'
            dim_id_neo4j = f"dim_{root_node_id}_{i}" # Neo4j node ID

            dim_metadata = NodeMetadata(
                description=dim_description,
                source_description="DecompositionStage (P1.2)",
                epistemic_status=EpistemicStatus.ASSUMPTION,
                disciplinary_tags=list(initial_disciplinary_tags),
                layer_id=operational_params.get("dimension_layer", root_node_layer_str),
                impact_score=0.7,
            )
            dimension_node_pydantic = Node(
                id=dim_id_neo4j, label=dim_label, type=NodeType.DECOMPOSITION_DIMENSION,
                confidence=ConfidenceVector.from_list(self.dimension_confidence_values),
                metadata=dim_metadata
            )
            node_props_for_neo4j = self._prepare_node_properties_for_neo4j(dimension_node_pydantic)
            type_label_value = NodeType.DECOMPOSITION_DIMENSION.value
            
            batch_dimension_node_data.append({
                "props": node_props_for_neo4j, 
                "type_label_value": type_label_value,
                "original_identifier": original_dim_identifier # To map back if needed
            })

        if batch_dimension_node_data:
            try:
                batch_node_query = """
                UNWIND $batch_data AS item
                MERGE (d:Node {id: item.props.id}) SET d += item.props
                WITH d, item.type_label_value AS typeLabelValue CALL apoc.create.addLabels(d, [typeLabelValue]) YIELD node
                RETURN node.id AS created_node_id, item.props.label AS created_label, item.original_identifier AS original_identifier
                """
                # Using item.props.label as created_label since node_props_for_neo4j contains 'label'
                results_nodes = execute_query(batch_node_query, {"batch_data": batch_dimension_node_data}, tx_type='write')
                
                for record in results_nodes:
                    created_node_id = record["created_node_id"]
                    created_label = record["created_label"] # This is the label from props, e.g., "Scope"
                    original_identifier = record["original_identifier"] # The original unique key for this dim_data

                    dimension_node_ids_created.append(created_node_id)
                    dimension_labels_created.append(created_label) # Store the actual label used
                    nodes_created_count += 1
                    created_dimensions_map[original_identifier] = created_node_id # Map original id to Neo4j id
                    logger.debug(f"Batch created/merged dimension node '{created_label}' (ID: {created_node_id}).")

            except Neo4jError as e:
                logger.error(f"Neo4j error during batch dimension node creation: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during batch dimension node creation: {e}")
        
        # Relationship Batching
        batch_relationship_data = []
        if dimension_node_ids_created: # Only proceed if nodes were created
            # Iterate through the original conceptual dimensions to prepare relationship data
            # using the mapping `created_dimensions_map`
            for i, dim_data in enumerate(dimensions_to_create_conceptual):
                original_dim_identifier = dim_data.get("id", dim_data.get("label", f"Dimension {i+1}"))
                created_dimension_id = created_dimensions_map.get(original_dim_identifier)
                
                if not created_dimension_id:
                    logger.warning(f"Could not find created Neo4j ID for original dimension identifier '{original_dim_identifier}'. Skipping relationship creation.")
                    continue

                dim_label_for_edge = dim_data.get("label", f"Dimension {i+1}") # For edge description

                edge_id = f"edge_{created_dimension_id}_decompof_{root_node_id}"
                edge_pydantic = Edge(
                    id=edge_id, source_id=created_dimension_id, target_id=root_node_id,
                    type=EdgeType.DECOMPOSITION_OF, confidence=0.95,
                    metadata=EdgeMetadata(description=f"'{dim_label_for_edge}' is a decomposition of '{decomposition_input_text[:30]}...'")
                )
                edge_props_for_neo4j = self._prepare_edge_properties_for_neo4j(edge_pydantic)
                batch_relationship_data.append({
                    "dim_id": created_dimension_id, 
                    "root_id": root_node_id, 
                    "props": edge_props_for_neo4j
                })
        
        if batch_relationship_data:
            try:
                batch_rel_query = """
                UNWIND $batch_rels AS rel_detail
                MATCH (dim_node:Node {id: rel_detail.dim_id})
                MATCH (root_node:Node {id: rel_detail.root_id})
                MERGE (dim_node)-[r:DECOMPOSITION_OF {id: rel_detail.props.id}]->(root_node)
                SET r += rel_detail.props
                RETURN count(r) AS total_rels_created
                """
                result_rels = execute_query(batch_rel_query, {"batch_rels": batch_relationship_data}, tx_type='write')
                if result_rels and result_rels[0].get("total_rels_created") is not None:
                    edges_created_count = result_rels[0]["total_rels_created"]
                    logger.debug(f"Batch created {edges_created_count} DECOMPOSITION_OF relationships.")
                else:
                    logger.error("Batch relationship creation query did not return expected count.")
            except Neo4jError as e:
                logger.error(f"Neo4j error during batch DECOMPOSITION_OF relationship creation: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during batch DECOMPOSITION_OF relationship creation: {e}")

        summary = f"Task decomposed into {nodes_created_count} dimensions in Neo4j: {', '.join(dimension_labels_created)}."
        metrics = {
            "dimensions_created_in_neo4j": nodes_created_count,
            "relationships_created_in_neo4j": edges_created_count,
        }
        # Ensure decomposition_results key is populated if other stages expect it.
        # The prompt for stage-skipping logic in got_processor.py assumes a "decomposition_results" key.
        # This key should contain the actual dimension data, not just IDs.
        # For now, dimension_node_ids_created is what we have. If fuller results are needed, this needs adjustment.
        decomposition_results_for_context = [
            {"id": node_id, "label": label} for node_id, label in zip(dimension_node_ids_created, dimension_labels_created)
        ]

        context_update = {
            "dimension_node_ids": dimension_node_ids_created,
            "decomposition_results": decomposition_results_for_context # Added based on got_processor expectation
        }
        
        output = StageOutput(
            summary=summary, metrics=metrics,
            next_stage_context_update={self.stage_name: context_update}
        )
        self._log_end(current_session_data.session_id, output)
        return output
