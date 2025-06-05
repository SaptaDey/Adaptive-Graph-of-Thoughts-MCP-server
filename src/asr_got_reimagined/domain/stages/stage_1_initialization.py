import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict, Set, Union

from loguru import logger
from pydantic import BaseModel

from asr_got_reimagined.config import Settings
from asr_got_reimagined.domain.models.common import (
    ConfidenceVector,
    EpistemicStatus,
)
from asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from asr_got_reimagined.domain.models.graph_elements import (
    Node,
    NodeMetadata,
    NodeType,
    Edge,
    EdgeType,
    EdgeMetadata,
    FalsificationCriteria, # Assuming this is a Pydantic model used in metadata
    BiasFlag,
)
# from asr_got_reimagined.domain.models.graph_state import ASRGoTGraph # No longer used
from asr_got_reimagined.domain.services.neo4j_utils import execute_query, Neo4jError

from asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput

# T = TypeVar("T", bound=BaseModel) # No longer needed here as rehydration helpers are removed

class InitializationStage(BaseStage):
    stage_name: str = "InitializationStage"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.root_node_label = "Task Understanding"
        self.initial_confidence_values = self.default_params.initial_confidence
        self.initial_layer = self.default_params.initial_layer

    def _prepare_node_properties_for_neo4j(self, node_pydantic: Node) -> Dict[str, Any]:
        """
        Converts a Node Pydantic model instance into a flat dictionary suitable for Neo4j properties.
        Handles datetime, Enum, ConfidenceVector, and nested NodeMetadata.
        """
        if node_pydantic is None:
            return {}

        props = {"id": node_pydantic.id, "label": node_pydantic.label} # Type is handled by label in query

        # ConfidenceVector flattening
        if node_pydantic.confidence:
            for cv_field, cv_val in node_pydantic.confidence.model_dump().items():
                if cv_val is not None:
                    props[f"confidence_{cv_field}"] = cv_val
        
        # NodeMetadata flattening
        if node_pydantic.metadata:
            for meta_field_name, meta_value in node_pydantic.metadata.model_dump().items():
                if meta_value is None:
                    # props[f"metadata_{meta_field_name}"] = None # Neo4j handles missing fields better
                    continue
                
                if isinstance(meta_value, datetime):
                    props[f"metadata_{meta_field_name}"] = meta_value.isoformat()
                elif isinstance(meta_value, Enum):
                    props[f"metadata_{meta_field_name}"] = meta_value.value
                elif isinstance(meta_value, (list, set)):
                     # Ensure all items are simple types or convert Pydantic models in list to dicts/JSON
                    if all(isinstance(item, (str, int, float, bool)) for item in meta_value):
                        props[f"metadata_{meta_field_name}"] = list(meta_value)
                    else: # List of complex objects (e.g., Pydantic models like FalsificationCriteria)
                        try:
                            # Attempt to dump each model in the list, then dump the whole list to JSON
                            # This assumes items in the list are Pydantic models themselves.
                            items_as_dicts = [item.model_dump() if hasattr(item, 'model_dump') else item for item in meta_value]
                            props[f"metadata_{meta_field_name}_json"] = json.dumps(items_as_dicts)
                        except TypeError as e:
                            logger.warning(f"Could not serialize list/set metadata field {meta_field_name} to JSON: {e}")
                            props[f"metadata_{meta_field_name}_str"] = str(meta_value) # Fallback
                elif hasattr(meta_value, 'model_dump'): # Other nested Pydantic models in metadata
                    try:
                        props[f"metadata_{meta_field_name}_json"] = json.dumps(meta_value.model_dump())
                    except TypeError as e:
                        logger.warning(f"Could not serialize metadata field {meta_field_name} to JSON: {e}")
                        props[f"metadata_{meta_field_name}_str"] = str(meta_value) # Fallback
                else:
                    props[f"metadata_{meta_field_name}"] = meta_value
        
        # Clean out None values explicitly, Neo4j handles missing fields better than properties with None
        return {k: v for k, v in props.items() if v is not None}

    async def execute(
        self, current_session_data: GoTProcessorSessionData # graph: ASRGoTGraph parameter removed
    ) -> StageOutput:
        self._log_start(current_session_data.session_id)
        initial_query = current_session_data.query
        operational_params = current_session_data.accumulated_context.get("operational_params", {})
        
        nodes_created_in_neo4j = 0
        used_existing_neo4j_node = False
        updated_existing_node_tags = False
        root_node_id_for_context: Optional[str] = None
        
        final_summary_message: str
        initial_disciplinary_tags_for_context: List[str]

        # Validate initial query
        if not initial_query or not isinstance(initial_query, str):
            error_message = "Invalid initial query. It must be a non-empty string."
            logger.error(error_message)
            return StageOutput(
                summary=error_message,
                metrics={"nodes_created_in_neo4j": 0, "used_existing_neo4j_node": False, "updated_existing_node_tags": False},
                next_stage_context_update={self.stage_name: {"error": error_message}},
            )

        logger.info(f"Attempting to find or create ROOT node in Neo4j for query: '{initial_query[:100]}...'")
        try:
            # 1. Find an existing ROOT node matching the query
            find_root_query = """
            MATCH (n:ROOT)
            WHERE n.metadata_query_context = $initial_query
            RETURN n.id AS nodeId, n.metadata_disciplinary_tags AS current_tags
            LIMIT 1
            """
            # Assuming execute_query can be awaited if it's async, or called directly if sync
            # For now, calling it directly as per the tool's current capabilities for neo4j_utils
            root_node_records = await execute_query(find_root_query, {"initial_query": initial_query}, tx_type="read")

            if root_node_records:
                root_record = root_node_records[0]
                root_node_id_for_context = root_record["nodeId"]
                used_existing_neo4j_node = True
                logger.info(f"Found existing ROOT node '{root_node_id_for_context}' in Neo4j matching query.")

                current_tags_from_db = set(root_record.get("current_tags") or [])
                newly_provided_tags = set(operational_params.get("initial_disciplinary_tags", []))
                
                combined_tags = current_tags_from_db.union(newly_provided_tags)

                if combined_tags != current_tags_from_db:
                    update_tags_query = """
                    MATCH (n:ROOT {id: $node_id})
                    SET n.metadata_disciplinary_tags = $tags
                    RETURN n.metadata_disciplinary_tags AS updated_tags
                    """
                    updated_tags_result = await execute_query(
                         update_tags_query, 
                         {"node_id": root_node_id_for_context, "tags": list(combined_tags)}, 
                         tx_type="write"
                     )
                    if updated_tags_result:
                        logger.info(f"Updated disciplinary tags for ROOT node '{root_node_id_for_context}' to: {updated_tags_result[0]['updated_tags']}")
                        updated_existing_node_tags = True
                        initial_disciplinary_tags_for_context = list(combined_tags)
                    else:
                        logger.warning(f"Failed to update tags for ROOT node '{root_node_id_for_context}'. Using existing tags.")
                        initial_disciplinary_tags_for_context = list(current_tags_from_db)
                else:
                    logger.info(f"No change in disciplinary tags for existing ROOT node '{root_node_id_for_context}'.")
                    initial_disciplinary_tags_for_context = list(current_tags_from_db)
                
                final_summary_message = f"Using existing ROOT node '{root_node_id_for_context}' from Neo4j. Disciplinary tags ensured."

            else: # No existing ROOT node found, create one
                logger.info("No existing ROOT node found in Neo4j. Creating a new one.")
                new_root_node_id_internal = uuid.uuid4().hex # Unique ID to avoid collisions across sessions
                
                default_disciplines = set(operational_params.get(
                    "initial_disciplinary_tags", 
                    self.settings.asr_got.default_parameters.default_disciplinary_tags
                ))
                initial_disciplinary_tags_for_context = list(default_disciplines)

                root_metadata_pydantic = NodeMetadata(
                    description=f"Initial understanding of the task based on the query: '{initial_query}'.",
                    query_context=initial_query,
                    source_description="Core GoT Protocol Definition (P1.1), User Query",
                    epistemic_status=EpistemicStatus.ASSUMPTION,
                    disciplinary_tags=initial_disciplinary_tags_for_context,
                    layer_id=operational_params.get("initial_layer", self.initial_layer),
                    impact_score=0.9,
                )
                root_node_pydantic = Node(
                    id=new_root_node_id_internal,
                    label=self.root_node_label,
                    type=NodeType.ROOT,
                    confidence=ConfidenceVector.from_list(self.initial_confidence_values),
                    metadata=root_metadata_pydantic,
                )
                
                node_props_for_neo4j = self._prepare_node_properties_for_neo4j(root_node_pydantic)
                
                create_query = """
                MERGE (n:Node {id: $props.id})
                SET n += $props
                WITH n, $type_label AS typeLabel
                CALL apoc.create.addLabels(n, [typeLabel]) YIELD node
                RETURN node.id AS new_node_id
                """
                query_params = {"props": node_props_for_neo4j, "type_label": NodeType.ROOT.value}
                # await execute_query(...) if execute_query becomes async
                creation_result = await execute_query(create_query, query_params, tx_type='write')
                
                if creation_result and creation_result[0].get("new_node_id"):
                    root_node_id_for_context = creation_result[0]["new_node_id"]
                    nodes_created_in_neo4j = 1
                    logger.info(f"New ROOT node '{root_node_id_for_context}' created in Neo4j.")
                    final_summary_message = f"New ROOT node '{root_node_id_for_context}' created in Neo4j."
                else:
                    # Fallback or error, though MERGE should ensure node existence
                    error_message = "Failed to create or verify new ROOT node in Neo4j."
                    logger.error(error_message + f" Query: {create_query}, Params: {query_params}")
                    # Return error StageOutput
                    return StageOutput(
                        summary=error_message,
                        metrics={"nodes_created_in_neo4j": 0, "used_existing_neo4j_node": False, "updated_existing_node_tags": False},
                        next_stage_context_update={self.stage_name: {"error": error_message}},
                    )

        except Neo4jError as e:
            error_message = f"Neo4j error during ROOT node initialization: {e}"
            logger.error(error_message)
            return StageOutput(
                summary=error_message,
                metrics={"nodes_created_in_neo4j": 0, "used_existing_neo4j_node": False, "updated_existing_node_tags": False},
                next_stage_context_update={self.stage_name: {"error": error_message}},
            )
        except Exception as e: # Catch any other unexpected errors
            error_message = f"Unexpected error during ROOT node initialization: {e}"
            logger.exception(error_message) # Log with stack trace
            return StageOutput(
                summary=error_message,
                metrics={"nodes_created_in_neo4j": 0, "used_existing_neo4j_node": False, "updated_existing_node_tags": False},
                next_stage_context_update={self.stage_name: {"error": error_message}},
            )
        
        if not root_node_id_for_context:
            error_message = "Critical error: No root_node_id established after Neo4j operations."
            logger.error(error_message)
            return StageOutput(
                summary=error_message,
                metrics={"nodes_created_in_neo4j": nodes_created_in_neo4j, 
                           "used_existing_neo4j_node": used_existing_neo4j_node, 
                           "updated_existing_node_tags": updated_existing_node_tags},
                next_stage_context_update={self.stage_name: {"error": error_message}},
            )

        context_update = {
            "root_node_id": root_node_id_for_context,
            "initial_disciplinary_tags": initial_disciplinary_tags_for_context,
        }
        
        # Since we don't have the node Pydantic object directly without re-fetching,
        # initial_confidence_avg cannot be easily calculated here unless fetching is too costly.
        # For simplicity, this metric might be removed or adapted if fetching is too costly.
        # For now, setting to a placeholder or 0.
        initial_confidence_avg_metric = 0.0 # Placeholder

        metrics = {
            "nodes_created_in_neo4j": nodes_created_in_neo4j,
            "used_existing_neo4j_node": used_existing_neo4j_node,
            "updated_existing_node_tags": updated_existing_node_tags,
            "initial_confidence_avg": initial_confidence_avg_metric, # Placeholder
            # "layer_count_initialized": 0, # This was related to ASRGoTGraph, no longer applicable here
        }

        output = StageOutput(
            summary=final_summary_message,
            metrics=metrics,
            next_stage_context_update={self.stage_name: context_update},
        )
        self._log_end(current_session_data.session_id, output)
        return output
