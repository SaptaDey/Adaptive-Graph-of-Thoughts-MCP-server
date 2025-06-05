from typing import Any, Optional, List, Dict, Set

from loguru import logger
from pydantic import BaseModel, Field

from asr_got_reimagined.config import Settings
from asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from asr_got_reimagined.domain.models.graph_elements import NodeType # Node removed as not used for in-memory graph
# from asr_got_reimagined.domain.models.graph_state import ASRGoTGraph # No longer used
from asr_got_reimagined.domain.services.neo4j_utils import execute_query, Neo4jError # Import Neo4j utils

from asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput


# Datetime might be needed if temporal_recency_days is implemented
# import datetime


# Pydantic model for defining a single subgraph extraction strategy
class SubgraphCriterion(BaseModel):
    name: str
    description: str
    min_avg_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_impact_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    node_types: Optional[List[NodeType]] = None
    include_disciplinary_tags: Optional[List[str]] = None
    exclude_disciplinary_tags: Optional[List[str]] = None
    layer_ids: Optional[List[str]] = None
    is_knowledge_gap: Optional[bool] = None
    include_neighbors_depth: int = Field(default=0, ge=0)


class ExtractedSubgraphData(BaseModel): # Renamed for clarity
    name: str
    description: str
    # Store lists of dicts for nodes and relationships as per output spec
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SubgraphExtractionStage(BaseStage):
    stage_name: str = "SubgraphExtractionStage"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.default_extraction_criteria: List[SubgraphCriterion] = [
            SubgraphCriterion(name="high_confidence_core", description="Nodes with high average confidence and impact.",
                              min_avg_confidence=getattr(self.default_params, "subgraph_min_confidence_threshold", 0.6),
                              min_impact_score=getattr(self.default_params, "subgraph_min_impact_threshold", 0.6),
                              node_types=[NodeType.HYPOTHESIS, NodeType.EVIDENCE, NodeType.INTERDISCIPLINARY_BRIDGE],
                              include_neighbors_depth=1),
            SubgraphCriterion(name="key_hypotheses_and_support", description="Key hypotheses and their direct support.",
                              node_types=[NodeType.HYPOTHESIS], min_avg_confidence=0.5, min_impact_score=0.5,
                              include_neighbors_depth=1),
            SubgraphCriterion(name="knowledge_gaps_focus", description="Identified knowledge gaps.",
                              is_knowledge_gap=True, node_types=[NodeType.PLACEHOLDER_GAP, NodeType.RESEARCH_QUESTION],
                              include_neighbors_depth=1),
        ]

    def _build_cypher_conditions_for_criterion(self, criterion: SubgraphCriterion, params: Dict[str, Any]) -> List[str]:
        """Builds a list of Cypher WHERE conditions based on the criterion."""
        conditions: List[str] = []
        # Assuming an overall average confidence property like 'confidence_overall_avg'
        # This needs to be calculated and stored by previous stages or derived if not present.
        # For now, let's use 'confidence_empirical_support' as a proxy if avg is not there.
        if criterion.min_avg_confidence is not None:
            conditions.append("(n.confidence_overall_avg >= $min_avg_confidence OR n.confidence_empirical_support >= $min_avg_confidence)")
            params["min_avg_confidence"] = criterion.min_avg_confidence
        if criterion.min_impact_score is not None:
            conditions.append("n.metadata_impact_score >= $min_impact_score")
            params["min_impact_score"] = criterion.min_impact_score
        if criterion.node_types:
            # Cypher needs labels, not n.type property for this kind of check usually
            # Assuming n.type property is used for NodeType enum value for now.
            type_conditions = []
            for i, nt in enumerate(criterion.node_types):
                param_name = f"node_type_{i}"
                type_conditions.append(f"n.type = ${param_name}") # Or use labels: "n:{nt.value}"
                params[param_name] = nt.value
            if type_conditions: conditions.append(f"({ ' OR '.join(type_conditions) })")

        if criterion.layer_ids:
            conditions.append("n.metadata_layer_id IN $layer_ids")
            params["layer_ids"] = criterion.layer_ids
        if criterion.is_knowledge_gap is not None: # Assuming 'metadata_is_knowledge_gap' boolean property
            conditions.append("n.metadata_is_knowledge_gap = $is_knowledge_gap")
            params["is_knowledge_gap"] = criterion.is_knowledge_gap
        if criterion.include_disciplinary_tags: # Assumes 'metadata_disciplinary_tags' is a list property
            # Check if any of the provided tags are in the node's tags list
            tag_conditions = []
            for i, tag in enumerate(criterion.include_disciplinary_tags):
                param_name = f"incl_tag_{i}"
                tag_conditions.append(f"${param_name} IN n.metadata_disciplinary_tags")
                params[param_name] = tag
            if tag_conditions: conditions.append(f"({ ' OR '.join(tag_conditions) })")
        if criterion.exclude_disciplinary_tags:
            tag_conditions = []
            for i, tag in enumerate(criterion.exclude_disciplinary_tags):
                param_name = f"excl_tag_{i}"
                tag_conditions.append(f"NOT (${param_name} IN n.metadata_disciplinary_tags)")
                params[param_name] = tag
            if tag_conditions: conditions.append(f"({ ' AND '.join(tag_conditions) })") # All excluded tags must not be present
        return conditions

    def _format_neo4j_node(self, neo4j_node_map: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a Neo4j node map (properties map + id + labels) into the desired output structure."""
        # Neo4j driver typically returns nodes as a Node object or a map.
        # Assuming execute_query returns a map like structure from APOC results or direct props.
        # If it's a direct Neo4j Node object, access props via `neo4j_node_map.items()`, labels via `neo4j_node_map.labels`.
        # For APOC results, it's usually a map.
        
        # This function assumes `neo4j_node_map` is already a dictionary representing the node.
        # If `execute_query` returns a list of `Node` objects from the driver,
        # this function would need to convert `n` to `properties(n)` and `labels(n)`.
        # The APOC query `RETURN n{.id, .labels, properties: properties(n)} AS node_map` is useful.
        
        # Let's assume APOC returns something like: {id: "id", labels: ["L1"], properties: {key:val}}
        # or if we query properties directly: {id:"id", label:"name", metadata_score:0.5, ...}
        # The target is: {"id": "node1", "labels": ["LABEL1", "Node"], "properties": {"prop1": "val1", ...}}

        # If APOC returns a structure like `n` (node object), then:
        # props = dict(neo4j_node_map.items()) # Get all properties
        # node_id = props.pop('id', neo4j_node_map.element_id) # Use 'id' property if exists, else element_id
        # labels = list(neo4j_node_map.labels)
        # return {"id": node_id, "labels": labels, "properties": props}
        
        # If APOC `subgraphAll` returns nodes that are maps (dictionaries):
        # It's common for APOC to return nodes as maps where properties are top-level.
        # We need to separate 'id' and 'labels' from other properties.
        
        # Assuming the query returns nodes as maps with 'id', 'labels', and 'properties' keys:
        # e.g., from `RETURN n {.id, .labels, properties:properties(n)} as node_data`
        if "properties" in neo4j_node_map and "id" in neo4j_node_map and "labels" in neo4j_node_map:
             return {
                "id": neo4j_node_map["id"],
                "labels": neo4j_node_map["labels"],
                "properties": neo4j_node_map["properties"]
            }
        # Fallback if properties are flat (less ideal, but possible from some queries)
        # This assumes 'id' is a property and we need to guess labels or they are part of props.
        # This part might need adjustment based on actual Cypher return structure.
        props_copy = dict(neo4j_node_map)
        node_id = props_copy.pop("id", None)
        labels = props_copy.pop("labels", ["Node"]) # Default if no labels field
        return {"id": node_id, "labels": labels, "properties": props_copy}


    def _format_neo4j_relationship(self, neo4j_rel_map: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a Neo4j relationship map into the desired output structure."""
        # Similar to nodes, depends on how relationships are returned.
        # APOC typically returns relationship objects or maps.
        # Target: {"id": "rel1", "type": "REL_TYPE", "source_id": "src_id", "target_id": "tgt_id", "properties": {...}}
        
        # Assuming APOC `subgraphAll` returns relationships as maps:
        # {id_prop: "rel_id", type: "TYPE", startNodeElementId: "start_id", endNodeElementId: "end_id", properties: {key:val}}
        # We need to map startNodeElementId/endNodeElementId to our 'id' property on nodes if they differ.
        # For now, assume startNode/endNode are our application IDs.
        
        # If query returns `RETURN r {.id_prop, .type, source_id:startNode(r).id, target_id:endNode(r).id, properties:properties(r)}`
        if "properties" in neo4j_rel_map and "id_prop" in neo4j_rel_map and "type" in neo4j_rel_map and "source_id" in neo4j_rel_map and "target_id" in neo4j_rel_map:
            return {
                "id": neo4j_rel_map["id_prop"], # Assuming 'id' property on relationship
                "type": neo4j_rel_map["type"],
                "source_id": neo4j_rel_map["source_id"],
                "target_id": neo4j_rel_map["target_id"],
                "properties": neo4j_rel_map["properties"]
            }
        # Fallback for simpler map structure
        props_copy = dict(neo4j_rel_map)
        rel_id = props_copy.pop("id", None) # Application-defined ID property
        rel_type = props_copy.pop("type", "RELATED_TO")
        source_id = props_copy.pop("source_id", None)
        target_id = props_copy.pop("target_id", None)
        return {"id": rel_id, "type": rel_type, "source_id": source_id, "target_id": target_id, "properties": props_copy}


    async def _extract_single_subgraph_from_neo4j(
        self, criterion: SubgraphCriterion
    ) -> ExtractedSubgraphData:
        seed_node_ids: Set[str] = set()
        params: Dict[str, Any] = {}
        conditions = self._build_cypher_conditions_for_criterion(criterion, params)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        seed_query = f"MATCH (n:Node) {where_clause} RETURN n.id AS id"
        
        extracted_nodes_dict: Dict[str, Dict[str, Any]] = {}
        extracted_rels_dict: Dict[str, Dict[str, Any]] = {}

        try:
            seed_results = await execute_query(seed_query, params, tx_type="read")
            if seed_results:
                seed_node_ids.update(record["id"] for record in seed_results if record.get("id"))

            if not seed_node_ids:
                logger.info(f"No seed nodes found for criterion '{criterion.name}'.")
                return ExtractedSubgraphData(name=criterion.name, description=criterion.description, metrics={"node_count":0, "relationship_count":0, "seed_node_count":0})

            logger.debug(f"Found {len(seed_node_ids)} seed nodes for '{criterion.name}'. Expanding with depth {criterion.include_neighbors_depth}.")

            # Use apoc.path.subgraphAll for each seed node and aggregate results
            # This can be inefficient if many seed nodes produce overlapping subgraphs.
            # A more optimized approach might use apoc.path.subgraphNodes with all seed_node_ids at once,
            # then separately query relationships if needed.
            # For now, iterative approach for clarity matching original expansion logic.
            
            # Let's use a single call to apoc.path.subgraphNodes for all seed nodes, then get relationships
            # This is more efficient than calling subgraphAll for each seed node.
            
            subgraph_query = """
            MATCH (seed_node:Node) WHERE seed_node.id IN $seed_ids
            CALL apoc.path.subgraphAll(seed_node, {maxLevel: $depth}) YIELD nodes, relationships
            UNWIND nodes AS n
            UNWIND relationships AS r
            // Return distinct nodes and relationships with their properties
            // Use map projection for consistent structure
            WITH collect(DISTINCT n) AS node_list, collect(DISTINCT r) AS rel_list
            RETURN [node IN node_list | node { .*, id_prop:node.id, labels_prop:labels(node), properties_prop:properties(node) }] AS extracted_nodes,
                   [rel IN rel_list | rel { .*, id_prop:rel.id, type_prop:type(rel), source_id_prop:startNode(rel).id, target_id_prop:endNode(rel).id, properties_prop:properties(rel)}] AS extracted_relationships
            """
            # The map projection above tries to capture everything.
            # Simpler if properties() includes id:
            # RETURN [n IN nodes | {id: n.id, labels: labels(n), properties: properties(n)}] AS extracted_nodes,
            #        [r IN relationships | {id: r.id, type: type(r), source_id: startNode(r).id, target_id: endNode(r).id, properties: properties(r)}] AS extracted_relationships

            # Using a slightly different APOC return structure to simplify processing:
            # We want to ensure 'id' is the primary identifier from our Node model, not elementId.
            # This query will return nodes and rels as maps, which is easier to process.
            # This is a common pattern with APOC when you want serializable graph structures.
            apoc_query = """
            UNWIND $seed_ids AS seedId
            MATCH (seed_node:Node {id: seedId})
            CALL apoc.path.subgraphAll(seed_node, {maxLevel: $max_level}) YIELD nodes, relationships
            RETURN nodes, relationships
            """
            
            # Aggregate results if apoc.path.subgraphAll is called per seed node (less efficient)
            # For now, let's assume a batch approach:
            
            batch_apoc_query = """
            MATCH (n:Node) WHERE n.id IN $seed_ids
            CALL apoc.path.subgraphNodes(n, {maxLevel: $max_level}) YIELD node
            // Get relationships involving these nodes
            WITH collect(node) AS subgraph_nodes
            UNWIND subgraph_nodes AS sn
            MATCH (sn)-[r]-(other_node)
            WHERE other_node IN subgraph_nodes // Ensure relationship is within the subgraph_nodes
            // Return nodes and relationships as maps
            RETURN [n_obj IN subgraph_nodes | {id: n_obj.id, labels: labels(n_obj), properties: properties(n_obj)}] AS final_nodes,
                   collect(DISTINCT {id: r.id, type: type(r), source_id: startNode(r).id, target_id: endNode(r).id, properties: properties(r)}) AS final_relationships
            """

            subgraph_results = await execute_query(batch_apoc_query, {"seed_ids": list(seed_node_ids), "max_level": criterion.include_neighbors_depth}, tx_type="read")

            if subgraph_results and subgraph_results[0]:
                raw_nodes = subgraph_results[0].get("final_nodes", [])
                raw_rels = subgraph_results[0].get("final_relationships", [])

                for node_map in raw_nodes:
                    fmt_node = self._format_neo4j_node(node_map)
                    if fmt_node["id"]: extracted_nodes_dict[fmt_node["id"]] = fmt_node
                
                for rel_map in raw_rels:
                    fmt_rel = self._format_neo4j_relationship(rel_map)
                    if fmt_rel["id"]: extracted_rels_dict[fmt_rel["id"]] = fmt_rel
            
        except Neo4jError as e:
            logger.error(f"Neo4j error extracting subgraph for criterion '{criterion.name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error during subgraph extraction for '{criterion.name}': {e}")

        final_nodes_list = list(extracted_nodes_dict.values())
        final_rels_list = list(extracted_rels_dict.values())

        logger.info(f"Extracted subgraph '{criterion.name}' with {len(final_nodes_list)} nodes and {len(final_rels_list)} relationships from Neo4j.")
        return ExtractedSubgraphData(
            name=criterion.name, description=criterion.description,
            nodes=final_nodes_list, relationships=final_rels_list,
            metrics={"node_count": len(final_nodes_list), "relationship_count": len(final_rels_list), "seed_node_count": len(seed_node_ids)}
        )

    async def execute(
        self, current_session_data: GoTProcessorSessionData # graph: ASRGoTGraph removed
    ) -> StageOutput:
        self._log_start(current_session_data.session_id)
        operational_params = current_session_data.accumulated_context.get("operational_params", {})
        custom_criteria_input = operational_params.get("subgraph_extraction_criteria")
        criteria_to_use: List[SubgraphCriterion] = []

        if isinstance(custom_criteria_input, list) and all(isinstance(c, dict) for c in custom_criteria_input):
            try:
                criteria_to_use = [SubgraphCriterion(**c) for c in custom_criteria_input]
                logger.info(f"Using {len(criteria_to_use)} custom subgraph extraction criteria.")
            except Exception as e:
                logger.warning(f"Failed to parse custom subgraph criteria: {e}. Using default criteria.")
                criteria_to_use = self.default_extraction_criteria
        else:
            criteria_to_use = self.default_extraction_criteria
            logger.info(f"Using {len(criteria_to_use)} default subgraph extraction criteria.")

        all_extracted_subgraphs_data: List[ExtractedSubgraphData] = []
        for criterion in criteria_to_use:
            try:
                subgraph_data = await self._extract_single_subgraph_from_neo4j(criterion)
                if subgraph_data.nodes: # Only add if non-empty
                    all_extracted_subgraphs_data.append(subgraph_data)
            except Exception as e: # Catch any unexpected error from the subgraph extraction
                logger.error(f"Error processing criterion '{criterion.name}': {e}")
                continue
        
        summary = f"Subgraph extraction complete. Extracted {len(all_extracted_subgraphs_data)} subgraphs based on {len(criteria_to_use)} criteria."
        total_nodes_extracted = sum(sg.metrics.get("node_count", 0) for sg in all_extracted_subgraphs_data)
        total_rels_extracted = sum(sg.metrics.get("relationship_count", 0) for sg in all_extracted_subgraphs_data)

        metrics = {
            "subgraphs_extracted_count": len(all_extracted_subgraphs_data),
            "total_criteria_evaluated": len(criteria_to_use),
            "total_nodes_in_extracted_subgraphs": total_nodes_extracted,
            "total_relationships_in_extracted_subgraphs": total_rels_extracted
        }
        for sg_data in all_extracted_subgraphs_data:
             metrics[f"subgraph_{sg_data.name}_node_count"] = sg_data.metrics.get("node_count",0)
             metrics[f"subgraph_{sg_data.name}_relationship_count"] = sg_data.metrics.get("relationship_count",0)

        context_update = {
            "subgraph_extraction_results": { # As per spec
                 "subgraphs": [sg.model_dump() for sg in all_extracted_subgraphs_data]
            }
        }
        # If the spec meant a flatter structure:
        # context_update = {
        #     "extracted_nodes": list_of_all_distinct_nodes_across_subgraphs,
        #     "extracted_relationships": list_of_all_distinct_rels_across_subgraphs 
        # }
        # The current implementation returns nodes/rels per named subgraph, which seems more aligned with `ExtractedSubgraphData`.

        return StageOutput(summary=summary, metrics=metrics, next_stage_context_update={self.stage_name: context_update})
