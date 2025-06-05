import random
from typing import Optional, Union, Dict, List, Any

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from asr_got_reimagined.config import Settings
from asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from asr_got_reimagined.domain.models.graph_elements import ( # Node might not be needed if processing dicts directly
    NodeType, # Still useful for type checking
)
# from asr_got_reimagined.domain.models.graph_state import ASRGoTGraph # No longer used
from asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput
from asr_got_reimagined.domain.stages.stage_6_subgraph_extraction import SubgraphExtractionStage # For context key
# No need to import ExtractedSubgraph if we use dicts or define a local version matching the input structure

import datetime # For parsing ISO date strings if necessary


# --- Pydantic models for structured output of Composition Stage ---
class CitationItem(
    BaseModel
):  # P1.6 Vancouver citations (K1.3 implies a specific style)
    id: Union[str, int]
    text: str 
    source_node_id: Optional[str] = None
    url: Optional[str] = None


class OutputSection(BaseModel):
    title: str
    content: str 
    type: str = Field(default="generic", examples=["summary", "analysis", "findings", "gaps", "interdisciplinary"])
    referenced_subgraph_name: Optional[str] = None
    related_node_ids: List[str] = Field(default_factory=list)


class ComposedOutput(BaseModel):
    title: str
    executive_summary: str
    sections: List[OutputSection] = Field(default_factory=list)
    citations: List[CitationItem] = Field(default_factory=list)
    reasoning_trace_appendix_summary: Optional[str] = None
    graph_topology_summary: Optional[str] = None

# Define a local version of ExtractedSubgraphData to guide parsing if not centrally defined
# This should match the output structure of SubgraphExtractionStage
class LocalExtractedSubgraphData(BaseModel):
    name: str
    description: str
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class CompositionStage(BaseStage):
    stage_name: str = "CompositionStage"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.citation_style = "Vancouver"

    async def _generate_executive_summary(
        self,
        extracted_subgraphs_data: List[LocalExtractedSubgraphData], # Changed type
        initial_query: str,
    ) -> str:
        num_subgraphs = len(extracted_subgraphs_data)
        subgraph_names = [sg.name for sg in extracted_subgraphs_data] # Access .name from parsed object
        summary = (
            f"Executive summary for the analysis of query: '{initial_query}'.\n"
            f"The ASR-GoT process identified {num_subgraphs} key subgraphs of interest: {', '.join(subgraph_names)}. "
            f"These subgraphs highlight various facets of the research topic, including "
            f"{', '.join(random.sample(subgraph_names, min(2, num_subgraphs)) if subgraph_names else ['key findings'])}. "
            f"Further details are provided in the subsequent sections."
        )
        logger.debug("Generated placeholder executive summary.")
        return summary

    async def _format_node_dict_as_claim(
        self,
        node_dict: Dict[str, Any], # Changed from Node Pydantic object to dict
    ) -> tuple[str, Optional[CitationItem]]:
        node_id = node_dict.get("id", "UnknownID")
        properties = node_dict.get("properties", {})
        node_label = properties.get("label", "Unknown Label")
        # Infer type from labels or properties.type
        node_type_str = properties.get("type") 
        if not node_type_str and "labels" in node_dict: # Try inferring from labels if type property not set
            # Example: if labels = ["HYPOTHESIS", "Node"], prefer "HYPOTHESIS"
            # This logic might need refinement based on how labels are structured by SubgraphExtraction
            labels = node_dict.get("labels")
            if labels:
                specific_labels = [l for l in labels if l != "Node"]
                node_type_str = specific_labels[0] if specific_labels else "UnknownType"
            else:
                node_type_str = "UnknownType"
        node_type_val = node_type_str or "UnknownType" # Fallback

        claim_text = f"Claim based on Node {node_id} ('{node_label}', Type: {node_type_val}): "
        
        description = properties.get("metadata_description", "") # Access nested property
        if description:
            claim_text += description[:100] + "..."

        created_at_iso = properties.get("metadata_created_at_iso", properties.get("created_at_iso")) # Check common places
        created_at_str = "Unknown Date"
        if created_at_iso:
            try:
                created_at_dt = datetime.datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
                created_at_str = created_at_dt.strftime('%Y-%m-%d')
            except ValueError:
                logger.warning(f"Could not parse date: {created_at_iso} for node {node_id}")
        
        citation_text = f"NexusMind Internal Node. ID: {node_id}. Label: {node_label}. Type: {node_type_val}. Created: {created_at_str}."
        citation = CitationItem(id=f"Node-{node_id}", text=citation_text, source_node_id=node_id)
        return f"{claim_text} [{citation.id}]", citation

    async def _generate_section_from_subgraph_data( # Renamed and signature changed
        self, subgraph_data: LocalExtractedSubgraphData # Changed from ASRGoTGraph and ExtractedSubgraph
    ) -> tuple[OutputSection, List[CitationItem]]:
        section_title = f"Analysis: {subgraph_data.name.replace('_', ' ').title()}"
        content_parts: List[str] = [
            f"This section discusses findings from the '{subgraph_data.name}' subgraph, which focuses on: {subgraph_data.description}.\n"
        ]
        citations: List[CitationItem] = []
        related_node_ids_for_section: List[str] = [n.get("id", "") for n in subgraph_data.nodes if n.get("id")]

        key_nodes_in_subgraph: List[Dict[str, Any]] = []
        for node_dict in subgraph_data.nodes:
            props = node_dict.get("properties", {})
            node_type_str = props.get("type")
            # Convert string type to NodeType enum member if valid, otherwise skip or handle
            try:
                node_type_enum = NodeType(node_type_str) if node_type_str else None
            except ValueError:
                node_type_enum = None # Unknown type

            # Example: Calculate average confidence from components if overall not present
            avg_confidence = props.get("confidence_overall_avg", 0.0)
            if avg_confidence == 0.0: # Try to compute from components
                conf_components = [
                    props.get("confidence_empirical_support", 0.0),
                    props.get("confidence_theoretical_basis", 0.0), # Assuming these names
                    props.get("confidence_methodological_rigor", 0.0),
                    props.get("confidence_consensus_alignment", 0.0)
                ]
                avg_confidence = sum(c for c in conf_components if isinstance(c, (int,float))) / len([c for c in conf_components if isinstance(c, (int,float))]) if any(isinstance(c, (int,float)) for c in conf_components) else 0.0

            impact_score = props.get("metadata_impact_score", 0.0)

            if node_type_enum in [NodeType.HYPOTHESIS, NodeType.EVIDENCE, NodeType.INTERDISCIPLINARY_BRIDGE] and \
               (avg_confidence > 0.6 or impact_score > 0.6):
                # Store the full node_dict for _format_node_dict_as_claim
                node_dict["calculated_avg_confidence"] = avg_confidence # Store for sorting
                key_nodes_in_subgraph.append(node_dict)
        
        key_nodes_in_subgraph.sort(key=lambda n_dict: (-(n_dict.get("properties", {}).get("metadata_impact_score", 0.0)), -n_dict.get("calculated_avg_confidence", 0.0)))

        for i, node_d in enumerate(key_nodes_in_subgraph[:3]):
            claim_text, citation = await self._format_node_dict_as_claim(node_d)
            content_parts.append(f"Key Point {i + 1}: {claim_text}")
            if citation: citations.append(citation)
            
            # Simplified relationship listing (if relationships are part of subgraph_data)
            node_id = node_d.get("id")
            if node_id and subgraph_data.relationships:
                incoming_rels_desc = []
                outgoing_rels_desc = []
                for rel_dict in subgraph_data.relationships:
                    rel_props = rel_dict.get("properties", {})
                    rel_type = rel_dict.get("type", "RELATED")
                    if rel_dict.get("target_id") == node_id:
                        incoming_rels_desc.append(f"{rel_type} from {rel_dict.get('source_id')}")
                    if rel_dict.get("source_id") == node_id:
                        outgoing_rels_desc.append(f"{rel_type} to {rel_dict.get('target_id')}")
                
                if incoming_rels_desc: content_parts.append(f"  - Connected from: {', '.join(incoming_rels_desc[:2])}")
                if outgoing_rels_desc: content_parts.append(f"  - Connects to: {', '.join(outgoing_rels_desc[:2])}")


        if not key_nodes_in_subgraph:
            content_parts.append("No specific high-impact claims identified in this subgraph based on current criteria.")

        section = OutputSection(title=section_title, content="\n".join(content_parts), type="analysis_subgraph",
                                referenced_subgraph_name=subgraph_data.name, related_node_ids=related_node_ids_for_section)
        logger.debug(f"Generated content for section: '{section_title}'.")
        return section, citations

    async def _generate_reasoning_trace_appendix_summary(self, session_data: GoTProcessorSessionData) -> str:
        lines = ["Summary of Reasoning Trace Appendix:"]
        for trace_item in session_data.stage_outputs_trace:
            lines.append(f"  Stage {trace_item['stage_number']}. {trace_item['stage_name']}: {trace_item['summary']} ({trace_item.get('duration_ms', 'N/A')}ms)")
        return "\n".join(lines)

    async def execute(
        self, current_session_data: GoTProcessorSessionData # graph: ASRGoTGraph removed
    ) -> StageOutput:
        self._log_start(current_session_data.session_id)

        subgraph_extraction_results_dict = current_session_data.accumulated_context.get(SubgraphExtractionStage.stage_name, {})
        # The actual list of subgraphs is nested under "subgraphs" key
        extracted_subgraphs_raw_data: List[Dict[str,Any]] = subgraph_extraction_results_dict.get("subgraph_extraction_results", {}).get("subgraphs", [])


        parsed_subgraphs: List[LocalExtractedSubgraphData] = []
        if extracted_subgraphs_raw_data:
            try:
                for data_dict in extracted_subgraphs_raw_data:
                    # Validate and parse each subgraph dictionary using the local Pydantic model
                    parsed_subgraphs.append(LocalExtractedSubgraphData(**data_dict))
            except ValidationError as e:
                logger.error(f"Error parsing extracted subgraph definitions: {e}")
                # Continue with successfully parsed ones or return error if critical
        
        initial_query = current_session_data.query

        if not parsed_subgraphs:
            logger.warning("No subgraphs parsed successfully. Composition will be minimal.")
            composed_output_obj = ComposedOutput(
                title=f"NexusMind Analysis (Minimal): {initial_query[:50]}...",
                executive_summary="No specific subgraphs were extracted or parsed for detailed composition.",
                reasoning_trace_appendix_summary=await self._generate_reasoning_trace_appendix_summary(current_session_data)
            )
            return StageOutput(summary="Composition complete (minimal due to no/invalid subgraphs).",
                               metrics={"sections_generated": 0, "citations_generated": 0},
                               next_stage_context_update={self.stage_name: {"final_composed_output": composed_output_obj.model_dump()}})

        all_citations: List[CitationItem] = []
        output_sections: List[OutputSection] = []

        exec_summary = await self._generate_executive_summary(parsed_subgraphs, initial_query)

        for subgraph_data_obj in parsed_subgraphs: # Iterate over parsed Pydantic objects
            try:
                section, section_citations = await self._generate_section_from_subgraph_data(subgraph_data_obj)
                output_sections.append(section)
                all_citations.extend(section_citations)
            except Exception as e:
                logger.error(f"Error generating section for subgraph '{subgraph_data_obj.name}': {e}")
        
        final_citations_map: Dict[str, CitationItem] = {}
        for cit in all_citations:
            key = str(cit.id) 
            if key not in final_citations_map: final_citations_map[key] = cit
        final_citations = list(final_citations_map.values())

        trace_appendix_summary = await self._generate_reasoning_trace_appendix_summary(current_session_data)

        composed_output_obj = ComposedOutput(
            title=f"NexusMind Analysis: {initial_query[:50]}...",
            executive_summary=exec_summary, sections=output_sections, citations=final_citations,
            reasoning_trace_appendix_summary=trace_appendix_summary,
        )

        summary = f"Composed final output with {len(output_sections)} sections and {len(final_citations)} citations."
        metrics = {"sections_generated": len(output_sections), "citations_generated": len(final_citations),
                   "subgraphs_processed": len(parsed_subgraphs)}
        
        context_update = {"final_composed_output": composed_output_obj.model_dump()}

        return StageOutput(summary=summary, metrics=metrics, next_stage_context_update={self.stage_name: context_update})
