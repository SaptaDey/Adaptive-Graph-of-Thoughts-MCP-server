from typing import Any, Optional, Dict, List

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from asr_got_reimagined.config import Settings
from asr_got_reimagined.domain.models.common import ConfidenceVector
from asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from asr_got_reimagined.domain.models.graph_elements import (
    NodeType, # Still useful for type checking
    BiasFlag, # For parsing bias_flags_json
    FalsificationCriteria, # For parsing falsification_criteria_json
    StatisticalPower, # For parsing statistical_power_json
)
# from asr_got_reimagined.domain.models.graph_state import ASRGoTGraph # No longer used
from asr_got_reimagined.domain.services.neo4j_utils import execute_query, Neo4jError # Import Neo4j utils

from asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput
from asr_got_reimagined.domain.stages.stage_7_composition import ( 
    ComposedOutput, # For parsing composed_output_dict
    CompositionStage, # For context key
)
import json # For parsing JSON string properties from Neo4j


# Structure for audit check results
class AuditCheckResult(BaseModel):
    check_name: str
    status: str = Field(
        default="NOT_RUN", examples=["PASS", "WARNING", "FAIL", "NOT_APPLICABLE"]
    )
    message: str
    details: Optional[Dict[str, Any]] = None


class ReflectionStage(BaseStage):
    stage_name: str = "ReflectionStage"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.high_confidence_threshold = getattr(self.default_params, "high_confidence_threshold", 0.7)
        self.high_impact_threshold = getattr(self.default_params, "high_impact_threshold", 0.7)
        self.min_falsifiable_hypothesis_ratio = getattr(self.default_params, "min_falsifiable_hypothesis_ratio", 0.6)
        self.max_high_severity_bias_nodes = getattr(self.default_params, "max_high_severity_bias_nodes", 0)
        self.min_powered_evidence_ratio = getattr(self.default_params, "min_powered_evidence_ratio", 0.5)
        self.audit_checklist_items = [ # These are conceptual names for checks
            "high_confidence_impact_coverage", "bias_flags_assessment", "knowledge_gaps_addressed",
            "hypothesis_falsifiability", "causal_claim_validity", "temporal_consistency",
            "statistical_rigor_of_evidence", "collaboration_attributions_check",
        ]

    async def _check_high_confidence_impact_coverage_from_neo4j(self) -> AuditCheckResult:
        query = """
        MATCH (n:Node)
        WHERE n.type IN ['HYPOTHESIS', 'EVIDENCE', 'INTERDISCIPLINARY_BRIDGE']
        RETURN n.confidence_overall_avg AS avg_confidence, // Assuming this field exists
               n.metadata_impact_score AS impact_score
        """
        # If confidence_overall_avg is not stored, fetch components:
        # RETURN n.confidence_empirical_support AS emp, n.confidence_theoretical_basis AS theo, ..., n.metadata_impact_score AS impact
        try:
            results = await execute_query(query, {}, tx_type="read")
            if not results:
                return AuditCheckResult(check_name="high_confidence_impact_coverage", status="NOT_APPLICABLE", message="No relevant nodes found.")

            high_conf_nodes = 0
            high_impact_nodes = 0
            total_relevant_nodes = len(results)

            for record in results:
                # Placeholder: if avg_confidence is not directly available, calculate from components
                # avg_conf = (record.get('emp',0) + record.get('theo',0) + ...) / num_components
                avg_conf = record.get("avg_confidence", 0.0) # Use direct field if available
                impact = record.get("impact_score", 0.0)
                if avg_conf >= self.high_confidence_threshold: high_conf_nodes += 1
                if impact >= self.high_impact_threshold: high_impact_nodes += 1
            
            conf_coverage = high_conf_nodes / total_relevant_nodes if total_relevant_nodes else 0
            impact_coverage = high_impact_nodes / total_relevant_nodes if total_relevant_nodes else 0
            message = f"Confidence coverage: {conf_coverage:.2%}. Impact coverage: {impact_coverage:.2%}."
            status = "PASS" if conf_coverage >= 0.3 and impact_coverage >= 0.2 else ("WARNING" if conf_coverage >=0.1 or impact_coverage >= 0.1 else "FAIL")
            return AuditCheckResult(check_name="high_confidence_impact_coverage", status=status, message=message)
        except Neo4jError as e:
            logger.error(f"Neo4j error in confidence/impact check: {e}")
            return AuditCheckResult(check_name="high_confidence_impact_coverage", status="FAIL", message=f"Query error: {e}")


    async def _check_bias_flags_assessment_from_neo4j(self) -> AuditCheckResult:
        query = """
        MATCH (n:Node) WHERE n.metadata_bias_flags_json IS NOT NULL
        RETURN n.metadata_bias_flags_json AS bias_flags_json
        """
        try:
            results = await execute_query(query, {}, tx_type="read")
            flagged_nodes_count = 0
            high_severity_bias_count = 0
            if results:
                flagged_nodes_count = len(results)
                for record in results:
                    bias_flags_list = json.loads(record["bias_flags_json"])  # Assuming it's a JSON string of a list of dicts
                    for flag_dict in bias_flags_list:
                        bias_flag = BiasFlag(**flag_dict)  # Parse into Pydantic model
                        if bias_flag.severity == "high":
                            high_severity_bias_count += 1

            message = f"Found {flagged_nodes_count} nodes with bias flags. {high_severity_bias_count} have high severity."
            status = "FAIL" if high_severity_bias_count > self.max_high_severity_bias_nodes else ("WARNING" if flagged_nodes_count > 0 else "PASS")
            return AuditCheckResult(check_name="bias_flags_assessment", status=status, message=message)
        except (Neo4jError, json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Error in bias flags check: {e}")
            return AuditCheckResult(check_name="bias_flags_assessment", status="FAIL", message=f"Error processing bias flags: {e}")

    async def _check_knowledge_gaps_addressed_from_neo4j(self, composed_output: Optional[ComposedOutput]) -> AuditCheckResult:
        query = "MATCH (g:Node) WHERE g.metadata_is_knowledge_gap = true RETURN count(g) as gap_nodes_count"
        gap_nodes_present = False
        try:
            results = await execute_query(query, {}, tx_type="read")
            if results and results[0]["gap_nodes_count"] > 0:
                gap_nodes_present = True
        except Neo4jError as e:
            logger.error(f"Neo4j error checking knowledge gaps: {e}")
            return AuditCheckResult(check_name="knowledge_gaps_addressed", status="FAIL", message=f"Query error: {e}")

        gaps_mentioned_in_output = False
        if composed_output:
            for section in (composed_output.sections or []):
                if "gap" in section.title.lower() or (section.type and "gap" in section.type.lower()):
                    gaps_mentioned_in_output = True; break
        
        if not gap_nodes_present: return AuditCheckResult(check_name="knowledge_gaps_addressed", status="NOT_APPLICABLE", message="No explicit knowledge gap nodes in graph.")
        status = "PASS" if gaps_mentioned_in_output else "WARNING"
        message = "Knowledge gaps found in graph were addressed in output." if status == "PASS" else "Knowledge gaps found but might not be explicitly in output."
        return AuditCheckResult(check_name="knowledge_gaps_addressed", status=status, message=message)

    async def _check_hypothesis_falsifiability_from_neo4j(self) -> AuditCheckResult:
        query = """
        MATCH (h:Node:HYPOTHESIS) 
        RETURN h.metadata_falsification_criteria_json IS NOT NULL AS has_criteria
        """
        try:
            results = await execute_query(query, {}, tx_type="read")
            if not results: return AuditCheckResult(check_name="hypothesis_falsifiability", status="NOT_APPLICABLE", message="No hypotheses found.")
            
            total_hypotheses = len(results)
            falsifiable_count = sum(1 for r in results if r["has_criteria"])
            ratio = falsifiable_count / total_hypotheses if total_hypotheses else 0
            message = f"{falsifiable_count}/{total_hypotheses} ({ratio:.2%}) hypotheses have falsifiability criteria."
            status = "PASS" if ratio >= self.min_falsifiable_hypothesis_ratio else ("WARNING" if ratio > 0 else "FAIL")
            return AuditCheckResult(check_name="hypothesis_falsifiability", status=status, message=message)
        except Neo4jError as e:
            logger.error(f"Neo4j error in falsifiability check: {e}")
            return AuditCheckResult(check_name="hypothesis_falsifiability", status="FAIL", message=f"Query error: {e}")

    async def _check_statistical_rigor_from_neo4j(self) -> AuditCheckResult:
        query = """
        MATCH (e:Node:EVIDENCE) 
        RETURN e.metadata_statistical_power_json AS stat_power_json
        """
        try:
            results = await execute_query(query, {}, tx_type="read")
            if not results: return AuditCheckResult(check_name="statistical_rigor_of_evidence", status="NOT_APPLICABLE", message="No evidence nodes.")

            total_evidence = len(results)
            adequately_powered_count = 0
            for record in results:
                if record["stat_power_json"]:
                    try:
                        stat_power_obj = StatisticalPower(**json.loads(record["stat_power_json"]))
                        if stat_power_obj.value >= 0.7: adequately_powered_count +=1
                    except (json.JSONDecodeError, ValidationError) as e_parse:
                        logger.warning(f"Could not parse statistical_power_json: {e_parse}")
            
            ratio = adequately_powered_count / total_evidence if total_evidence else 0
            message = f"{adequately_powered_count}/{total_evidence} ({ratio:.2%}) evidence nodes meet power criteria (>=0.7)."
            status = "PASS" if ratio >= self.min_powered_evidence_ratio else "WARNING"
            return AuditCheckResult(check_name="statistical_rigor_of_evidence", status=status, message=message)
        except Neo4jError as e:
            logger.error(f"Neo4j error in statistical rigor check: {e}")
            return AuditCheckResult(check_name="statistical_rigor_of_evidence", status="FAIL", message=f"Query error: {e}")

    async def _check_causal_claim_validity(self) -> AuditCheckResult:
        return AuditCheckResult(check_name="causal_claim_validity", status="NOT_RUN", message="Causal claim validity check (Neo4j) not fully implemented.")
    async def _check_temporal_consistency(self) -> AuditCheckResult:
        return AuditCheckResult(check_name="temporal_consistency", status="NOT_RUN", message="Temporal consistency check (Neo4j) not fully implemented.")
    async def _check_collaboration_attributions(self) -> AuditCheckResult:
        return AuditCheckResult(check_name="collaboration_attributions_check", status="NOT_RUN", message="Attribution check (Neo4j) not fully implemented.")

    async def _calculate_final_confidence(self, audit_results: List[AuditCheckResult]) -> ConfidenceVector:
        final_conf = ConfidenceVector(empirical_support=0.5, theoretical_basis=0.5, methodological_rigor=0.5, consensus_alignment=0.5)
        falsifiability_check = next((r for r in audit_results if r.check_name == "hypothesis_falsifiability"), None)
        bias_check = next((r for r in audit_results if r.check_name == "bias_flags_assessment"), None)
        stat_rigor_check = next((r for r in audit_results if r.check_name == "statistical_rigor_of_evidence"),None)

        if falsifiability_check:
            if falsifiability_check.status == "PASS": final_conf.methodological_rigor += 0.2
            elif falsifiability_check.status == "WARNING": final_conf.methodological_rigor += 0.05
            elif falsifiability_check.status == "FAIL": final_conf.methodological_rigor -= 0.2
        if bias_check:
            if bias_check.status == "PASS": final_conf.methodological_rigor += 0.1
            elif bias_check.status == "FAIL": final_conf.methodological_rigor -= 0.15
        if stat_rigor_check:
            if stat_rigor_check.status == "PASS": final_conf.empirical_support += 0.2
            elif stat_rigor_check.status == "WARNING": final_conf.empirical_support -= 0.1
        
        final_conf.empirical_support = max(0.0, min(1.0, final_conf.empirical_support))
        final_conf.theoretical_basis = max(0.0, min(1.0, final_conf.theoretical_basis))
        final_conf.methodological_rigor = max(0.0, min(1.0, final_conf.methodological_rigor))
        final_conf.consensus_alignment = max(0.0, min(1.0, final_conf.consensus_alignment))
        logger.info(f"Calculated final confidence vector: {final_conf.model_dump()}")
        return final_conf

    async def execute(
        self, current_session_data: GoTProcessorSessionData # graph: ASRGoTGraph removed
    ) -> StageOutput:
        self._log_start(current_session_data.session_id)
        composition_stage_output = current_session_data.accumulated_context.get(CompositionStage.stage_name, {})
        composed_output_dict = composition_stage_output.get("final_composed_output")
        composed_output_obj: Optional[ComposedOutput] = None
        if composed_output_dict:
            try: composed_output_obj = ComposedOutput(**composed_output_dict)
            except (ValidationError, TypeError) as e: logger.warning(f"Could not parse ComposedOutput for reflection: {e}")
        
        audit_results: List[AuditCheckResult] = []
        audit_checks_to_run = {
            "high_confidence_impact_coverage": self._check_high_confidence_impact_coverage_from_neo4j,
            "bias_flags_assessment": self._check_bias_flags_assessment_from_neo4j,
            "knowledge_gaps_addressed": lambda: self._check_knowledge_gaps_addressed_from_neo4j(composed_output_obj), # Pass data
            "hypothesis_falsifiability": self._check_hypothesis_falsifiability_from_neo4j,
            "statistical_rigor_of_evidence": self._check_statistical_rigor_from_neo4j,
            "causal_claim_validity": self._check_causal_claim_validity,
            "temporal_consistency": self._check_temporal_consistency,
            "collaboration_attributions_check": self._check_collaboration_attributions,
        }

        for check_name, check_func in audit_checks_to_run.items():
            try:
                 # Check if it's one of the adapted functions needing specific args
                if check_name == "knowledge_gaps_addressed":
                     audit_results.append(await check_func()) # Already a lambda with arg
                else:
                    audit_results.append(await check_func())
            except Exception as e:
                logger.error(f"Error in audit check '{check_name}': {e}")
                audit_results.append(AuditCheckResult(check_name=check_name, status="ERROR", message=str(e)))

        active_audit_results = [r for r in audit_results if r.status != "NOT_RUN"]
        final_confidence_vector = await self._calculate_final_confidence(active_audit_results)

        summary = (f"Reflection stage complete. Performed {len(active_audit_results)} active audit checks. "
                   f"Final overall confidence assessed. "
                   f"PASS: {sum(1 for r in active_audit_results if r.status == 'PASS')}, "
                   f"WARNING: {sum(1 for r in active_audit_results if r.status == 'WARNING')}, "
                   f"FAIL: {sum(1 for r in active_audit_results if r.status == 'FAIL')}.")
        metrics = {
            "audit_checks_performed_count": len(active_audit_results),
            "audit_pass_count": sum(1 for r in active_audit_results if r.status == "PASS"),
            "audit_warning_count": sum(1 for r in active_audit_results if r.status == "WARNING"),
            "audit_fail_count": sum(1 for r in active_audit_results if r.status == "FAIL"),
            "final_confidence_empirical": final_confidence_vector.empirical_support,
            "final_confidence_theoretical": final_confidence_vector.theoretical_basis,
            "final_confidence_methodological": final_confidence_vector.methodological_rigor,
            "final_confidence_consensus": final_confidence_vector.consensus_alignment,
        }
        context_update = {
            "final_confidence_vector_from_reflection": final_confidence_vector.to_list(),
            "audit_check_results": [res.model_dump() for res in audit_results],
        }
        return StageOutput(summary=summary, metrics=metrics, next_stage_context_update={self.stage_name: context_update})
