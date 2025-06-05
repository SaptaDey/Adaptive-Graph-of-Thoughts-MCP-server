import time
import uuid
import importlib
from typing import Any, Optional, Dict

from loguru import logger

from asr_got_reimagined.domain.models.common_types import (
    ComposedOutput,
    GoTProcessorSessionData,
)
from asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput

# Import the import_stages function for lazy loading
from asr_got_reimagined.domain.stages import import_stages

class GoTProcessor:
    def __init__(self, settings):
        """
        Initializes a GoTProcessor instance with the provided settings.
        """
        self.settings = settings
        logger.info("Initializing GoTProcessor")
        self.stages = self._initialize_stages()
        logger.info(f"GoTProcessor initialized with {len(self.stages)} configured and enabled stages.")

    def _initialize_stages(self) -> list[BaseStage]:
        """
        Instantiates and returns the ordered list of processing stages based on the configuration.
        
        Returns:
            A list of initialized stage objects.
        Raises:
            RuntimeError: If a configured stage module/class cannot be loaded.
        """
        initialized_stages: list[BaseStage] = []
        if not hasattr(self.settings.asr_got, 'pipeline_stages') or not self.settings.asr_got.pipeline_stages:
            logger.warning("Pipeline stages not defined or empty in settings.asr_got.pipeline_stages. Processor will have no stages.")
            return initialized_stages

        for stage_config in self.settings.asr_got.pipeline_stages:
            if stage_config.enabled:
                try:
                    module_name, class_name = stage_config.module_path.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    stage_cls = getattr(module, class_name)
                    
                    # Check if the loaded class is a subclass of BaseStage
                    if not issubclass(stage_cls, BaseStage):
                        logger.error(f"Configured stage class {stage_config.module_path} for stage '{stage_config.name}' is not a subclass of BaseStage. Skipping.")
                        continue # Or raise error

                    initialized_stages.append(stage_cls(self.settings))
                    logger.info(f"Successfully loaded and initialized stage: '{stage_config.name}' from {stage_config.module_path}")
                except ImportError as e:
                    logger.error(f"Error importing module for stage '{stage_config.name}' from path '{stage_config.module_path}': {e}")
                    # For critical stages like Initialization, this should be a fatal error.
                    # For this example, we'll make any load failure fatal.
                    raise RuntimeError(f"Failed to load module for stage: {stage_config.name} ({stage_config.module_path})") from e
                except AttributeError as e:
                    logger.error(f"Error getting class '{class_name}' from module '{module_name}' for stage '{stage_config.name}': {e}")
                    raise RuntimeError(f"Failed to load class for stage: {stage_config.name} ({class_name} from {module_name})") from e
                except Exception as e:
                    logger.error(f"An unexpected error occurred while loading stage '{stage_config.name}': {e}")
                    raise RuntimeError(f"Unexpected error loading stage: {stage_config.name}") from e
            else:
                logger.info(f"Stage '{stage_config.name}' is disabled and will not be loaded.")
                
        return initialized_stages

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        operational_params: Optional[dict[str, Any]] = None,
        initial_context: Optional[dict[str, Any]] = None,
    ) -> GoTProcessorSessionData:
        """
        Processes a natural language query through the ASR-GoT pipeline, executing each stage in sequence and managing session state, context, and error handling.
        
        Args:
            query: The natural language query to process.
            session_id: Optional session identifier for continuing or managing a session.
            operational_params: Optional parameters to control processing behavior.
            initial_context: Optional initial context to seed the processing.
        
        Returns:
            GoTProcessorSessionData containing the final answer, confidence vector, accumulated context, graph state, and a trace of stage outputs.
        
        This method initializes or continues a session, orchestrates the execution of all processing stages, logs detailed input and output information for each stage, handles errors (especially during initialization), and compiles the final results and metrics for the query.
        """
        # Load stage classes lazily to avoid circular imports
        _stages = import_stages()

        start_total_time = time.time()
        logger.info(f"Starting NexusMind query processing for: '{query[:100]}...'")

        # Initialize or retrieve session data
        current_session_data = GoTProcessorSessionData(
            session_id=session_id or f"session-{uuid.uuid4()}", query=query
        )

        # ASRGoTGraph instantiation removed.
        # The graph_state attribute in GoTProcessorSessionData will also be removed.
        # If any metadata was stored in graph_state.graph_metadata, 
        # it needs a new home if still required (e.g., directly in accumulated_context).
        # For this task, assuming such metadata is either not critical or handled by stages.
        # current_session_data.accumulated_context["graph_metadata"] = {
        #     "query": query,
        #     "session_id": current_session_data.session_id
        # } # Example if we wanted to keep this info

        # Process initial context
        if initial_context:
            current_session_data.accumulated_context["initial_context"] = (
                initial_context
            )

        # Process operational parameters
        op_params = operational_params or {}
        current_session_data.accumulated_context["operational_params"] = op_params

        if not self.stages:
            logger.error("No stages initialized for GoTProcessor. Cannot process query.")
            current_session_data.final_answer = "Error: Query processor is not configured with any processing stages."
            current_session_data.final_confidence_vector = [0.0, 0.0, 0.0, 0.0]
            return current_session_data
            
        logger.info(f"Executing {len(self.stages)} configured processing stages.")

        for i, stage_instance in enumerate(self.stages):
            stage_start_time = time.time()
            
            stage_module_path = f"{stage_instance.__class__.__module__}.{stage_instance.__class__.__name__}"
            stage_config_item = next((s_conf for s_conf in self.settings.asr_got.pipeline_stages if s_conf.module_path == stage_module_path), None)
            
            stage_name_for_log = stage_config_item.name if stage_config_item else stage_instance.__class__.__name__
            # current_stage_context_key is the stage's defined static 'stage_name' (e.g., InitializationStage.stage_name)
            # This is used for consistent context keying and identifying stage types for logic.
            current_stage_context_key = stage_instance.stage_name 

            logger.info(f"Executing stage {i + 1}/{len(self.stages)}: {stage_name_for_log} (Context Key: {current_stage_context_key})")

            logger.debug(f"--- Preparing for Stage: {stage_name_for_log} ---")
            if current_stage_context_key == _stages["InitializationStage"].stage_name:
                 logger.debug(f"Input for {stage_name_for_log}: Query='{query[:100]}...', InitialContextKeys={list(initial_context.keys()) if initial_context else []}, OpParamsKeys={list(op_params.keys())}")
            else:
                 context_keys = list(current_session_data.accumulated_context.keys())
                 logger.debug(f"Accumulated context keys before {stage_name_for_log}: {context_keys}")
            logger.debug(f"--- End Preparing for Stage: {stage_name_for_log} ---")

            try:
                stage_result = await stage_instance.execute(current_session_data=current_session_data)

                logger.debug(f"--- Output from Stage: {stage_name_for_log} ---")
                if isinstance(stage_result, StageOutput):
                    if stage_result.error_message: 
                        logger.error(f"Stage {stage_name_for_log} reported an error: {stage_result.error_message}")
                    if hasattr(stage_result, "next_stage_context_update") and stage_result.next_stage_context_update:
                        logger.debug(f"Raw output (next_stage_context_update): {stage_result.next_stage_context_update}")
                    else:
                        logger.debug(f"Stage {stage_name_for_log} produced StageOutput but 'next_stage_context_update' is missing or empty.")
                    if hasattr(stage_result, "summary") and stage_result.summary:
                        logger.debug(f"Summary: {stage_result.summary}")
                    if hasattr(stage_result, "metrics") and stage_result.metrics:
                        logger.debug(f"Metrics: {stage_result.metrics}")
                elif stage_result is not None: # Stage might return non-StageOutput (though discouraged)
                    logger.debug(f"Raw output (non-StageOutput): {stage_result}")
                else: # Stage returned None
                    logger.debug(f"Stage {stage_name_for_log} execution returned None.")
                logger.debug(f"--- End Output from Stage: {stage_name_for_log} ---")

                if stage_result and hasattr(stage_result, "next_stage_context_update") and stage_result.next_stage_context_update:
                    current_session_data.accumulated_context.update(stage_result.next_stage_context_update)
                elif stage_result: 
                    logger.warning(f"Stage {stage_name_for_log} produced a result but it was missing 'next_stage_context_update' or it was empty. No context updated by this stage directly.")

                stage_duration_ms = int((time.time() - stage_start_time) * 1000)
                trace_summary = f"Completed {stage_name_for_log}"
                if isinstance(stage_result, StageOutput) and stage_result.summary:
                    trace_summary = stage_result.summary 

                trace_entry = {
                    "stage_number": i + 1, "stage_name": stage_name_for_log,
                    "duration_ms": stage_duration_ms, "summary": trace_summary,
                }
                if isinstance(stage_result, StageOutput) and stage_result.error_message:
                    trace_entry["error"] = stage_result.error_message
                    if stage_result.error_message not in trace_summary: 
                        trace_entry["summary"] = f"{trace_summary} (Reported Error: {stage_result.error_message})"
                
                current_session_data.stage_outputs_trace.append(trace_entry)
                logger.info(f"Completed stage {i + 1}: {stage_name_for_log} in {stage_duration_ms}ms")

                # --- Halting Logic Helper ---
                def _update_trace_for_halt(halt_log_message: str, halt_reason_summary: str):
                    last_trace_entry = current_session_data.stage_outputs_trace[-1]
                    if "error" not in last_trace_entry: # Add error info if not already there from stage_result
                        last_trace_entry["error"] = halt_log_message
                        last_trace_entry["summary"] = halt_reason_summary 
                    elif halt_log_message not in last_trace_entry["error"]: # Append if different
                        last_trace_entry["error"] += f"; {halt_log_message}"

                def _halt_processing(reason_summary: str, log_message: str):
                    logger.error(log_message)
                    current_session_data.final_answer = reason_summary
                    current_session_data.final_confidence_vector = [0.0, 0.0, 0.0, 0.0]
                    _update_trace_for_halt(log_message, reason_summary)

                # Define stage names once to avoid repeating lookups
                initialization_stage_name = _stages["InitializationStage"].stage_name
                decomposition_stage_name = _stages["DecompositionStage"].stage_name
                hypothesis_stage_name = _stages["HypothesisStage"].stage_name
                evidence_stage_name = _stages["EvidenceStage"].stage_name
                subgraph_extraction_stage_name = _stages["SubgraphExtractionStage"].stage_name
                
                # --- Stage-Specific Halting Checks (using current_stage_context_key) ---
                if current_stage_context_key == _stages["InitializationStage"].stage_name:
                    init_context_data = current_session_data.accumulated_context.get(initialization_stage_name, {})
                    error_summary = None
                    if isinstance(stage_result, StageOutput) and stage_result.error_message:
                        error_summary = stage_result.error_message
                    elif init_context_data.get("error"):
                        error_summary = str(init_context_data.get("error"))
                    elif not init_context_data.get("root_node_id"):
                        error_summary = f"{stage_name_for_log} did not provide root_node_id."
                    
                    if error_summary:
                        halt_reason = f"Processing halted: {stage_name_for_log} failed: {error_summary}"
                        _halt_processing(halt_reason, f"Halting due to critical error in {stage_name_for_log}: {error_summary}")
                        break 

                elif current_stage_context_key == decomposition_stage_name:
                    decomp_context_data = current_session_data.accumulated_context.get(decomposition_stage_name, {})
                    if not decomp_context_data.get("decomposition_results", []) and not current_session_data.final_answer:
                        _halt_processing("Processing halted: The query could not be broken down into actionable components.",
                                         f"Halting: No components after {stage_name_for_log}.")
                        break
                
                elif current_stage_context_key == hypothesis_stage_name:
                    hypo_context_data = current_session_data.accumulated_context.get(hypothesis_stage_name, {})
                    if not hypo_context_data.get("hypotheses_results", []) and not current_session_data.final_answer:
                        _halt_processing("Processing halted: No hypotheses could be generated.",
                                         f"Halting: No hypotheses generated after {stage_name_for_log}.")
                        break

                elif current_stage_context_key == evidence_stage_name:
                    evidence_context_data = current_session_data.accumulated_context.get(evidence_stage_name, {})
                    evidence_integration_summary = evidence_context_data.get("evidence_integration_summary", {})
                    if evidence_integration_summary.get("total_evidence_integrated", -1) == 0:
                        logger.warning(f"No evidence was integrated by {stage_name_for_log}. Proceeding with caution.")
                        current_session_data.accumulated_context["no_evidence_found"] = True
                
                elif current_stage_context_key == subgraph_extraction_stage_name:
                    subgraph_context_data = current_session_data.accumulated_context.get(subgraph_extraction_stage_name, {})
                    subgraph_details = subgraph_context_data.get("subgraph_extraction_details", {})
                    if subgraph_details.get("nodes_extracted", -1) == 0:
                        logger.warning(f"No subgraph was extracted by {stage_name_for_log}. Proceeding with caution.")
                        current_session_data.accumulated_context["no_subgraph_extracted"] = True
            
            except Exception as e: 
                logger.exception(f"Unhandled critical error during execution of stage {stage_name_for_log}: {e!s}")
                halt_msg = f"A critical unhandled error occurred during the '{stage_name_for_log}' stage. Processing cannot continue."
                current_session_data.final_answer = halt_msg
                current_session_data.final_confidence_vector = [0.0,0.0,0.0,0.0]
                
                critical_error_trace = {
                    "stage_number": i + 1, "stage_name": stage_name_for_log,
                    "error": f"Unhandled Critical Exception: {str(e)}", "summary": halt_msg,
                    "duration_ms": int((time.time() - stage_start_time) * 1000)
                }
                # Update or append trace for this critical error
                if current_session_data.stage_outputs_trace and \
                   current_session_data.stage_outputs_trace[-1]["stage_name"] == stage_name_for_log and \
                   current_session_data.stage_outputs_trace[-1]["stage_number"] == i + 1:
                    current_session_data.stage_outputs_trace[-1].update(critical_error_trace)
                else:
                    current_session_data.stage_outputs_trace.append(critical_error_trace)
                break 

        # Get remaining stage names for final processing
        composition_stage_name = _stages["CompositionStage"].stage_name
        reflection_stage_name = _stages["ReflectionStage"].stage_name
        
        # --- Final Answer and Confidence Extraction ---
        if not current_session_data.final_answer: 
            composition_context_key = composition_stage_name
            composition_stage_data = current_session_data.accumulated_context.get(composition_context_key, {})
            final_composed_output_dict = composition_stage_data.get("final_composed_output")

            if final_composed_output_dict and isinstance(final_composed_output_dict, dict):
                try:
                    final_output_obj = ComposedOutput(**final_composed_output_dict)
                    current_session_data.final_answer = f"{final_output_obj.executive_summary}\n\n(Full report details generated)"
                except Exception as e:
                    logger.error(f"Could not parse final_composed_output from {composition_stage_name}: {e}")
                    current_session_data.final_answer = "Error during final composition of answer."
            else:
                logger.warning(f"{composition_stage_name} did not produce a final_composed_output structure or it was invalid. Context data: {composition_stage_data}")
                current_session_data.final_answer = f"{composition_stage_name} did not produce a valid final output structure."

        reflection_context_key = reflection_stage_name
        reflection_stage_data = current_session_data.accumulated_context.get(reflection_context_key, {})
        final_confidence = reflection_stage_data.get("final_confidence_vector_from_reflection", [0.1,0.1,0.1,0.1])
        
        if "Processing halted" in (current_session_data.final_answer or "") or \
           "Error:" in (current_session_data.final_answer or ""):
            current_session_data.final_confidence_vector = [0.0, 0.0, 0.0, 0.0]
        else:
            current_session_data.final_confidence_vector = final_confidence

        total_execution_time_ms = int((time.time() - start_total_time) * 1000)
        current_session_data.execution_time_ms = total_execution_time_ms
        logger.info(
            f"NexusMind query processing completed for session {current_session_data.session_id} in {total_execution_time_ms}ms."
        )
        return current_session_data

    async def shutdown_resources(self):
        logger.info("Shutting down GoTProcessor resources")
        return
