# Developing Custom ASR-GoT Stages

## Introduction

NexusMind's ASR-GoT (Automated Scientific Reasoning - Graph of Thoughts) pipeline is designed for modularity, allowing developers to extend its capabilities by creating custom processing stages. Each stage in the pipeline performs a specific task, transforming or analyzing the data within the `GoTProcessorSessionData` and contributing to the overall reasoning process.

This guide provides a basic outline for developing your own custom stages.

## Stage Structure

A custom stage should inherit from the `BaseStage` class found in `src.asr_got_reimagined.domain.stages.base_stage`.

Key components of a stage:

1.  **`stage_name` (Static Class Attribute):**
    *   A unique string identifier for the stage. This name is used as a key in the `accumulated_context` within `GoTProcessorSessionData` to store and retrieve results specific to this stage.
    *   Example: `stage_name = "MyCustomAnalysisStage"`

2.  **`__init__(self, settings)`:**
    *   The constructor can be used to accept the global application `settings` (an instance of `AppSettings` from `src.asr_got_reimagined.config`) if your stage needs access to configuration parameters.

3.  **`async def execute(self, current_session_data: GoTProcessorSessionData) -> StageOutput:`:**
    *   This is the main method where your stage's logic resides.
    *   It takes `current_session_data` as input, allowing access to the query, existing graph state (if applicable, though direct graph manipulation is now stage-local within Neo4j), and outputs from previous stages stored in `current_session_data.accumulated_context`.
    *   It must return an instance of `StageOutput` (from `src.asr_got_reimagined.domain.stages.base_stage`).

## `StageOutput` Class

The `StageOutput` class is used to return results from your stage:

*   `stage_name` (str): The name of the stage producing this output (should match your stage's `stage_name`).
*   `status` (str, Optional): A simple status string (e.g., "success", "failure", "partial_success").
*   `summary` (str, Optional): A brief human-readable summary of what the stage accomplished.
*   `error_message` (str, Optional): If an error occurred, a message describing the error.
*   `next_stage_context_update` (Dict\[str, Any], Optional): A dictionary containing data that should be added to or updated in the `GoTProcessorSessionData.accumulated_context`.
    *   **Best Practice:** Store your stage's primary results under a key that matches your stage's `stage_name` to avoid conflicts with other stages.
        Example: `{"MyCustomAnalysisStage": {"analysis_result": "...", "score": 0.8}}`
*   `metrics` (Dict\[str, Any], Optional): Any metrics collected during the stage's execution (e.g., processing time for sub-tasks, number of items processed).

## Example Custom Stage (Conceptual)

```python
# src/asr_got_reimagined/domain/stages/my_custom_stage.py (Example Path)

from asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput
from asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from src.asr_got_reimagined.config import Settings # Assuming settings are passed
from loguru import logger
class MyCustomAnalysisStage(BaseStage):
    stage_name = "MyCustomAnalysisStage"

    def __init__(self, settings: Settings): # Pass global settings if needed
        super().__init__(settings)
        # You can access self.settings here if needed for configuration
        # e.g., self.my_custom_param = settings.asr_got.default_parameters.some_param

    async def execute(
        self,
        current_session_data: GoTProcessorSessionData,
    ) -> StageOutput:
        logger.info(f"[{self.stage_name}] Starting execution for session: {current_session_data.session_id}")
        output = StageOutput(stage_name=self.stage_name)

        try:
            # 1. Get necessary input from current_session_data.accumulated_context
            # For example, data from a previous stage:
            # prev_stage_output = current_session_data.accumulated_context.get("PreviousStageName", {})
            # input_data = prev_stage_output.get("some_key")

            # 2. Perform your custom logic
            # (e.g., interact with Neo4j via neo4j_utils, call external APIs, perform complex calculations)
            analysis_result = f"Analysis of '{current_session_data.query[:20]}...' completed."
            custom_score = 0.75

            # 3. Prepare results for the next stage or final output
            output.next_stage_context_update = {
                self.stage_name: {
                    "processed_data": analysis_result,
                    "custom_score": custom_score,
                    "status": "success"
                }
            }
            output.summary = f"Successfully analyzed data. Score: {custom_score}"
            output.status = "success"
            logger.info(f"[{self.stage_name}] Execution successful.")

        except Exception as e:
            error_msg = f"Error during {self.stage_name}: {str(e)}"
            logger.error(error_msg)
            output.error_message = error_msg
            output.status = "failure"
            output.next_stage_context_update = {
                 self.stage_name: {
                    "status": "failure",
                    "error_details": error_msg
                }
            }
            
        return output
```

## Integrating Your Custom Stage

1.  **Place your stage file** in a suitable location (e.g., within `src/asr_got_reimagined/domain/stages/` or a new subdirectory for custom/plugin stages).
2.  **Configure the pipeline** in `config/settings.yaml` to include your stage:
    ```yaml
    # config/settings.yaml (snippet)
    asr_got:
      # ...
      pipeline_stages:
        # ... other stages ...
        - name: "My Custom Analysis" # Friendly name for your stage
          module_path: "src.asr_got_reimagined.domain.stages.my_custom_stage.MyCustomAnalysisStage" # Adjust path as needed
          enabled: true
        # ... other stages ...
    ```
3.  Ensure your stage's `module_path` is correct and the class name matches.

## Best Practices

*   **Idempotency:** If possible, design stages to be idempotent, meaning running them multiple times with the same input produces the same result.
*   **Error Handling:** Implement robust error handling within your stage and populate `StageOutput.error_message` appropriately.
*   **Configuration:** If your stage requires specific parameters, consider adding them to the `ASRGoTDefaultParams` model (in `config.py`) or a dedicated configuration model, and access them via `self.settings`.
*   **Logging:** Use `loguru` for detailed logging within your stage to aid in debugging and monitoring.
*   **Testing:** Write unit tests for your custom stage to ensure its logic is correct and it handles various inputs and edge cases properly.

This placeholder guide should help you get started. Refer to the existing stages in `src/asr_got_reimagined/domain/stages/` for more detailed examples.
