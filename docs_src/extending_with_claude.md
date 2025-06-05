# Extending NexusMind with Direct Claude API Calls

## Introduction

NexusMind's ASR-GoT (Automated Scientific Reasoning - Graph of Thoughts) pipeline is designed to be modular. While many reasoning tasks can be handled by its core stages, there might be scenarios where directly leveraging a powerful Large Language Model (LLM) like Anthropic's Claude can be beneficial for specific, complex sub-tasks or to use Claude as a specialized tool within the pipeline.

This guide outlines how to configure NexusMind for direct Claude API calls and provides conceptual examples for integrating these calls within custom service functions or ASR-GoT stages.

## Configuration

To enable direct Claude API calls, you need to configure the `claude_api` section in your `config/settings.yaml` file.

```yaml
# config/settings.yaml (snippet)

# ... other settings ...

# Optional: Direct Claude API integration settings (if the app needs to call Claude API itself)
claude_api:
  api_key: "env_var:CLAUDE_API_KEY" # Recommended: Load from environment variable CLAUDE_API_KEY
  # api_key: "your_actual_claude_api_key_here" # Alternatively, direct input (less secure)
  default_model: "claude-3-opus-20240229" # Or any other Claude model you intend to use
  timeout_seconds: 120 # Timeout for API requests in seconds
  max_retries: 2       # Maximum number of retries for API requests
```

**Key Configuration Fields:**

*   `api_key` (Optional\[str]):
    *   Your API key for accessing the Claude API.
    *   **Security Best Practice:** It is strongly recommended to provide the API key via an environment variable named `CLAUDE_API_KEY`. The application is configured to recognize the `"env_var:CLAUDE_API_KEY"` string and substitute it with the actual value from the environment variable at runtime.
    *   Alternatively, for development or testing, you can directly paste your API key, but avoid this for production deployments.
    *   If not provided, any direct Claude API calls will fail.
*   `default_model` (str):
    *   The default Claude model to use for API calls (e.g., `"claude-3-opus-20240229"`, `"claude-3-sonnet-20240229"`).
    *   Default: `"claude-3-opus-20240229"`
*   `timeout_seconds` (int):
    *   The timeout duration in seconds for waiting for a response from the Claude API.
    *   Default: `120`
*   `max_retries` (int):
    *   The maximum number of times the application will attempt to retry a failed API request.
    *   Default: `2`

If the `claude_api` section is omitted or commented out, NexusMind will not be able to make direct calls to the Claude API. The settings loader will treat `settings.claude_api` as `None`.

## Service Function Implementation (Conceptual)

You can create service functions or utility methods that encapsulate the logic for calling the Claude API. These functions can then be used by various parts of the application, such as custom ASR-GoT stages.

Here's a conceptual example using `httpx` for asynchronous requests:

```python
# Hypothetical service function (e.g., in src/asr_got_reimagined/services/claude_service.py)
import httpx
from typing import Optional, Dict, Any
from loguru import logger

# Assuming 'settings' is the global AppSettings instance from src.asr_got_reimagined.config
from src.asr_got_reimagined.config import settings

CLAUDE_API_BASE_URL = "https://api.anthropic.com/v1" # Example base URL

async def call_claude_api(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_override: Optional[str] = None,
    max_tokens: int = 1024,
) -> Optional[Dict[str, Any]]:
    """
    Makes a conceptual call to the Claude API (e.g., messages endpoint).
    """
    if not settings.claude_api or not settings.claude_api.api_key:
        logger.error("Claude API settings or API key is not configured.")
        return None

    api_key = settings.claude_api.api_key
    # Resolve environment variable if specified (Pydantic settings usually handle this,
    # but direct usage might need explicit resolution if not using Pydantic model directly for key)
    if api_key.startswith("env_var:"):
        env_var_name = api_key.split(":", 1)[1]
        import os
        api_key = os.getenv(env_var_name)
        if not api_key:
            logger.error(f"Environment variable {env_var_name} for Claude API key not set.")
            return None
            
    model_to_use = model_override or settings.claude_api.default_model
    timeout = settings.claude_api.timeout_seconds
    # Max retries would typically be handled by the HTTP client's retry mechanism or a wrapper.

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01", # Required header
        "content-type": "application/json"
    }

    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": model_to_use,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system_prompt:
        payload["system"] = system_prompt


    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Note: For production, consider more robust retry logic (e.g., with tenacity)
            # For simplicity, only one attempt is shown here. Max_retries from config is not used directly.
            response = await client.post(f"{CLAUDE_API_BASE_URL}/messages", headers=headers, json=payload)
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
            
            response_data = response.json()
            logger.debug(f"Claude API response: {response_data}")
            return response_data

    except httpx.HTTPStatusError as e:
        logger.error(f"Claude API request failed with status {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Claude API request failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while calling Claude API: {e}")
        
    return None

# Example Usage (conceptual):
# async def main():
#     response = await call_claude_api("Explain quantum entanglement in simple terms.")
#     if response and response.get("content"):
#         print(response.get("content")[0].get("text"))
# main()
```

**Key points for the service function:**
*   Reads configuration from `settings.claude_api`.
*   Handles API key resolution (especially if using the `env_var:` prefix manually, though Pydantic settings often handle this transparently when the `ClaudeAPIConfig` model is instantiated).
*   Uses an HTTP client (`httpx` in this example) to make the API call.
*   Sets necessary headers, including `x-api-key` and `anthropic-version`.
*   Sends a JSON payload appropriate for the target Claude API endpoint (e.g., `/v1/messages`).
*   Includes basic error handling for network issues or API errors.

## Custom Stage Example (Conceptual)

A custom ASR-GoT stage can leverage the service function described above to incorporate Claude's capabilities into the reasoning pipeline.

```python
# Hypothetical custom stage (e.g., in src/asr_got_reimagined/domain/stages/custom_claude_stage.py)
from src.asr_got_reimagined.domain.stages.base_stage import BaseStage, StageOutput
from src.asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
from loguru import logger

# Assuming the service function is available, e.g.:
# from src.asr_got_reimagined.services.claude_service import call_claude_api

class ClaudeQueryStage(BaseStage):
    stage_name = "ClaudeQueryStage" # Unique name for this stage type

    async def execute(
        self,
        current_session_data: GoTProcessorSessionData,
    ) -> StageOutput:
        """
        Example stage that uses Claude to answer a query from the session data.
        """
        output = StageOutput(stage_name=self.stage_name)
        
        # Get some input from the accumulated context
        # For example, a specific query or data piece generated by a previous stage
        query_for_claude = current_session_data.accumulated_context.get("query_for_claude", current_session_data.query)
        
        if not query_for_claude:
            output.error_message = "No query found for Claude in session data."
            logger.warning(output.error_message)
            return output

        logger.info(f"[{self.stage_name}] Calling Claude API for: {query_for_claude[:100]}...")
        
        # Conceptual: In a real implementation, call_claude_api would be imported
        # from its actual location (e.g., a services module).
        # For this example, we'll mock its existence.
        # claude_response_data = await call_claude_api(prompt=query_for_claude)
        
        # Mocked response for illustration:
        claude_response_data = { "content": [{"type": "text", "text": f"Mocked Claude response to: {query_for_claude}"}] }

        if claude_response_data and claude_response_data.get("content"):
            claude_text_response = claude_response_data.get("content")[0].get("text")
            logger.info(f"[{self.stage_name}] Received response from Claude.")
            
            # Add Claude's response to the accumulated context for subsequent stages
            output.next_stage_context_update = {
                self.stage_name: { # Key results under this stage's name
                    "claude_response_text": claude_text_response,
                    "status": "success"
                }
            }
            output.summary = f"Successfully queried Claude. Response: {claude_text_response[:100]}..."
        else:
            output.error_message = "Failed to get a valid response from Claude API."
            logger.error(f"[{self.stage_name}] {output.error_message}")
            output.next_stage_context_update = {
                 self.stage_name: {
                    "status": "failure",
                    "error_message": output.error_message
                }
            }
            
        return output
```

**Key points for the custom stage:**
*   It inherits from `BaseStage`.
*   It uses the `current_session_data` to get inputs.
*   It calls the `call_claude_api` service function.
*   It processes the response and updates `current_session_data.accumulated_context` via `StageOutput.next_stage_context_update` for subsequent stages.
*   It provides a summary of its operation.

To use this custom stage, you would add its configuration to the `pipeline_stages` list in `config/settings.yaml`:
```yaml
# config/settings.yaml (snippet)
asr_got:
  # ... other asr_got settings ...
  pipeline_stages:
    # ... other stages ...
    - name: "Custom Claude Query"
      module_path: "src.asr_got_reimagined.domain.stages.custom_claude_stage.ClaudeQueryStage"
      enabled: true
    # ... other stages ...
```

## Disclaimer

Integrating direct LLM calls like those to the Claude API is an advanced extension point. Developers should carefully consider:

*   **API Costs:** Frequent or large-scale calls to Claude can incur significant costs. Monitor usage and optimize prompts/requests.
*   **Rate Limits:** Be aware of Claude API rate limits and implement appropriate backoff and retry strategies (potentially more robust than the basic example).
*   **Latency:** Direct API calls, especially synchronous ones, can add latency to the ASR-GoT pipeline. Consider asynchronous patterns or background tasks for non-critical API calls if they might block pipeline progression.
*   **Error Handling:** Implement comprehensive error handling for API unavailability, timeouts, invalid responses, etc.
*   **Data Privacy and Security:** Ensure that data sent to external APIs complies with privacy policies and security requirements.

By understanding these considerations, developers can effectively extend NexusMind's capabilities using the Claude API.
```
