from __future__ import annotations

import os

from loguru import logger

from ..config import env_settings

LLM_QUERY_LOGS: list[dict[str, str]] = []


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to a configured large language model (LLM) provider and return the generated response.
    
    Depending on environment settings, queries either the Claude or OpenAI API using the specified model and API key. The function logs the last five prompt-response pairs for reference. If an error occurs during the LLM call, returns an error message string.
    
    Parameters:
        prompt (str): The user prompt to send to the LLM.
    
    Returns:
        str: The LLM's response text, or an error message if the call fails.
    """
    provider = env_settings.llm_provider.lower()
    try:
        if provider == "claude":
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=env_settings.anthropic_api_key)
            resp = client.messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229"),
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.content[0].text
        else:
            import openai  # type: ignore
            client = openai.OpenAI(api_key=env_settings.openai_api_key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.choices[0].message.content.strip()
        LLM_QUERY_LOGS.append({"prompt": prompt, "response": result})
        if len(LLM_QUERY_LOGS) > 5:
            LLM_QUERY_LOGS.pop(0)
        return result
    except Exception as e:  # pragma: no cover
        logger.error(f"LLM call failed: {e}")
        return f"LLM error: {e}"
