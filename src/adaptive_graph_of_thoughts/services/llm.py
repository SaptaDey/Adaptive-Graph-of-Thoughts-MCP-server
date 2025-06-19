from __future__ import annotations

import os

from loguru import logger

from ..config import env_settings

LLM_QUERY_LOGS: list[dict[str, str]] = []


def ask_llm(prompt: str) -> str:
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
            openai.api_key = env_settings.openai_api_key
            resp = openai.ChatCompletion.create(
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
