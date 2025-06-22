from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from ..config import env_settings


@dataclass
class LLMConfig:
    """Basic configuration for an LLM provider."""

    provider: str
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    base_url: Optional[str] = None
    additional_headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMProvider:
    """Minimal provider interface used in tests."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, Any]:  # pragma: no cover - runtime only
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """Stub OpenAI provider used for unit tests."""

    def complete(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, Any]:  # pragma: no cover - runtime only
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": len(messages)},
        }


class AnthropicProvider(LLMProvider):
    """Stub Anthropic provider used for unit tests."""

    def complete(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, Any]:  # pragma: no cover - runtime only
        return {"content": [{"text": "ok"}], "usage": {"input_tokens": len(messages)}}


class LLMException(Exception):
    pass


class TokenLimitExceeded(LLMException):
    pass


class RateLimitExceeded(LLMException):
    pass


class LLMService:
    """Simplified LLM service supporting basic caching for unit tests."""

    def __init__(self, config: LLMConfig) -> None:
        if config.provider.lower() == "openai":
            self.provider = OpenAIProvider(config)
        elif config.provider.lower() == "anthropic":
            self.provider = AnthropicProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 3600
        self._usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_by_model": {},
            "errors": [],
        }

    # Utility methods used in tests
    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        if not messages:
            raise ValueError("Messages cannot be empty")
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")
            if set(msg.keys()) != {"role", "content"}:
                raise ValueError("Each message must have 'role' and 'content' keys")
            if msg["role"] not in {"system", "user", "assistant"}:
                raise ValueError(
                    "Message role must be 'system', 'user', or 'assistant'"
                )
            if len(msg["content"]) > 1000:  # Limit content length
                raise ValueError("Message content exceeds maximum length")

    def _generate_cache_key(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        import json

        return json.dumps({"messages": messages, **kwargs}, sort_keys=True)

    def _cache_response(self, key: str, response: Dict[str, Any]) -> None:
        import time

        self._cache[key] = {"response": response, "timestamp": time.time()}

    def _get_cached_response(self, key: str) -> Optional[Dict[str, Any]]:
        import time

        cached = self._cache.get(key)
        if not cached:
            return None
        if time.time() - cached["timestamp"] > self._cache_ttl:
            del self._cache[key]
            return None
        return cached["response"]

    def clear_cache(self) -> None:
        self._cache.clear()

    def set_cache_ttl(self, ttl: int) -> None:
        self._cache_ttl = ttl


LLM_QUERY_LOGS: list[dict[str, str]] = []


def validate_prompt(prompt: str) -> str:
    """Validate and sanitize LLM prompt"""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if len(prompt) > 100000:
        raise ValueError("Prompt exceeds maximum length")

    sanitized = re.sub(
        r"<script[^>]*>.*?</script>", "", prompt, flags=re.IGNORECASE | re.DOTALL
    )
    sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)

    return sanitized.strip()


def ask_llm(prompt: str, max_tokens: Optional[int] = None) -> str:
    """
    Send a prompt to a configured large language model (LLM) provider and return the generated response.

    Depending on environment settings, queries either the Claude or OpenAI API using the specified model and API key. The function logs the last five prompt-response pairs for reference. If an error occurs during the LLM call, returns an error message string.

    Parameters:
        prompt (str): The user prompt to send to the LLM.

    Returns:
        str: The LLM's response text, or an error message if the call fails.
    """
    validated_prompt = validate_prompt(prompt)
    provider = env_settings.llm_provider.lower()

    if not env_settings.openai_api_key and not env_settings.anthropic_api_key:
        raise ValueError("No LLM API key configured")

    try:
        if provider == "claude":
            if not env_settings.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")

            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=env_settings.anthropic_api_key)
            resp = client.messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229"),
                messages=[{"role": "user", "content": validated_prompt}],
                max_tokens=max_tokens or 4000,
            )
            result = resp.content[0].text
        else:
            if not env_settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")

            import openai  # type: ignore

            client = openai.OpenAI(api_key=env_settings.openai_api_key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": validated_prompt}],
                max_tokens=max_tokens or 4000,
            )
            result = resp.choices[0].message.content.strip()

        LLM_QUERY_LOGS.append(
            {
                "prompt": validated_prompt[:100] + "..."
                if len(validated_prompt) > 100
                else validated_prompt,
                "response": result[:100] + "..." if len(result) > 100 else result,
            }
        )
        if len(LLM_QUERY_LOGS) > 5:
            LLM_QUERY_LOGS.pop(0)
        return result
    except Exception as e:  # pragma: no cover
        logger.error(f"LLM call failed: {e}")
        return f"LLM error: {e}"
