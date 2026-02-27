from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger

from ..config import env_settings

try:
    import openai  # type: ignore
    from openai import AsyncOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore
    AsyncOpenAI = None  # type: ignore

try:
    import anthropic  # type: ignore
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


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

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - runtime only
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI provider with async support."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            default_headers=config.additional_headers,
        )

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": len(messages)}}

    async def generate_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs,
            )
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "total_tokens": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            }
        except Exception as e:
            msg = str(e)
            if "token" in msg.lower():
                raise TokenLimitExceeded(f"Token limit exceeded: {e}")
            if "rate" in msg.lower():
                raise RateLimitExceeded(f"Rate limit exceeded: {e}")
            raise LLMException(f"OpenAI API error: {e}")

    async def generate_streaming_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield {
                        "content": chunk.choices[0].delta.content,
                        "finish_reason": chunk.choices[0].finish_reason,
                    }
        except (LLMException, TokenLimitExceeded, RateLimitExceeded):
            raise
        except Exception as e:
            raise LLMException(f"OpenAI streaming error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic provider with async support."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
            default_headers=config.additional_headers,
        )

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        return {"content": [{"text": "ok"}], "usage": {"input_tokens": len(messages)}}

    def _convert_messages(self, messages: List[Dict[str, str]]):
        """Convert messages to Anthropic format, extracting system message."""
        system_msg = ""
        conv_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                conv_msgs.append(msg)
        return system_msg, conv_msgs

    async def generate_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        system_msg, conv_msgs = self._convert_messages(messages)
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                messages=conv_msgs,
                max_tokens=self.config.max_tokens,
                system=system_msg or None,
                **kwargs,
            )
            return {
                "content": response.content[0].text,
                "usage": {
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                },
                "model": response.model,
                "finish_reason": response.stop_reason,
            }
        except Exception as e:
            msg = str(e)
            if "token" in msg.lower():
                raise TokenLimitExceeded(f"Token limit exceeded: {e}")
            if "rate" in msg.lower():
                raise RateLimitExceeded(f"Rate limit exceeded: {e}")
            raise LLMException(f"Anthropic API error: {e}")

    async def generate_streaming_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
        system_msg, conv_msgs = self._convert_messages(messages)
        try:
            stream = await self.client.messages.create(
                model=self.config.model,
                messages=conv_msgs,
                max_tokens=self.config.max_tokens,
                system=system_msg or None,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text") and chunk.delta.text is not None:
                    yield {
                        "content": chunk.delta.text,
                        "finish_reason": chunk.stop_reason if hasattr(chunk, "stop_reason") else None,
                    }
        except (LLMException, TokenLimitExceeded, RateLimitExceeded):
            raise
        except Exception as e:
            raise LLMException(f"Anthropic streaming error: {e}")


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
                raise ValueError("Message role must be 'system', 'user', or 'assistant'")
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
            if anthropic is None:  # pragma: no cover
                raise ImportError("anthropic package is required")
            client = anthropic.Anthropic(api_key=env_settings.anthropic_api_key)
            resp = client.messages.create(
                model=os.getenv("CLAUDE_MODEL") or "claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.content[0].text
        else:
            if openai is None:  # pragma: no cover
                raise ImportError("openai package is required")
            client = openai.OpenAI(api_key=env_settings.openai_api_key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo",
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
