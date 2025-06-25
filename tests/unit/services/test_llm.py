import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, List, Any
from dataclasses import asdict

# Import the LLM service components from the correct path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../git/src"))

from adaptive_graph_of_thoughts.services.llm import (
    LLMConfig,
    LLMService,
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    LLMException,
    TokenLimitExceeded,
    RateLimitExceeded,
)


# Test fixtures for configuration and mock data
@pytest.fixture
def base_config():
    """Basic LLM configuration for testing."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="test-api-key-12345",
        max_tokens=1000,
        temperature=0.7,
        timeout=30.0,
        max_retries=3,
        retry_delay=1.0,
    )


@pytest.fixture
def anthropic_config():
    """Anthropic LLM configuration for testing."""
    return LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        api_key="test-anthropic-key",
        max_tokens=2000,
        temperature=0.5,
    )


@pytest.fixture
def sample_messages():
    """Sample messages for testing completion requests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with coding?"},
    ]


@pytest.fixture
def simple_messages():
    """Simple user message for basic testing."""
    return [{"role": "user", "content": "Hello world"}]


@pytest.fixture
def mock_openai_response():
    """Mock response from OpenAI API."""
    mock_choice = MagicMock()
    mock_choice.message.content = "I'm doing well, thank you for asking!"
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 15
    mock_usage.completion_tokens = 10
    mock_usage.total_tokens = 25

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "gpt-4"

    return mock_response


@pytest.fixture
def mock_anthropic_response():
    """Mock response from Anthropic API."""
    mock_content = MagicMock()
    mock_content.text = "Hello! I'm Claude, nice to meet you."

    mock_usage = MagicMock()
    mock_usage.input_tokens = 12
    mock_usage.output_tokens = 8

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage = mock_usage
    mock_response.model = "claude-3-sonnet-20240229"
    mock_response.stop_reason = "end_turn"

    return mock_response


class TestLLMConfig:
    """Test LLMConfig dataclass functionality."""

    def test_config_initialization_with_defaults(self):
        """Test LLMConfig initialization with default values."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test-key")

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.max_tokens == 4000
        assert config.temperature == 0.7
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.base_url is None
        assert config.additional_headers == {}

    def test_config_initialization_with_custom_values(self):
        """Test LLMConfig initialization with custom values."""
        headers = {"Custom-Header": "test-value"}
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-opus-20240229",
            api_key="custom-key",
            max_tokens=2000,
            temperature=0.9,
            timeout=60.0,
            max_retries=5,
            retry_delay=2.0,
            base_url="https://custom.api.endpoint",
            additional_headers=headers,
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-3-opus-20240229"
        assert config.max_tokens == 2000
        assert config.temperature == 0.9
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.base_url == "https://custom.api.endpoint"
        assert config.additional_headers == headers

    def test_config_serialization(self, base_config):
        """Test LLMConfig can be serialized to dict."""
        config_dict = asdict(base_config)

        assert isinstance(config_dict, dict)
        assert config_dict["provider"] == "openai"
        assert config_dict["model"] == "gpt-4"
        assert config_dict["api_key"] == "test-api-key-12345"

    def test_config_edge_case_values(self):
        """Test LLMConfig with edge case values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="",  # Empty API key
            max_tokens=1,  # Minimum tokens
            temperature=0.0,  # Minimum temperature
            timeout=0.1,  # Very short timeout
            max_retries=0,  # No retries
            retry_delay=0.0,  # No delay
        )

        assert config.api_key == ""
        assert config.max_tokens == 1
        assert config.temperature == 0.0
        assert config.timeout == 0.1
        assert config.max_retries == 0
        assert config.retry_delay == 0.0


class TestLLMExceptions:
    """Test custom LLM exception classes."""

    def test_llm_exception_basic(self):
        """Test basic LLMException functionality."""
        error_msg = "Test error message"
        exception = LLMException(error_msg)

        assert str(exception) == error_msg
        assert isinstance(exception, Exception)

    def test_token_limit_exceeded_exception(self):
        """Test TokenLimitExceeded exception."""
        error_msg = "Token limit of 1000 exceeded"
        exception = TokenLimitExceeded(error_msg)

        assert str(exception) == error_msg
        assert isinstance(exception, LLMException)
        assert isinstance(exception, Exception)

    def test_rate_limit_exceeded_exception(self):
        """Test RateLimitExceeded exception."""
        error_msg = "Rate limit of 100 requests per minute exceeded"
        exception = RateLimitExceeded(error_msg)

        assert str(exception) == error_msg
        assert isinstance(exception, LLMException)
        assert isinstance(exception, Exception)

    def test_exception_inheritance_chain(self):
        """Test exception inheritance chain is correct."""
        assert issubclass(TokenLimitExceeded, LLMException)
        assert issubclass(RateLimitExceeded, LLMException)
        assert issubclass(LLMException, Exception)

    def test_exception_with_complex_data(self):
        """Test exceptions can handle complex error data."""
        error_data = {
            "error_code": "RATE_LIMIT_EXCEEDED",
            "retry_after": 60,
            "request_id": "req_123456",
        }
        exception = RateLimitExceeded(f"Rate limit exceeded: {json.dumps(error_data)}")

        error_str = str(exception)
        assert "RATE_LIMIT_EXCEEDED" in error_str
        assert "retry_after" in error_str
        assert "req_123456" in error_str


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    @pytest.fixture
    def openai_provider(self, base_config):
        """Create OpenAI provider instance for testing."""
        with patch(
            "adaptive_graph_of_thoughts.services.llm.AsyncOpenAI"
        ) as mock_openai:
            provider = OpenAIProvider(base_config)
            provider.client = AsyncMock()
            return provider

    def test_openai_provider_initialization(self, base_config):
        """Test OpenAI provider initialization."""
        with patch(
            "adaptive_graph_of_thoughts.services.llm.AsyncOpenAI"
        ) as mock_openai:
            provider = OpenAIProvider(base_config)

            assert provider.config == base_config
            mock_openai.assert_called_once_with(
                api_key=base_config.api_key,
                base_url=base_config.base_url,
                timeout=base_config.timeout,
                max_retries=base_config.max_retries,
                default_headers=base_config.additional_headers,
            )

    @pytest.mark.asyncio
    async def test_generate_completion_success(
        self, openai_provider, simple_messages, mock_openai_response
    ):
        """Test successful completion generation with OpenAI."""
        openai_provider.client.chat.completions.create.return_value = (
            mock_openai_response
        )

        result = await openai_provider.generate_completion(simple_messages)

        assert result["content"] == "I'm doing well, thank you for asking!"
        assert result["usage"]["total_tokens"] == 25
        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 10
        assert result["model"] == "gpt-4"
        assert result["finish_reason"] == "stop"

        openai_provider.client.chat.completions.create.assert_called_once_with(
            model=openai_provider.config.model,
            messages=simple_messages,
            max_tokens=openai_provider.config.max_tokens,
            temperature=openai_provider.config.temperature,
        )

    @pytest.mark.asyncio
    async def test_generate_completion_with_kwargs(
        self, openai_provider, simple_messages, mock_openai_response
    ):
        """Test completion generation with additional kwargs."""
        openai_provider.client.chat.completions.create.return_value = (
            mock_openai_response
        )

        kwargs = {"top_p": 0.9, "frequency_penalty": 0.5}
        result = await openai_provider.generate_completion(simple_messages, **kwargs)

        assert result["content"] == "I'm doing well, thank you for asking!"

        openai_provider.client.chat.completions.create.assert_called_once_with(
            model=openai_provider.config.model,
            messages=simple_messages,
            max_tokens=openai_provider.config.max_tokens,
            temperature=openai_provider.config.temperature,
            top_p=0.9,
            frequency_penalty=0.5,
        )

    @pytest.mark.asyncio
    async def test_generate_completion_token_limit_error(
        self, openai_provider, simple_messages
    ):
        """Test completion generation with token limit error."""
        error_msg = "Token limit exceeded for this request"
        openai_provider.client.chat.completions.create.side_effect = Exception(
            error_msg
        )

        with pytest.raises(TokenLimitExceeded, match="Token limit exceeded"):
            await openai_provider.generate_completion(simple_messages)

    @pytest.mark.asyncio
    async def test_generate_completion_rate_limit_error(
        self, openai_provider, simple_messages
    ):
        """Test completion generation with rate limit error."""
        error_msg = "Rate limit exceeded. Please try again later"
        openai_provider.client.chat.completions.create.side_effect = Exception(
            error_msg
        )

        with pytest.raises(RateLimitExceeded, match="Rate limit exceeded"):
            await openai_provider.generate_completion(simple_messages)

    @pytest.mark.asyncio
    async def test_generate_completion_generic_error(
        self, openai_provider, simple_messages
    ):
        """Test completion generation with generic error."""
        error_msg = "Network connection failed"
        openai_provider.client.chat.completions.create.side_effect = Exception(
            error_msg
        )

        with pytest.raises(LLMException, match="OpenAI API error"):
            await openai_provider.generate_completion(simple_messages)

    @pytest.mark.asyncio
    async def test_generate_streaming_completion_success(
        self, openai_provider, simple_messages
    ):
        """Test successful streaming completion generation."""
        # Mock streaming response chunks
        mock_chunks = [
            MagicMock(
                choices=[
                    MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)
                ]
            ),
            MagicMock(
                choices=[
                    MagicMock(delta=MagicMock(content=" world"), finish_reason=None)
                ]
            ),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="!"), finish_reason="stop")]
            ),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        openai_provider.client.chat.completions.create.return_value = mock_stream()

        results = []
        async for chunk in openai_provider.generate_streaming_completion(
            simple_messages
        ):
            results.append(chunk)

        assert len(results) == 3
        assert results[0]["content"] == "Hello"
        assert results[1]["content"] == " world"
        assert results[2]["content"] == "!"
        assert results[2]["finish_reason"] == "stop"

        openai_provider.client.chat.completions.create.assert_called_once_with(
            model=openai_provider.config.model,
            messages=simple_messages,
            max_tokens=openai_provider.config.max_tokens,
            temperature=openai_provider.config.temperature,
            stream=True,
        )

    @pytest.mark.asyncio
    async def test_generate_streaming_completion_empty_content(
        self, openai_provider, simple_messages
    ):
        """Test streaming completion with empty content chunks."""
        mock_chunks = [
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content=None), finish_reason=None)]
            ),
            MagicMock(
                choices=[
                    MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)
                ]
            ),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content=None), finish_reason="stop")]
            ),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        openai_provider.client.chat.completions.create.return_value = mock_stream()

        results = []
        async for chunk in openai_provider.generate_streaming_completion(
            simple_messages
        ):
            results.append(chunk)

        # Should only yield chunks with actual content
        assert len(results) == 1
        assert results[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_generate_streaming_completion_error(
        self, openai_provider, simple_messages
    ):
        """Test streaming completion with error."""
        error_msg = "Streaming connection failed"
        openai_provider.client.chat.completions.create.side_effect = Exception(
            error_msg
        )

        with pytest.raises(LLMException, match="OpenAI streaming error"):
            async for _ in openai_provider.generate_streaming_completion(
                simple_messages
            ):
                pass


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    @pytest.fixture
    def anthropic_provider(self, anthropic_config):
        """Create Anthropic provider instance for testing."""
        with patch(
            "adaptive_graph_of_thoughts.services.llm.anthropic.AsyncAnthropic"
        ) as mock_anthropic:
            provider = AnthropicProvider(anthropic_config)
            provider.client = AsyncMock()
            return provider

    def test_anthropic_provider_initialization(self, anthropic_config):
        """Test Anthropic provider initialization."""
        with patch(
            "adaptive_graph_of_thoughts.services.llm.anthropic.AsyncAnthropic"
        ) as mock_anthropic:
            provider = AnthropicProvider(anthropic_config)

            assert provider.config == anthropic_config
            mock_anthropic.assert_called_once_with(
                api_key=anthropic_config.api_key,
                timeout=anthropic_config.timeout,
                max_retries=anthropic_config.max_retries,
                default_headers=anthropic_config.additional_headers,
            )

    def test_convert_messages_with_system(self, anthropic_provider):
        """Test message conversion with system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        system_msg, conv_msgs = anthropic_provider._convert_messages(messages)

        assert system_msg == "You are a helpful assistant."
        assert len(conv_msgs) == 3
        assert conv_msgs[0] == {"role": "user", "content": "Hello"}
        assert conv_msgs[1] == {"role": "assistant", "content": "Hi there!"}
        assert conv_msgs[2] == {"role": "user", "content": "How are you?"}

    def test_convert_messages_without_system(self, anthropic_provider):
        """Test message conversion without system message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        system_msg, conv_msgs = anthropic_provider._convert_messages(messages)

        assert system_msg == ""
        assert len(conv_msgs) == 2
        assert conv_msgs == messages

    def test_convert_messages_multiple_systems(self, anthropic_provider):
        """Test message conversion with multiple system messages."""
        messages = [
            {"role": "system", "content": "First system message."},
            {"role": "system", "content": "Second system message."},
            {"role": "user", "content": "Hello"},
        ]

        system_msg, conv_msgs = anthropic_provider._convert_messages(messages)

        # Should use the last system message
        assert system_msg == "Second system message."
        assert len(conv_msgs) == 1
        assert conv_msgs[0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_generate_completion_success(
        self, anthropic_provider, sample_messages, mock_anthropic_response
    ):
        """Test successful completion generation with Anthropic."""
        anthropic_provider.client.messages.create.return_value = mock_anthropic_response

        result = await anthropic_provider.generate_completion(sample_messages)

        assert result["content"] == "Hello! I'm Claude, nice to meet you."
        assert result["usage"]["total_tokens"] == 20  # 12 + 8
        assert result["usage"]["prompt_tokens"] == 12
        assert result["usage"]["completion_tokens"] == 8
        assert result["model"] == "claude-3-sonnet-20240229"
        assert result["finish_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_generate_completion_with_system_message(
        self, anthropic_provider, mock_anthropic_response
    ):
        """Test completion generation with system message."""
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write a Python function."},
        ]

        anthropic_provider.client.messages.create.return_value = mock_anthropic_response

        result = await anthropic_provider.generate_completion(messages)

        assert result["content"] == "Hello! I'm Claude, nice to meet you."

        # Verify the client was called with correct parameters
        call_args = anthropic_provider.client.messages.create.call_args
        assert call_args[1]["system"] == "You are a coding assistant."
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["content"] == "Write a Python function."

    @pytest.mark.asyncio
    async def test_generate_completion_without_system_message(
        self, anthropic_provider, mock_anthropic_response
    ):
        """Test completion generation without system message."""
        messages = [{"role": "user", "content": "Hello"}]

        anthropic_provider.client.messages.create.return_value = mock_anthropic_response

        result = await anthropic_provider.generate_completion(messages)

        assert result["content"] == "Hello! I'm Claude, nice to meet you."

        # Verify system parameter is None when no system message
        call_args = anthropic_provider.client.messages.create.call_args
        assert call_args[1]["system"] is None

    @pytest.mark.asyncio
    async def test_generate_completion_token_limit_error(
        self, anthropic_provider, simple_messages
    ):
        """Test completion generation with token limit error."""
        error_msg = "Request exceeds maximum token limit"
        anthropic_provider.client.messages.create.side_effect = Exception(error_msg)

        with pytest.raises(TokenLimitExceeded, match="Token limit exceeded"):
            await anthropic_provider.generate_completion(simple_messages)

    @pytest.mark.asyncio
    async def test_generate_completion_rate_limit_error(
        self, anthropic_provider, simple_messages
    ):
        """Test completion generation with rate limit error."""
        error_msg = "API rate limit exceeded"
        anthropic_provider.client.messages.create.side_effect = Exception(error_msg)

        with pytest.raises(RateLimitExceeded, match="Rate limit exceeded"):
            await anthropic_provider.generate_completion(simple_messages)

    @pytest.mark.asyncio
    async def test_generate_completion_generic_error(
        self, anthropic_provider, simple_messages
    ):
        """Test completion generation with generic error."""
        error_msg = "Authentication failed"
        anthropic_provider.client.messages.create.side_effect = Exception(error_msg)

        with pytest.raises(LLMException, match="Anthropic API error"):
            await anthropic_provider.generate_completion(simple_messages)

    @pytest.mark.asyncio
    async def test_generate_streaming_completion_success(
        self, anthropic_provider, simple_messages
    ):
        """Test successful streaming completion generation."""
        mock_chunks = [
            MagicMock(delta=MagicMock(text="Hello"), stop_reason=None),
            MagicMock(delta=MagicMock(text=" world"), stop_reason=None),
            MagicMock(delta=MagicMock(text="!"), stop_reason="end_turn"),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        anthropic_provider.client.messages.create.return_value = mock_stream()

        results = []
        async for chunk in anthropic_provider.generate_streaming_completion(
            simple_messages
        ):
            results.append(chunk)

        assert len(results) == 3
        assert results[0]["content"] == "Hello"
        assert results[1]["content"] == " world"
        assert results[2]["content"] == "!"
        assert results[2]["finish_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_generate_streaming_completion_no_delta(
        self, anthropic_provider, simple_messages
    ):
        """Test streaming completion with chunks missing delta."""
        mock_chunks = [
            MagicMock(spec=[]),  # Missing delta attribute
            MagicMock(delta=MagicMock(text="Hello")),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        anthropic_provider.client.messages.create.return_value = mock_stream()

        results = []
        async for chunk in anthropic_provider.generate_streaming_completion(
            simple_messages
        ):
            results.append(chunk)

        # Should only yield chunks with valid delta
        assert len(results) == 1
        assert results[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_generate_streaming_completion_error(
        self, anthropic_provider, simple_messages
    ):
        """Test streaming completion with error."""
        error_msg = "Streaming failed"
        anthropic_provider.client.messages.create.side_effect = Exception(error_msg)

        with pytest.raises(LLMException, match="Anthropic streaming error"):
            async for _ in anthropic_provider.generate_streaming_completion(
                simple_messages
            ):
                pass


class TestLLMService:
    """Test main LLMService functionality."""

    def test_llm_service_initialization_openai(self, base_config):
        """Test LLMService initialization with OpenAI provider."""
        with patch(
            "adaptive_graph_of_thoughts.services.llm.OpenAIProvider"
        ) as mock_provider:
            service = LLMService(base_config)

            assert service.config == base_config
            mock_provider.assert_called_once_with(base_config)
            assert service._usage_stats["total_requests"] == 0
            assert service._usage_stats["total_tokens"] == 0
            assert service._usage_stats["total_cost"] == 0.0
            assert service._usage_stats["requests_by_model"] == {}
            assert service._usage_stats["errors"] == []
            assert service._cache == {}
            assert service._cache_ttl == 3600

    def test_llm_service_initialization_anthropic(self, anthropic_config):
        """Test LLMService initialization with Anthropic provider."""
        with patch(
            "adaptive_graph_of_thoughts.services.llm.AnthropicProvider"
        ) as mock_provider:
            service = LLMService(anthropic_config)

            assert service.config == anthropic_config
            mock_provider.assert_called_once_with(anthropic_config)

    def test_llm_service_unsupported_provider(self):
        """Test LLMService initialization with unsupported provider."""
        config = LLMConfig(
            provider="unsupported", model="test-model", api_key="test-key"
        )

        with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
            LLMService(config)

    def test_validate_messages_empty(self, base_config):
        """Test message validation with empty message list."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)

            with pytest.raises(ValueError, match="Messages cannot be empty"):
                service._validate_messages([])

    def test_validate_messages_invalid_format(self, base_config):
        """Test message validation with invalid message format."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)

            # Test non-dict message
            with pytest.raises(ValueError, match="Each message must be a dictionary"):
                service._validate_messages(["invalid"])

            # Test missing role
            with pytest.raises(
                ValueError, match="Each message must have 'role' and 'content' keys"
            ):
                service._validate_messages([{"content": "test"}])

            # Test missing content
            with pytest.raises(
                ValueError, match="Each message must have 'role' and 'content' keys"
            ):
                service._validate_messages([{"role": "user"}])

            # Test invalid role
            with pytest.raises(
                ValueError,
                match="Message role must be 'system', 'user', or 'assistant'",
            ):
                service._validate_messages([{"role": "invalid", "content": "test"}])

    def test_validate_messages_valid(self, base_config, sample_messages):
        """Test message validation with valid messages."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)

            # Should not raise any exception
            service._validate_messages(sample_messages)

    def test_generate_cache_key(self, base_config):
        """Test cache key generation."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)

            messages = [{"role": "user", "content": "test"}]
            key1 = service._generate_cache_key(messages)
            key2 = service._generate_cache_key(messages)

            # Same messages should generate same key
            assert key1 == key2
            assert isinstance(key1, str)

            # Different messages should generate different keys
            different_messages = [{"role": "user", "content": "different"}]
            key3 = service._generate_cache_key(different_messages)
            assert key1 != key3

    def test_generate_cache_key_with_kwargs(self, base_config):
        """Test cache key generation with kwargs."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)

            messages = [{"role": "user", "content": "test"}]
            key1 = service._generate_cache_key(messages)
            key2 = service._generate_cache_key(messages, top_p=0.9)

            # Different kwargs should generate different keys
            assert key1 != key2

    def test_cache_operations(self, base_config):
        """Test cache operations."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)

            cache_key = "test_key"
            response = {"content": "test response"}

            # Test caching response
            service._cache_response(cache_key, response)
            assert cache_key in service._cache
            assert service._cache[cache_key]["response"] == response
            assert "timestamp" in service._cache[cache_key]

            # Test getting cached response
            cached = service._get_cached_response(cache_key)
            assert cached == response

    def test_cache_expiration(self, base_config):
        """Test cache expiration."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)
            service.set_cache_ttl(1)  # 1 second TTL

            cache_key = "test_key"
            response = {"content": "test response"}

            # Cache response
            service._cache_response(cache_key, response)

            # Should be available immediately
            cached = service._get_cached_response(cache_key)
            assert cached == response

            # Mock time passage
            with patch(
                "adaptive_graph_of_thoughts.services.llm.time.time",
                return_value=time.time() + 2,
            ):
                cached = service._get_cached_response(cache_key)
                assert cached is None  # Should be expired
                assert cache_key not in service._cache  # Should be removed

    def test_clear_cache(self, base_config):
        """Test cache clearing."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)

            # Add some cache entries
            service._cache["key1"] = {
                "response": {"content": "test1"},
                "timestamp": time.time(),
            }
            service._cache["key2"] = {
                "response": {"content": "test2"},
                "timestamp": time.time(),
            }

            assert len(service._cache) == 2

            service.clear_cache()
            assert len(service._cache) == 0

    def test_set_cache_ttl(self, base_config):
        """Test setting cache TTL."""
        with patch("adaptive_graph_of_thoughts.services.llm.OpenAIProvider"):
            service = LLMService(base_config)

            assert service._cache_ttl == 3600  # Default

            service.set_cache_ttl(7200)
            assert service._cache_ttl == 7200
