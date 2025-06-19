import pytest
from unittest.mock import Mock, patch, MagicMock, call
import os
from typing import Dict, Any

from adaptive_graph_of_thoughts.services.llm import ask_llm, LLM_QUERY_LOGS
from adaptive_graph_of_thoughts.config import EnvSettings

@pytest.fixture
def mock_env_settings_openai():
    """Fixture providing OpenAI configuration."""
    return EnvSettings(
        llm_provider="openai",
        openai_api_key="test_openai_key",
        anthropic_api_key=None
    )

@pytest.fixture
def mock_env_settings_claude():
    """Fixture providing Claude configuration."""
    return EnvSettings(
        llm_provider="claude",
        openai_api_key=None,
        anthropic_api_key="test_claude_key"
    )

@pytest.fixture
def sample_prompt():
    """Fixture providing a sample prompt for testing."""
    return "What is the capital of France?"

@pytest.fixture
def empty_prompt():
    """Fixture providing an empty prompt for edge case testing."""
    return ""

@pytest.fixture
def long_prompt():
    """Fixture providing a long prompt for testing edge cases."""
    return "This is a very long prompt " * 100

@pytest.fixture
def mock_openai_response():
    """Fixture providing a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Paris is the capital of France."
    return mock_response

@pytest.fixture
def mock_claude_response():
    """Fixture providing a mock Claude API response."""
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Paris is the capital of France."
    return mock_response

@pytest.fixture(autouse=True)
def clean_query_logs():
    """Automatically clean query logs before and after each test."""
    LLM_QUERY_LOGS.clear()
    yield
    LLM_QUERY_LOGS.clear()


class TestOpenAIProvider:
    """Test cases for OpenAI provider functionality."""

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_openai_successful_call(self, mock_openai, mock_env_settings,
                                   mock_env_settings_openai, sample_prompt, mock_openai_response):
        """Test successful OpenAI API call."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        result = ask_llm(sample_prompt)

        assert result == "Paris is the capital of France."
        mock_openai.OpenAI.assert_called_once_with(api_key="test_key")
        mock_client.chat.completions.create.assert_called_once()

        # Verify query logging
        assert len(LLM_QUERY_LOGS) == 1
        assert LLM_QUERY_LOGS[0]["prompt"] == sample_prompt
        assert LLM_QUERY_LOGS[0]["response"] == "Paris is the capital of France."

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    @patch.dict(os.environ, {'OPENAI_MODEL': 'gpt-4'})
    def test_openai_custom_model_from_env(self, mock_openai, mock_env_settings,
                                         sample_prompt, mock_openai_response):
        """Test OpenAI with custom model from environment variable."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        result = ask_llm(sample_prompt)

        # Verify the model parameter was passed correctly
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-4'
        assert result == "Paris is the capital of France."

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_openai_default_model(self, mock_openai, mock_env_settings,
                                  sample_prompt, mock_openai_response):
        """Test OpenAI with default model when no env var set."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        # Ensure no OPENAI_MODEL env var
        with patch.dict(os.environ, {}, clear=True):
            result = ask_llm(sample_prompt)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-3.5-turbo'  # Default model

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_openai_content_stripping(self, mock_openai, mock_env_settings, sample_prompt):
        """Test that OpenAI response content is properly stripped."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        # Mock response with whitespace
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "  Response with whitespace  \n\t"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        result = ask_llm(sample_prompt)

        assert result == "Response with whitespace"  # Should be stripped


class TestClaudeProvider:
    """Test cases for Claude provider functionality."""

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.anthropic')
    def test_claude_successful_call(self, mock_anthropic, mock_env_settings,
                                    sample_prompt, mock_claude_response):
        """Test successful Claude API call."""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_claude_key"

        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_claude_response

        result = ask_llm(sample_prompt)

        assert result == "Paris is the capital of France."
        mock_anthropic.Anthropic.assert_called_once_with(api_key="test_claude_key")
        mock_client.messages.create.assert_called_once()

        # Verify query logging
        assert len(LLM_QUERY_LOGS) == 1
        assert LLM_QUERY_LOGS[0]["prompt"] == sample_prompt
        assert LLM_QUERY_LOGS[0]["response"] == "Paris is the capital of France."

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.anthropic')
    @patch.dict(os.environ, {'CLAUDE_MODEL': 'claude-3-opus-20240229'})
    def test_claude_custom_model_from_env(self, mock_anthropic, mock_env_settings,
                                          sample_prompt, mock_claude_response):
        """Test Claude with custom model from environment variable."""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_claude_key"

        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_claude_response

        result = ask_llm(sample_prompt)

        # Verify the model parameter was passed correctly
        call_args = mock_client.messages.create.call_args
        assert call_args[1]['model'] == 'claude-3-opus-20240229'
        assert result == "Paris is the capital of France."

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.anthropic')
    def test_claude_default_model(self, mock_anthropic, mock_env_settings,
                                  sample_prompt, mock_claude_response):
        """Test Claude with default model when no env var set."""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_claude_key"

        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_claude_response

        # Ensure no CLAUDE_MODEL env var
        with patch.dict(os.environ, {}, clear=True):
            result = ask_llm(sample_prompt)

        call_args = mock_client.messages.create.call_args
        assert call_args[1]['model'] == 'claude-3-sonnet-20240229'  # Default model

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.anthropic')
    def test_claude_message_format(self, mock_anthropic, mock_env_settings,
                                   sample_prompt, mock_claude_response):
        """Test that Claude messages are formatted correctly."""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_claude_key"

        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_claude_response

        ask_llm(sample_prompt)

        # Verify message format
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == sample_prompt


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    @patch('adaptive_graph_of_thoughts.services.llm.logger')
    def test_openai_api_error(self, mock_logger, mock_openai, mock_env_settings, sample_prompt):
        """Test handling of OpenAI API errors."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = ask_llm(sample_prompt)

        assert result == "LLM error: API Error"
        mock_logger.error.assert_called_once()
        assert "LLM call failed: API Error" in str(mock_logger.error.call_args)

        # Verify no query was logged on error
        assert len(LLM_QUERY_LOGS) == 0

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.anthropic')
    @patch('adaptive_graph_of_thoughts.services.llm.logger')
    def test_claude_api_error(self, mock_logger, mock_anthropic, mock_env_settings, sample_prompt):
        """Test handling of Claude API errors."""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_claude_key"

        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("Claude API Error")

        result = ask_llm(sample_prompt)

        assert result == "LLM error: Claude API Error"
        mock_logger.error.assert_called_once()
        assert "LLM call failed: Claude API Error" in str(mock_logger.error.call_args)

        # Verify no query was logged on error
        assert len(LLM_QUERY_LOGS) == 0

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_authentication_error(self, mock_openai, mock_env_settings, sample_prompt):
        """Test handling of authentication errors."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "invalid_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")

        result = ask_llm(sample_prompt)

        assert "Invalid API key" in result
        assert result.startswith("LLM error:")

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_rate_limit_error(self, mock_openai, mock_env_settings, sample_prompt):
        """Test handling of rate limit errors."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")

        result = ask_llm(sample_prompt)

        assert "Rate limit exceeded" in result
        assert result.startswith("LLM error:")

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_malformed_response_error(self, mock_openai, mock_env_settings, sample_prompt):
        """Test handling of malformed API responses."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = []  # Empty choices would cause IndexError
        mock_client.chat.completions.create.return_value = mock_response

        result = ask_llm(sample_prompt)

        assert result.startswith("LLM error:")
        assert "list index out of range" in result


class TestQueryLogging:
    """Test cases for query logging functionality."""

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_query_logging_single_request(self, mock_openai, mock_env_settings,
                                         sample_prompt, mock_openai_response):
        """Test query logging for a single request."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        result = ask_llm(sample_prompt)

        assert len(LLM_QUERY_LOGS) == 1
        log_entry = LLM_QUERY_LOGS[0]
        assert log_entry["prompt"] == sample_prompt
        assert log_entry["response"] == "Paris is the capital of France."

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_query_logging_multiple_requests(self, mock_openai, mock_env_settings, mock_openai_response):
        """Test query logging for multiple requests."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        prompts = [f"Question {i}" for i in range(3)]

        for prompt in prompts:
            ask_llm(prompt)

        assert len(LLM_QUERY_LOGS) == 3
        for i, log_entry in enumerate(LLM_QUERY_LOGS):
            assert log_entry["prompt"] == f"Question {i}"
            assert log_entry["response"] == "Paris is the capital of France."

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_query_logging_rotation_after_five(self, mock_openai, mock_env_settings, mock_openai_response):
        """Test that query logs are rotated after 5 entries."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        # Make 7 requests to test rotation
        for i in range(7):
            ask_llm(f"Question {i}")

        # Should only keep the last 5
        assert len(LLM_QUERY_LOGS) == 5

        # Verify the logs contain the last 5 requests (Questions 2-6)
        expected_prompts = [f"Question {i}" for i in range(2, 7)]
        actual_prompts = [log["prompt"] for log in LLM_QUERY_LOGS]
        assert actual_prompts == expected_prompts

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_no_logging_on_error(self, mock_openai, mock_env_settings, sample_prompt):
        """Test that failed requests are not logged."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = ask_llm(sample_prompt)

        # Error should be returned but not logged
        assert result.startswith("LLM error:")
        assert len(LLM_QUERY_LOGS) == 0

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_mixed_success_and_error_logging(self, mock_openai, mock_env_settings, mock_openai_response):
        """Test logging behavior with mixed successful and failed requests."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # First request succeeds
        mock_client.chat.completions.create.return_value = mock_openai_response
        ask_llm("Success question")

        # Second request fails
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        ask_llm("Error question")

        # Third request succeeds again
        mock_client.chat.completions.create.side_effect = None
        mock_client.chat.completions.create.return_value = mock_openai_response
        ask_llm("Another success")

        # Should only have 2 logged entries (the successful ones)
        assert len(LLM_QUERY_LOGS) == 2
        assert LLM_QUERY_LOGS[0]["prompt"] == "Success question"
        assert LLM_QUERY_LOGS[1]["prompt"] == "Another success"


class TestEdgeCases:
    """Test cases for edge cases and input validation."""

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_empty_string_prompt(self, mock_openai, mock_env_settings, mock_openai_response):
        """Test handling of empty string prompt."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        result = ask_llm("")

        # Should still work with empty prompt
        assert result == "Paris is the capital of France."

        # Verify empty prompt was logged
        assert len(LLM_QUERY_LOGS) == 1
        assert LLM_QUERY_LOGS[0]["prompt"] == ""

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_very_long_prompt(self, mock_openai, mock_env_settings, long_prompt, mock_openai_response):
        """Test handling of very long prompts."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        result = ask_llm(long_prompt)

        assert result == "Paris is the capital of France."
        assert LLM_QUERY_LOGS[0]["prompt"] == long_prompt

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_special_characters_in_prompt(self, mock_openai, mock_env_settings, mock_openai_response):
        """Test handling of special characters in prompts."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        special_prompt = "What about émojis 🤖 and symbols: @#$%^&*()?!"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        result = ask_llm(special_prompt)

        assert result == "Paris is the capital of France."
        assert LLM_QUERY_LOGS[0]["prompt"] == special_prompt

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_unicode_prompt(self, mock_openai, mock_env_settings, mock_openai_response):
        """Test handling of Unicode characters in prompts."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        unicode_prompt = "こんにちは世界 🌍 Здравствуй мир 🌎 مرحبا بالعالم 🌏"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        result = ask_llm(unicode_prompt)

        assert result == "Paris is the capital of France."
        assert LLM_QUERY_LOGS[0]["prompt"] == unicode_prompt

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    def test_unsupported_provider(self, mock_env_settings, sample_prompt):
        """Test handling of unsupported provider (should default to OpenAI)."""
        mock_env_settings.llm_provider = "unsupported_provider"
        mock_env_settings.openai_api_key = "test_key"

        with patch('adaptive_graph_of_thoughts.services.llm.openai') as mock_openai:
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "OpenAI response"
            mock_client.chat.completions.create.return_value = mock_response

            result = ask_llm(sample_prompt)

            # Should fall through to OpenAI since provider is not "claude"
            assert result == "OpenAI response"
            mock_openai.OpenAI.assert_called_once()


class TestProviderConfiguration:
    """Test cases for provider configuration and case sensitivity."""

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.anthropic')
    def test_claude_provider_case_insensitive(self, mock_anthropic, mock_env_settings,
                                               sample_prompt, mock_claude_response):
        """Test that Claude provider detection is case insensitive."""
        test_cases = ["claude", "CLAUDE", "Claude", "ClAuDe"]

        for provider_case in test_cases:
            LLM_QUERY_LOGS.clear()  # Clear logs between tests

            mock_env_settings.llm_provider = provider_case
            mock_env_settings.anthropic_api_key = "test_claude_key"

            mock_client = Mock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = mock_claude_response

            result = ask_llm(sample_prompt)

            assert result == "Paris is the capital of France."
            mock_anthropic.Anthropic.assert_called_with(api_key="test_claude_key")

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_openai_provider_case_insensitive(self, mock_openai, mock_env_settings,
                                              sample_prompt, mock_openai_response):
        """Test that non-Claude providers default to OpenAI regardless of case."""
        test_cases = ["openai", "OPENAI", "OpenAI", "gpt", "GPT", "anything_else"]

        for provider_case in test_cases:
            LLM_QUERY_LOGS.clear()  # Clear logs between tests

            mock_env_settings.llm_provider = provider_case
            mock_env_settings.openai_api_key = "test_openai_key"

            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            result = ask_llm(sample_prompt)

            assert result == "Paris is the capital of France."
            mock_openai.OpenAI.assert_called_with(api_key="test_openai_key")


class TestIntegrationScenarios:
    """Integration-style test cases for realistic usage scenarios."""

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    @patch('adaptive_graph_of_thoughts.services.llm.anthropic')
    def test_provider_switching_simulation(self, mock_anthropic, mock_openai, mock_env_settings):
        """Test switching between providers in the same session."""
        # Start with OpenAI
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "openai_key"

        mock_openai_client = Mock()
        mock_openai.OpenAI.return_value = mock_openai_client
        mock_openai_response = Mock()
        mock_openai_response.choices = [Mock()]
        mock_openai_response.choices[0].message = Mock()
        mock_openai_response.choices[0].message.content = "OpenAI Response"
        mock_openai_client.chat.completions.create.return_value = mock_openai_response

        result1 = ask_llm("First question")
        assert result1 == "OpenAI Response"

        # Switch to Claude
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "claude_key"

        mock_claude_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_claude_client
        mock_claude_response = Mock()
        mock_claude_response.content = [Mock()]
        mock_claude_response.content[0].text = "Claude Response"
        mock_claude_client.messages.create.return_value = mock_claude_response

        result2 = ask_llm("Second question")
        assert result2 == "Claude Response"

        # Verify both calls were logged
        assert len(LLM_QUERY_LOGS) == 2
        assert LLM_QUERY_LOGS[0]["response"] == "OpenAI Response"
        assert LLM_QUERY_LOGS[1]["response"] == "Claude Response"

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_conversation_simulation(self, mock_openai, mock_env_settings):
        """Test a multi-turn conversation simulation."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Simulate different responses for each turn
        responses = [
            "Hello! How can I help you?",
            "Paris is the capital of France.",
            "The population of Paris is about 2.2 million.",
            "Yes, I can help with more questions!"
        ]

        mock_responses = []
        for response_text in responses:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = response_text
            mock_responses.append(mock_response)

        mock_client.chat.completions.create.side_effect = mock_responses

        questions = [
            "Hello!",
            "What is the capital of France?",
            "What is the population of Paris?",
            "Can you help me with more questions?"
        ]

        results = []
        for question in questions:
            results.append(ask_llm(question))

        # Verify all responses
        assert results == responses

        # Verify conversation was logged correctly
        assert len(LLM_QUERY_LOGS) == 4
        for i, (question, response) in enumerate(zip(questions, responses)):
            assert LLM_QUERY_LOGS[i]["prompt"] == question
            assert LLM_QUERY_LOGS[i]["response"] == response

    @patch('adaptive_graph_of_thoughts.services.llm.env_settings')
    @patch('adaptive_graph_of_thoughts.services.llm.openai')
    def test_stress_logging_rotation(self, mock_openai, mock_env_settings):
        """Test that logging works correctly under stress with many requests."""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Standard response"
        mock_client.chat.completions.create.return_value = mock_response

        # Make 20 requests to thoroughly test rotation
        for i in range(20):
            ask_llm(f"Request number {i}")

        # Should still only have 5 logs (the last 5)
        assert len(LLM_QUERY_LOGS) == 5

        # Verify the last 5 requests are logged (15-19)
        for i, log_entry in enumerate(LLM_QUERY_LOGS):
            expected_prompt = f"Request number {15 + i}"
            assert log_entry["prompt"] == expected_prompt
            assert log_entry["response"] == "Standard response"