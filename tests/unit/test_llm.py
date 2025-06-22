import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from adaptive_graph_of_thoughts.services.llm import ask_llm, LLM_QUERY_LOGS


@pytest.fixture
def mock_env_settings_claude():
    """Mock environment settings for Claude provider"""
    mock_settings = Mock()
    mock_settings.llm_provider = "claude"
    mock_settings.anthropic_api_key = "test_claude_key"
    return mock_settings


@pytest.fixture
def mock_env_settings_openai():
    """Mock environment settings for OpenAI provider"""
    mock_settings = Mock()
    mock_settings.llm_provider = "openai"
    mock_settings.openai_api_key = "test_openai_key"
    return mock_settings


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response"""
    mock_response = Mock()
    mock_response.content = [Mock(text="Claude response text")]
    return mock_response


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="  OpenAI response text  "))]
    return mock_response


@pytest.fixture(autouse=True)
def clear_llm_logs():
    """Clear LLM query logs before each test"""
    LLM_QUERY_LOGS.clear()
    yield
    LLM_QUERY_LOGS.clear()


class TestLLMClaudeProvider:
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.anthropic")
    def test_ask_llm_claude_success(
        self, mock_anthropic, mock_env_settings, mock_claude_response
    ):
        """Test successful LLM call with Claude provider"""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_key"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.Anthropic.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "Claude response text"
        mock_anthropic.Anthropic.assert_called_once_with(api_key="test_key")
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Test prompt"}],
        )

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.anthropic")
    @patch.dict(os.environ, {"CLAUDE_MODEL": "claude-3-opus-20240229"})
    def test_ask_llm_claude_custom_model(
        self, mock_anthropic, mock_env_settings, mock_claude_response
    ):
        """Test Claude provider with custom model from environment variable"""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_key"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.Anthropic.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "Claude response text"
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Test prompt"}],
        )

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.anthropic")
    def test_ask_llm_claude_api_error(self, mock_anthropic, mock_env_settings):
        """Test Claude provider with API error"""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_key"

        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.Anthropic.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "LLM error: API Error"

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.anthropic")
    def test_ask_llm_claude_case_insensitive(
        self, mock_anthropic, mock_env_settings, mock_claude_response
    ):
        """Test Claude provider with case insensitive provider name"""
        mock_env_settings.llm_provider = "CLAUDE"
        mock_env_settings.anthropic_api_key = "test_key"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_claude_response
        mock_anthropic.Anthropic.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "Claude response text"


class TestLLMOpenAIProvider:
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_openai_success(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test successful LLM call with OpenAI provider"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "OpenAI response text"
        mock_openai.OpenAI.assert_called_once_with(api_key="test_key")
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test prompt"}]
        )

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    @patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4"})
    def test_ask_llm_openai_custom_model(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test OpenAI provider with custom model from environment variable"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "OpenAI response text"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=[{"role": "user", "content": "Test prompt"}]
        )

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_openai_api_error(self, mock_openai, mock_env_settings):
        """Test OpenAI provider with API error"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "LLM error: API Error"

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_openai_default_provider(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test that non-Claude provider defaults to OpenAI"""
        mock_env_settings.llm_provider = "unknown_provider"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "OpenAI response text"

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_openai_response_stripping(self, mock_openai, mock_env_settings):
        """Test that OpenAI response content is properly stripped of whitespace"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="  \n  Response with whitespace  \n  "))
        ]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        assert result == "Response with whitespace"


class TestLLMQueryLogging:
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_llm_query_logging_single_call(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test that LLM queries are logged correctly"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        assert len(LLM_QUERY_LOGS) == 1
        assert LLM_QUERY_LOGS[0]["prompt"] == "Test prompt"
        assert LLM_QUERY_LOGS[0]["response"] == "OpenAI response text"

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_llm_query_logging_multiple_calls(self, mock_openai, mock_env_settings):
        """Test multiple LLM query logging"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Make multiple calls with different responses
        for i in range(3):
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=f"Response {i}"))]
            mock_client.chat.completions.create.return_value = mock_response

            ask_llm(f"Prompt {i}")

        assert len(LLM_QUERY_LOGS) == 3
        for i in range(3):
            assert LLM_QUERY_LOGS[i]["prompt"] == f"Prompt {i}"
            assert LLM_QUERY_LOGS[i]["response"] == f"Response {i}"

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_llm_query_logging_rotation(self, mock_openai, mock_env_settings):
        """Test that LLM query logs are rotated after 5 entries"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Make 7 calls to test rotation (should keep only last 5)
        for i in range(7):
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=f"Response {i}"))]
            mock_client.chat.completions.create.return_value = mock_response

            ask_llm(f"Prompt {i}")

        assert len(LLM_QUERY_LOGS) == 5
        # Should have entries 2-6 (first two rotated out)
        for i in range(5):
            expected_index = i + 2
            assert LLM_QUERY_LOGS[i]["prompt"] == f"Prompt {expected_index}"
            assert LLM_QUERY_LOGS[i]["response"] == f"Response {expected_index}"

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_llm_query_logging_error_not_logged(self, mock_openai, mock_env_settings):
        """Test that failed LLM queries are not logged"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        # Error responses should not be logged in LLM_QUERY_LOGS
        assert len(LLM_QUERY_LOGS) == 0
        assert result == "LLM error: API Error"


class TestLLMEdgeCases:
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_empty_prompt(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test LLM call with empty prompt"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            ask_llm("")

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_very_long_prompt(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test LLM call with very long prompt"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        long_prompt = "A" * 10000
        result = ask_llm(long_prompt)

        assert result == "OpenAI response text"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": long_prompt}]
        )

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_special_characters(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test LLM call with special characters in prompt"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        special_prompt = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?"
        result = ask_llm(special_prompt)

        assert result == "OpenAI response text"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": special_prompt}],
        )

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_unicode_prompt(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test LLM call with unicode characters in prompt"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        unicode_prompt = "Hello 世界! 🌍🚀 café naïve résumé"
        result = ask_llm(unicode_prompt)

        assert result == "OpenAI response text"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": unicode_prompt}],
        )

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_prompt_sanitization(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Ensure potentially harmful content is sanitized before sending"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        prompt = "<script>alert('x')</script>Hi javascript:alert('x')"
        ask_llm(prompt)

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi alert('x')"}],
        )

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.anthropic")
    def test_ask_llm_claude_malformed_response(self, mock_anthropic, mock_env_settings):
        """Test Claude provider with malformed response"""
        mock_env_settings.llm_provider = "claude"
        mock_env_settings.anthropic_api_key = "test_key"

        mock_response = Mock()
        mock_response.content = []  # Empty content list

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        result = ask_llm("Test prompt")

        assert "LLM error:" in result

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_ask_llm_openai_none_response(self, mock_openai, mock_env_settings):
        """Test OpenAI provider with None response content"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        assert "LLM error:" in result


class TestLLMParameterized:
    @pytest.mark.parametrize(
        "provider,expected_calls",
        [
            ("claude", "anthropic"),
            ("CLAUDE", "anthropic"),
            ("Claude", "anthropic"),
            ("openai", "openai"),
            ("OPENAI", "openai"),
            ("OpenAI", "openai"),
            ("gpt", "openai"),
            ("unknown", "openai"),
            ("", "openai"),
        ],
    )
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.anthropic")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_provider_routing(
        self, mock_openai, mock_anthropic, mock_env_settings, provider, expected_calls
    ):
        """Test that different provider names route to correct implementations"""
        mock_env_settings.llm_provider = provider
        mock_env_settings.anthropic_api_key = "claude_key"
        mock_env_settings.openai_api_key = "openai_key"

        if expected_calls == "anthropic":
            mock_claude_response = Mock()
            mock_claude_response.content = [Mock(text="Claude response")]
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_claude_response
            mock_anthropic.Anthropic.return_value = mock_client
        else:
            mock_openai_response = Mock()
            mock_openai_response.choices = [
                Mock(message=Mock(content="OpenAI response"))
            ]
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.OpenAI.return_value = mock_client

        result = ask_llm("Test prompt")

        if expected_calls == "anthropic":
            assert "Claude response" in result
            mock_anthropic.Anthropic.assert_called_once()
        else:
            assert "OpenAI response" in result
            mock_openai.OpenAI.assert_called_once()

    @pytest.mark.parametrize(
        "prompt",
        [
            "Simple prompt",
            "",
            "A" * 1000,
            "Unicode: 世界 🌍",
            "Special chars: @#$%^&*()",
            "Multi\nline\nprompt",
            "Tab\tcharacters",
            "Quotes 'single' \"double\"",
        ],
    )
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_various_prompt_types(
        self, mock_openai, mock_env_settings, prompt, mock_openai_response
    ):
        """Test LLM with various prompt types and characters"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        result = ask_llm(prompt)

        assert result == "OpenAI response text"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )


class TestLLMStressAndPerformance:
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_multiple_sequential_calls(self, mock_openai, mock_env_settings):
        """Test multiple sequential LLM calls"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Make 10 sequential calls
        for i in range(10):
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=f"Response {i}"))]
            mock_client.chat.completions.create.return_value = mock_response

            result = ask_llm(f"Prompt {i}")
            assert result == f"Response {i}"

        # Verify all calls were made
        assert mock_client.chat.completions.create.call_count == 10

        # Verify log rotation worked correctly (should only have last 5)
        assert len(LLM_QUERY_LOGS) == 5
        for i in range(5):
            expected_index = i + 5
            assert LLM_QUERY_LOGS[i]["prompt"] == f"Prompt {expected_index}"

    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.logger")
    def test_error_logging(self, mock_logger, mock_env_settings):
        """Test that errors are properly logged"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        # Mock import failure
        with patch(
            "adaptive_graph_of_thoughts.services.llm.openai",
            side_effect=ImportError("Module not found"),
        ):
            result = ask_llm("Test prompt")

            assert result == "LLM error: Module not found"
            mock_logger.error.assert_called_once_with(
                "LLM call failed: Module not found"
            )


class TestLLMIntegration:
    def test_llm_query_logs_global_state(self):
        """Test that LLM_QUERY_LOGS is properly managed as global state"""
        initial_length = len(LLM_QUERY_LOGS)

        with patch(
            "adaptive_graph_of_thoughts.services.llm.env_settings"
        ) as mock_env_settings:
            with patch("adaptive_graph_of_thoughts.services.llm.openai") as mock_openai:
                mock_env_settings.llm_provider = "openai"
                mock_env_settings.openai_api_key = "test_key"

                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content="Test response"))]
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.OpenAI.return_value = mock_client

                ask_llm("Test prompt")

                assert len(LLM_QUERY_LOGS) == initial_length + 1

    @patch.dict(os.environ, {}, clear=True)
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_environment_variable_defaults(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test default model values when environment variables are not set"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        ask_llm("Test prompt")

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test prompt"}]
        )

    @patch.dict(os.environ, {"CLAUDE_MODEL": "", "OPENAI_MODEL": ""})
    @patch("adaptive_graph_of_thoughts.services.llm.env_settings")
    @patch("adaptive_graph_of_thoughts.services.llm.openai")
    def test_empty_environment_variables(
        self, mock_openai, mock_env_settings, mock_openai_response
    ):
        """Test behavior with empty environment variables"""
        mock_env_settings.llm_provider = "openai"
        mock_env_settings.openai_api_key = "test_key"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.OpenAI.return_value = mock_client

        ask_llm("Test prompt")

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test prompt"}]
        )


def test_module_imports():
    """Test that all required modules can be imported without errors"""
    from adaptive_graph_of_thoughts.services.llm import ask_llm, LLM_QUERY_LOGS

    assert callable(ask_llm)
    assert isinstance(LLM_QUERY_LOGS, list)


def test_llm_query_logs_type():
    """Test that LLM_QUERY_LOGS is the correct type"""
    assert isinstance(LLM_QUERY_LOGS, list)

    sample_entry = {"prompt": "test", "response": "test"}
    LLM_QUERY_LOGS.append(sample_entry)

    assert len(LLM_QUERY_LOGS) >= 1
    assert "prompt" in LLM_QUERY_LOGS[-1]
    assert "response" in LLM_QUERY_LOGS[-1]

    LLM_QUERY_LOGS.clear()
