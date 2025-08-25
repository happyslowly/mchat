import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from mchat.chat import Chat
from mchat.config import Config, ConfigManager
from mchat.llm_client import LLMClient
from mchat.session import ChatSession


class TestChatMessageBuilding:
    """Test message building - core logic for LLM communication"""

    @pytest.fixture
    def mock_chat(self):
        with patch("mchat.config.ConfigManager._load_config") as mock_config:
            mock_config.return_value = Config(
                base_url="http://test",
                model="test-model",
                summary_model="test-model",
                history_limit=4,
                api_key=None,
            )
            with patch(
                "mchat.llm_client.LLMClient.list_models", return_value=["test-model"]
            ):
                chat = Chat()

                # Mock chat session with history
                chat._chat_session._system_prompt = "You are helpful"
                chat._chat_session._summary = "Previous chat about coding"
                chat._chat_session._history = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm good"},
                ]
                chat._last_summarized_index = 1  # Summarized first 2 messages
                return chat

    def test_build_messages_with_system_and_summary(self, mock_chat):
        """Test message building includes system prompt + summary + unsummarized history"""
        messages = mock_chat._build_messages("What's next?")

        assert len(messages) == 4  # system + 2 unsummarized + current
        assert messages[0]["role"] == "system"
        assert "You are helpful" in messages[0]["content"]
        assert "Previous chat about coding" in messages[0]["content"]

        # Should include unsummarized messages (index 2-3)
        assert messages[1]["content"] == "How are you?"
        assert messages[2]["content"] == "I'm good"

        # Current user message
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "What's next?"

    def test_build_messages_no_system_prompt(self, mock_chat):
        """Test with only summary, no system prompt"""
        mock_chat._chat_session._system_prompt = ""

        messages = mock_chat._build_messages("Test")

        assert len(messages) == 4  # system(summary only) + 2 unsummarized + current
        assert messages[0]["role"] == "system"
        assert (
            messages[0]["content"]
            == "\n\nPrevious conversation summary: Previous chat about coding"
        )

    def test_build_messages_no_summary(self, mock_chat):
        """Test with system prompt but no summary"""
        mock_chat._chat_session._summary = ""

        messages = mock_chat._build_messages("Test")

        assert len(messages) == 4  # system + 2 unsummarized + current
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    def test_build_messages_no_history(self, mock_chat):
        """Test with empty conversation history"""
        mock_chat._chat_session._history = []
        mock_chat._last_summarized_index = -1

        messages = mock_chat._build_messages("First message")

        assert len(messages) == 2  # system + current
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "First message"


class TestLLMClientProcessing:
    """Test LLM client SSE processing"""

    @pytest.fixture
    def llm_client(self):
        return LLMClient("http://test", None)

    def test_process_chunk_single_chunk(self, llm_client):
        """Test processing a single SSE chunk"""
        chunk_data = json.dumps(
            {
                "choices": [
                    {"delta": {"content": "Hello", "reasoning_content": "Thinking..."}}
                ]
            }
        )
        sse_data = f"data: {chunk_data}\n".encode()

        thinking, content = llm_client._process_chunk(sse_data, "", "")

        assert thinking == "Thinking..."
        assert content == "Hello"

    def test_process_chunk_multiple_chunks(self, llm_client):
        """Test accumulating multiple chunks"""
        chunk1 = json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
        chunk2 = json.dumps({"choices": [{"delta": {"content": " world"}}]})

        data1 = f"data: {chunk1}\n".encode()
        data2 = f"data: {chunk2}\n".encode()

        thinking, content = llm_client._process_chunk(data1, "", "")
        assert content == "Hello"

        thinking, content = llm_client._process_chunk(data2, thinking, content)
        assert content == "Hello world"

    def test_process_chunk_done_signal(self, llm_client):
        """Test DONE signal stops processing"""
        data = "data: [DONE]\ndata: should_be_ignored\n".encode()

        thinking, content = llm_client._process_chunk(data, "", "")

        assert thinking == ""
        assert content == ""

    def test_process_chunk_invalid_json(self, llm_client):
        """Test handling of malformed JSON chunks"""
        data = "data: {invalid json}\n".encode()

        # Should not raise exception, just log error
        thinking, content = llm_client._process_chunk(data, "", "")
        assert thinking == ""
        assert content == ""

    def test_process_chunk_empty_choices(self, llm_client):
        """Test handling chunks with no choices"""
        chunk = json.dumps({"choices": []})
        data = f"data: {chunk}\n".encode()

        thinking, content = llm_client._process_chunk(data, "", "")

        assert thinking == ""
        assert content == ""


class TestLLMClientHeaders:
    """Test HTTP header construction"""

    def test_build_headers_no_api_key(self):
        """Test headers without API key"""
        client = LLMClient("http://test", None)

        headers = client._build_headers()

        assert headers == {"Content-Type": "application/json"}

    def test_build_headers_with_api_key(self):
        """Test headers with API key"""
        client = LLMClient("http://test", "sk-test123")

        headers = client._build_headers()

        expected = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-test123",
        }
        assert headers == expected


class TestConfigManager:
    """Test configuration loading and management"""

    def test_config_singleton(self):
        """Test ConfigManager is a proper singleton"""
        from mchat.config import config_manager

        manager1 = ConfigManager()
        manager2 = ConfigManager()

        assert manager1 is manager2
        assert manager1 is config_manager

    def test_config_update(self):
        """Test runtime config updates"""
        with patch("mchat.config.ConfigManager._load_config") as mock_load:
            mock_load.return_value = Config(
                base_url="http://test", model="test-model", history_limit=-1
            )

            manager = ConfigManager()
            original_model = manager.config.model

            # Update config
            manager.update(model="new-model")

            assert manager.config.model == "new-model"
            assert manager.config.model != original_model


class TestChatSession:
    """Test chat session management"""

    @pytest.fixture
    def mock_session(self):
        with patch("builtins.open"):
            with patch(
                "json.loads",
                return_value={
                    "system_prompt": "Test prompt",
                    "history": [{"role": "user", "content": "test"}],
                    "summary": "Test summary",
                },
            ):
                return ChatSession()

    def test_session_properties(self, mock_session):
        """Test session property access"""
        assert mock_session.system_prompt == "Test prompt"
        assert mock_session.summary == "Test summary"
        assert len(mock_session.history) == 1
        assert mock_session.history[0]["content"] == "test"

    def test_session_updates(self, mock_session):
        """Test session property updates"""
        mock_session.system_prompt = "New prompt"
        assert mock_session.system_prompt == "New prompt"

        mock_session.summary = "New summary"
        assert mock_session.summary == "New summary"

    def test_add_to_history(self, mock_session):
        """Test adding messages to history"""
        original_length = len(mock_session.history)

        mock_session.add_to_history({"role": "assistant", "content": "response"})

        assert len(mock_session.history) == original_length + 1
        assert mock_session.history[-1]["content"] == "response"

    def test_clear_session(self, mock_session):
        """Test clearing session data"""
        mock_session.clear()

        assert mock_session.system_prompt == ""
        assert mock_session.summary == ""
        assert len(mock_session.history) == 0


class TestSummarization:
    """Test background summarization logic"""

    @pytest_asyncio.fixture
    async def mock_chat_with_history(self):
        """Create chat with realistic conversation history"""
        with patch("mchat.config.ConfigManager._load_config") as mock_config:
            mock_config.return_value = Config(
                base_url="http://test",
                model="test",
                summary_model="summary-model",
                history_limit=4,
                api_key=None,
            )
            with patch("mchat.llm_client.LLMClient.list_models", return_value=["test"]):
                chat = Chat()

                # Setup history with 6 messages, limit=4 (keep last 4)
                chat._chat_session._system_prompt = ""
                chat._chat_session._summary = "Old summary"
                chat._chat_session._history = [
                    {"role": "user", "content": "Msg 1"},  # Index 0
                    {"role": "assistant", "content": "Reply 1"},  # Index 1
                    {"role": "user", "content": "Msg 2"},  # Index 2
                    {"role": "assistant", "content": "Reply 2"},  # Index 3
                    {"role": "user", "content": "Msg 3"},  # Index 4
                    {"role": "assistant", "content": "Reply 3"},  # Index 5
                ]
                chat._last_summarized_index = -1  # Nothing summarized yet

                return chat

    @pytest.mark.asyncio
    async def test_summarize_with_history_limit(self, mock_chat_with_history):
        """Test summarization respects history limit"""
        chat = mock_chat_with_history

        # Mock the LLM client completion call
        with patch.object(
            chat._llm_client, "completion", new_callable=AsyncMock
        ) as mock_completion:
            mock_completion.return_value = "New summary of messages 1-2"

            await chat._summarize()

            # Should summarize messages 0-1 (first 2), keeping last 4 messages
            assert chat._last_summarized_index == 1
            assert chat._chat_session.summary == "New summary of messages 1-2"

            # Verify API call was made
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args
            # Use summary_model for summarization
            expected_model = chat._config.summary_model or chat._config.model
            assert call_args[0][0] == expected_model

            # Check summary prompt contains the right messages
            prompt_content = call_args[0][1][0]["content"]
            assert "Old summary" in prompt_content
            assert "Msg 1" in prompt_content
            assert "Reply 1" in prompt_content
            # Should not contain messages that will be kept (2-5)
            assert "Msg 3" not in prompt_content

    @pytest.mark.asyncio
    async def test_summarize_no_messages_to_process(self, mock_chat_with_history):
        """Test summarization when no new messages to summarize"""
        chat = mock_chat_with_history
        chat._last_summarized_index = 5  # Already summarized everything

        original_summary = chat._chat_session.summary

        with patch.object(
            chat._llm_client, "completion", new_callable=AsyncMock
        ) as mock_completion:
            await chat._summarize()

            # Should not make API call
            mock_completion.assert_not_called()
            assert chat._chat_session.summary == original_summary


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_chat_updated.py -v
    pass

