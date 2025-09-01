import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from rich.console import Console

from mchat.chat import Chat
from mchat.config import Config
from mchat.llm_client import LLMClient
from mchat.session import ChatSession


class TestChatMessageBuilding:
    """Test message building - core logic for LLM communication"""

    @pytest.fixture
    def mock_chat(self):
        cfg = Config(
            base_url="http://test",
            model="test-model",
            summary_model="test-model",
            max_history_turns=4,
            api_key=None,
            timeout=-1,
            save_interval=300,
        )
        chat = Chat(Console(), config=cfg)

        # Mock chat session with history
        chat._chat_session._system_prompt = "You are helpful"
        chat._chat_session._summary = "Previous chat about coding"
        chat._chat_session._history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]
        chat._chat_session.last_summarized_index = 1  # Summarized first 2 messages
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
        mock_chat._chat_session.last_summarized_index = -1

        messages = mock_chat._build_messages("First message")

        assert len(messages) == 2  # system + current
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "First message"


class TestLLMClientProcessing:
    """Test LLM client SSE processing"""

    @pytest.fixture
    def llm_client(self):
        return LLMClient("http://test", 0)

    def test_process_chunk_single_chunk(self, llm_client):
        """Test processing a single SSE chunk"""
        payload = {
            "choices": [
                {
                    "delta": {"content": "Hello", "reasoning_content": "Thinking..."},
                    "finish_reason": None,
                }
            ]
        }
        line = f"data: {json.dumps(payload)}"

        result = llm_client._process_chunk(line)

        assert result.thinking == "Thinking..."
        assert result.content == "Hello"

    def test_process_chunk_multiple_chunks(self, llm_client):
        """Test accumulating multiple chunks"""
        payload1 = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        payload2 = {
            "choices": [{"delta": {"content": " world"}, "finish_reason": None}]
        }

        r1 = llm_client._process_chunk(f"data: {json.dumps(payload1)}")
        assert r1.content == "Hello"

        r2 = llm_client._process_chunk(f"data: {json.dumps(payload2)}")
        assert r2.content == " world"

    def test_process_chunk_done_signal(self, llm_client):
        """Test DONE signal stops processing"""
        line = "data: [DONE]"

        result = llm_client._process_chunk(line)

        assert result.thinking == ""
        assert result.content == ""
        assert result.is_done is True

    def test_process_chunk_invalid_json(self, llm_client):
        """Test handling of malformed JSON chunks"""
        line = "data: {invalid json}"

        # Should not raise exception, just log error
        result = llm_client._process_chunk(line)
        assert result.thinking == ""
        assert result.content == ""

    def test_process_chunk_empty_choices(self, llm_client):
        """Test handling chunks with no choices"""
        chunk = json.dumps({"choices": []})
        line = f"data: {chunk}"

        result = llm_client._process_chunk(line)

        assert result.thinking == ""
        assert result.content == ""


class TestLLMClientHeaders:
    """Test HTTP header construction"""

    def test_build_headers_no_api_key(self):
        """Test headers without API key"""
        client = LLMClient("http://test", 0)

        headers = client._build_headers()

        assert headers == {"Content-Type": "application/json"}

    def test_build_headers_with_api_key(self):
        """Test headers with API key"""
        client = LLMClient("http://test", 0, "sk-test123")

        headers = client._build_headers()

        expected = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-test123",
        }
        assert headers == expected


class TestConfig:
    """Test configuration model"""

    def test_config_creation_and_update(self):
        cfg = Config(
            base_url="http://test",
            model="test-model",
            max_history_turns=-1,
        )
        assert cfg.base_url == "http://test"
        assert cfg.model == "test-model"
        # Update
        original_model = cfg.model
        cfg.model = "new-model"
        assert cfg.model == "new-model"
        assert cfg.model != original_model


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
        cfg = Config(
            base_url="http://test",
            model="test",
            summary_model="summary-model",
            max_history_turns=2,  # keep last 2 turns (4 messages)
            api_key=None,
        )
        chat = Chat(Console(), config=cfg)

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
        chat._chat_session.last_summarized_index = -1  # Nothing summarized yet

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
            assert chat._chat_session.last_summarized_index == 1
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
        chat._chat_session.last_summarized_index = 5  # Already summarized everything

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
