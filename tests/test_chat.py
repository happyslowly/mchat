from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from rich.console import Console

from mchat.chat import Chat
from mchat.config import Config
from mchat.llm_client import LLMClient


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
            google_api_key="",
            google_search_engine_id="",
        )
        chat = Chat(Console(), config=cfg)

        # Mock chat session with history
        chat._session_manager.current_session.system_prompt = "You are helpful"
        chat._session_manager.current_session.summary = "Previous chat about coding"
        chat._session_manager.current_session.history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]
        chat._session_manager.current_session.last_summarized_index = (
            1  # Summarized first 2 messages
        )
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
        mock_chat._session_manager.current_session.system_prompt = ""

        messages = mock_chat._build_messages("Test")

        assert len(messages) == 4  # system(summary only) + 2 unsummarized + current
        assert messages[0]["role"] == "system"
        assert (
            messages[0]["content"]
            == "\n\nPrevious conversation summary: Previous chat about coding"
        )

    def test_build_messages_no_summary(self, mock_chat):
        """Test with system prompt but no summary"""
        mock_chat._session_manager.current_session.summary = ""

        messages = mock_chat._build_messages("Test")

        assert len(messages) == 4  # system + 2 unsummarized + current
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    def test_build_messages_no_history(self, mock_chat):
        """Test with empty conversation history"""
        mock_chat._session_manager.current_session.history = []
        mock_chat._session_manager.current_session.last_summarized_index = -1

        messages = mock_chat._build_messages("First message")

        assert len(messages) == 2  # system + current
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "First message"


class TestLLMClientProcessing:
    """Test LLM client streaming processing"""

    @pytest.mark.asyncio
    async def test_stream_emits_thinking_and_content(self):
        from types import SimpleNamespace

        async def stream_gen():
            # First chunk with thinking + content
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content="Thinking...",
                            content="Hello",
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ]
            )
            # Stop chunk
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None, content=None, tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ]
            )

        with patch("mchat.llm_client.AsyncOpenAI") as MockAI:
            instance = MockAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=stream_gen())

            client = LLMClient("http://test", 0)
            events = []
            async for ev in client.stream_completion("model", []):
                events.append((ev.type, ev.data))

            assert ("thinking", "Thinking...") in events
            assert ("content", "Hello") in events

    @pytest.mark.asyncio
    async def test_stream_accumulates_multiple_content_chunks(self):
        from types import SimpleNamespace

        async def stream_gen():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None, content="Hello", tool_calls=None
                        ),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None, content=" world", tool_calls=None
                        ),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None, content=None, tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ]
            )

        with patch("mchat.llm_client.AsyncOpenAI") as MockAI:
            instance = MockAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=stream_gen())

            client = LLMClient("http://test", 0)
            contents = []
            async for ev in client.stream_completion("model", []):
                if ev.type == "content":
                    contents.append(ev.data)

            assert contents == ["Hello", " world"]


class TestLLMClientHeaders:
    """Test client initialization with API key"""

    def test_init_no_api_key_uses_dummy(self):
        with patch("mchat.llm_client.AsyncOpenAI") as MockAI:
            LLMClient("http://test", 0)
            MockAI.assert_called_with(
                base_url="http://test", api_key="dummy-key", timeout=0
            )

    def test_init_with_api_key(self):
        with patch("mchat.llm_client.AsyncOpenAI") as MockAI:
            LLMClient("http://test", 0, "sk-test123")
            MockAI.assert_called_with(
                base_url="http://test", api_key="sk-test123", timeout=0
            )


class TestConfig:
    """Test configuration model"""

    def test_config_creation_and_update(self):
        cfg = Config(
            base_url="http://test",
            model="test-model",
            max_history_turns=-1,
            google_api_key="",
            google_search_engine_id="",
        )
        assert cfg.base_url == "http://test"
        assert cfg.model == "test-model"
        # Update
        original_model = cfg.model
        cfg.model = "new-model"
        assert cfg.model == "new-model"
        assert cfg.model != original_model


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
            google_api_key="",
            google_search_engine_id="",
        )
        chat = Chat(Console(), config=cfg)

        # Setup history with 6 messages, limit=4 (keep last 4)
        chat._session_manager.current_session.system_prompt = ""
        chat._session_manager.current_session.summary = "Old summary"
        chat._session_manager.current_session.history = [
            {"role": "user", "content": "Msg 1"},  # Index 0
            {"role": "assistant", "content": "Reply 1"},  # Index 1
            {"role": "user", "content": "Msg 2"},  # Index 2
            {"role": "assistant", "content": "Reply 2"},  # Index 3
            {"role": "user", "content": "Msg 3"},  # Index 4
            {"role": "assistant", "content": "Reply 3"},  # Index 5
        ]
        chat._session_manager.current_session.last_summarized_index = (
            -1
        )  # Nothing summarized yet

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
            assert chat._session_manager.current_session.last_summarized_index == 1
            assert (
                chat._session_manager.current_session.summary
                == "New summary of messages 1-2"
            )

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
        chat._session_manager.current_session.last_summarized_index = (
            5  # Already summarized everything
        )

        original_summary = chat._session_manager.current_session.summary

        with patch.object(
            chat._llm_client, "completion", new_callable=AsyncMock
        ) as mock_completion:
            await chat._summarize()

            # Should not make API call
            mock_completion.assert_not_called()
            assert chat._session_manager.current_session.summary == original_summary


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_chat_updated.py -v
    pass
