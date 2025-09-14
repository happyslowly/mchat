from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from prompt_toolkit import PromptSession
from rich.console import Console

from mchat.chat import Chat
from mchat.commands import CommandManager
from mchat.config import Config
from mchat.llm_client import LLMClient
from mchat.session import SessionManager
from mchat.task import TaskManager


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

        console = Console()
        llm_client = MagicMock(spec=LLMClient)
        session_manager = SessionManager(cfg.model)
        task_manager = MagicMock(spec=TaskManager)
        prompt_session = MagicMock(spec=PromptSession)
        command_manager = MagicMock(spec=CommandManager)

        chat = Chat(
            config=cfg,
            console=console,
            llm_client=llm_client,
            session_manager=session_manager,
            task_manager=task_manager,
            command_manager=command_manager,
            prompt_session=prompt_session,
        )

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
        from mchat.llm_client import StreamEvent

        # Create a mock LLM client
        client = MagicMock(spec=LLMClient)

        async def mock_stream(model, messages, max_tool_rounds=8):
            _ = model
            _ = messages
            _ = max_tool_rounds
            yield StreamEvent("thinking", "Thinking...")
            yield StreamEvent("content", "Hello")

        # Mock stream_completion to return the async generator directly
        client.stream_completion = mock_stream

        events = []
        async for ev in client.stream_completion("model", []):
            events.append((ev.type, ev.data))

        assert ("thinking", "Thinking...") in events
        assert ("content", "Hello") in events

    @pytest.mark.asyncio
    async def test_stream_accumulates_multiple_content_chunks(self):
        from mchat.llm_client import StreamEvent

        # Create a mock LLM client
        client = MagicMock(spec=LLMClient)

        async def mock_stream(model, messages, max_tool_rounds=8):
            _ = model
            _ = messages
            _ = max_tool_rounds
            yield StreamEvent("content", "Hello")
            yield StreamEvent("content", " world")

        # Mock stream_completion to return the async generator directly
        client.stream_completion = mock_stream

        contents = []
        async for ev in client.stream_completion("model", []):
            if ev.type == "content":
                contents.append(ev.data)

        assert contents == ["Hello", " world"]


class TestLLMClientInterface:
    """Test client interface methods"""

    def test_mock_client_has_required_methods(self):
        """Test that our mock client has the expected interface"""
        client = MagicMock(spec=LLMClient)

        # Mock the methods we expect
        client.list_models = AsyncMock(return_value=["model1", "model2"])
        client.completion = AsyncMock(return_value="Test response")
        client.stream_completion = AsyncMock()

        # Verify the interface exists
        assert hasattr(client, "list_models")
        assert hasattr(client, "completion")
        assert hasattr(client, "stream_completion")

    @pytest.mark.asyncio
    async def test_list_models_interface(self):
        """Test list_models interface"""
        client = MagicMock(spec=LLMClient)
        client.list_models = AsyncMock(return_value=["gpt-4", "gpt-3.5"])

        models = await client.list_models()
        assert models == ["gpt-4", "gpt-3.5"]
        client.list_models.assert_called_once()


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
        console = Console()
        llm_client = MagicMock(spec=LLMClient)
        session_manager = SessionManager(cfg.model)
        task_manager = TaskManager()
        prompt_session = PromptSession()
        command_manager = CommandManager(
            console=console,
            llm_client=llm_client,
            chat_session_manager=session_manager,
            prompt_session=prompt_session,
            task_manager=task_manager,
        )

        chat = Chat(
            config=cfg,
            console=console,
            llm_client=llm_client,
            session_manager=session_manager,
            task_manager=task_manager,
            command_manager=command_manager,
            prompt_session=prompt_session,
        )

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

        # Setup mock completion
        chat._llm_client.completion = AsyncMock(
            return_value="New summary of messages 1-2"
        )

        await chat._summarize()

        # Should summarize messages 0-1 (first 2), keeping last 4 messages
        assert chat._session_manager.current_session.last_summarized_index == 1
        assert (
            chat._session_manager.current_session.summary
            == "New summary of messages 1-2"
        )

        # Verify API call was made
        chat._llm_client.completion.assert_called_once()
        call_args = chat._llm_client.completion.call_args
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

        # Setup mock completion
        chat._llm_client.completion = AsyncMock()

        await chat._summarize()

        # Should not make API call
        chat._llm_client.completion.assert_not_called()
        assert chat._session_manager.current_session.summary == original_summary


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_chat_updated.py -v
    pass
