import json
import os
from pathlib import Path

import aiofiles
from loguru import logger

from mchat.config import config_manager


class ChatSession:
    def __init__(self):
        session_path = self._get_session_path()
        try:
            with open(session_path, "rb") as f:
                doc = json.loads(f.read())
                self._system_prompt = doc["system_prompt"]
                self._history = doc["history"]
                self._summary = doc["summary"]
                self._last_summarized_index = doc.get("last_summarized_index", -1)
        except Exception as e:
            logger.warning(f"Failed to load chat session file", e)
            self._system_prompt = ""
            self._summary = ""
            self._history: list[dict] = []
            self._last_summarized_index = -1
        self._model = config_manager.config.model

    @staticmethod
    def _get_session_path():
        if "XDG_DATA_HOME" in os.environ:
            data_path = Path(os.environ["XDG_DATA_HOME"])
        else:
            data_path = Path.home() / ".local" / "share"
        return data_path / "mchat" / "session.json"

    @property
    def history(self):
        return self._history

    @property
    def system_prompt(self):
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self._system_prompt = value

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def last_summarized_index(self):
        return self._last_summarized_index

    @last_summarized_index.setter
    def last_summarized_index(self, value):
        self._last_summarized_index = value

    async def save(self):
        session_path = self._get_session_path()
        os.makedirs(session_path.parent, exist_ok=True)
        async with aiofiles.open(session_path, "w") as f:
            await f.write(
                json.dumps(
                    {
                        "system_prompt": self._system_prompt,
                        "history": self._history,
                        "summary": self._summary,
                        "last_summarized_index": self._last_summarized_index,
                    }
                )
            )

    def add_to_history(self, message: dict):
        self._history.append(message)

    def clear(self):
        self._system_prompt = ""
        self._summary = ""
        self._history: list[dict] = []
        self._last_summarized_index = -1

    async def create_summary(self, llm_client, config, end_index: int | None = None):
        start_index = self._last_summarized_index + 1

        current_messages = self._history.copy()

        if end_index is not None:
            messages_to_summarize = current_messages[start_index:end_index]
        else:
            messages_to_summarize = (
                current_messages[start_index:]
                if config.max_history_turns == -1
                else current_messages[start_index : -config.max_history_turns * 2]
            )
        if not messages_to_summarize:
            return

        recent_history_text = "\n".join(
            [f"{m['role']}:{m['content']}" for m in messages_to_summarize]
        )

        summary_prompt = f"""
Summarize this conversation, incorporating the previous summary if provided.

Previous summary: {self._summary}

Recent conversation:
{recent_history_text}

Create a concise summary (2-3 sentences) that:
- Incorporates key points from the previous summary
- Adds important new topics and conclusions
- Maintains context needed for future messages

Summary:
"""

        try:
            summary_model = config.summary_model or config.model
            self._summary = await llm_client.completion(
                summary_model, [{"role": "user", "content": summary_prompt}]
            )
            new_index = start_index + len(messages_to_summarize) - 1
            self._last_summarized_index = new_index
        except Exception as e:
            raise RuntimeError(f"Failed to create conversation summary: {e}")
