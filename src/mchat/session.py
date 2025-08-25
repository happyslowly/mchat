import json
import os
from pathlib import Path

import aiofiles
from loguru import logger


class ChatSession:
    def __init__(self):
        session_path = self._get_session_path()
        try:
            with open(session_path, "rb") as f:
                doc = json.loads(f.read())
                self._system_prompt = doc["system_prompt"]
                self._history = doc["history"]
                self._summary = doc["summary"]
        except Exception as e:
            logger.warning(f"Failed to load chat session file", e)
            self._system_prompt = ""
            self._summary = ""
            self._history: list[dict] = []

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
                    }
                )
            )

    def add_to_history(self, message: dict):
        self._history.append(message)

    def clear(self):
        self._system_prompt = ""
        self._summary = ""
        self._history: list[dict] = []
