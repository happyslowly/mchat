import json
import os
from pathlib import Path

import aiofiles
from loguru import logger

from mchat.schemas import History, Message


class HistoryManager:
    def __init__(self):
        self.history = self._load()

    @staticmethod
    def _get_history_path():
        if "XDG_DATA_HOME" in os.environ:
            data_path = Path(os.environ["XDG_DATA_HOME"])
        else:
            data_path = Path.home() / ".local" / "share"
        return data_path / "mchat" / "history.json"

    def _load(self) -> History:
        history_path = self._get_history_path()
        try:
            with open(history_path, "rb") as f:
                doc = json.loads(f.read())
                return History(
                    system_prompt=doc["system_prompt"], messages=doc["messages"]
                )
        except Exception as e:
            logger.warning(f"Failed to load history file", e)
            return History(system_prompt=None, messages=[])

    async def save(self):
        history_path = self._get_history_path()
        os.makedirs(history_path.parent, exist_ok=True)
        async with aiofiles.open(history_path, "w") as f:
            await f.write(json.dumps(self.history.model_dump()))

    def add(self, message: Message):
        self.history.messages.append(message)

    def update_system_prompt(self, system_prompt: str):
        self.history.system_prompt = system_prompt

    def clear(self):
        self.history = History(system_prompt=self.history.system_prompt, messages=[])
