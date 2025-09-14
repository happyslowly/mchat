import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field
from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage

from mchat.llm_client import LLMClient
from mchat.utils import db


class Session(BaseModel):
    id: int
    title: str
    model: str
    system_prompt: str = ""
    summary: str = ""
    history: list[dict] = Field(default_factory=list)
    last_summarized_index: int = -1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionMeta(BaseModel):
    id: int
    latest_session_id: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManager:
    def __init__(self, default_model: str, continue_last_session: bool = True):
        self._repo = SessionManagerRepo()

        self._model = default_model
        self._continue_last_session = continue_last_session
        self._session_meta = self._get_or_create_session_meta()
        self._current_session = self._get_or_create_current_session()

    @property
    def current_session(self) -> Session:
        return self._current_session

    def list_sessions(self) -> list[dict]:
        items: list[dict] = []
        for session in self._repo.get_sessions():
            items.append(
                {
                    "id": session.id,
                    "title": session.title,
                    "model": session.model,
                    "created_at": session.created_at.astimezone().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "updated_at": session.updated_at.astimezone().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
        items.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
        return items

    def create_session(self) -> Session:
        session = self._new_session()
        self._current_session = session
        self._session_meta.latest_session_id = session.id
        return session

    def switch_session(self, session_id: int) -> Session:
        session = self._repo.get_session(session_id)
        if not session:
            raise ValueError(f"Session id `{session_id}` not found")
        self._current_session = session
        self._session_meta.latest_session_id = session.id
        return session

    def delete_session(self, session_id: int):
        if session_id == self._current_session.id:
            raise ValueError("Cannot delete current active session")
        self._repo.delete_session(session_id)

    def add_to_history(self, message: dict):
        self._current_session.history.append(message)
        self._current_session.updated_at = datetime.now(timezone.utc)
        self._repo.update_session(self._current_session)

    def clear_session(self):
        self._clear_history()
        self._current_session.system_prompt = ""
        self._repo.update_session(self._current_session)

    def clear_history(self):
        self._clear_history()
        self._repo.update_session(self._current_session)

    def set_system_prompt(self, system_prompt):
        self._current_session.system_prompt = system_prompt
        self._current_session.updated_at = datetime.now(timezone.utc)
        self._repo.update_session(self._current_session)

    def set_model(self, model: str):
        self._current_session.model = model
        self._current_session.updated_at = datetime.now(timezone.utc)
        self._repo.update_session(self._current_session)

    def flush(self) -> None:
        self._repo.update_session(self._current_session)
        self._repo.update_session_meta(self._session_meta)
        self._repo.flush()

    def close(self) -> None:
        self._repo.update_session(self._current_session)
        self._repo.update_session_meta(self._session_meta)
        self._repo.close()

    async def generate_title(self, llm_client: LLMClient, summary_model: str):
        input = self._current_session.summary or "\n".join(
            str(self._current_session.history[:4])
        )

        prompt = f"""
You are given a conversation between a user and an AI assistant.
Your task is to generate a short, descriptive, and engaging title for the conversation.

Requirements:
- The title should be concise (max 8 words).
- It should capture the main topic or problem.
- Avoid generic phrases like "Chat" or "Conversation."
- Capitalize like a headline.

Conversation:
{ input }

Title:
"""

        try:
            self._current_session.title = await llm_client.completion(
                summary_model,
                [{"role": "user", "content": prompt}],
            )
            self._current_session.updated_at = datetime.now(timezone.utc)
            self._repo.update_session(self._current_session)
        except Exception as e:
            raise RuntimeError(f"Failed to generate session title: {e}")

    async def create_summary(
        self,
        llm_client: LLMClient,
        summary_model: str,
        max_history_turns: int = -1,
        end_index: int | None = None,
    ):
        start_index = self._current_session.last_summarized_index + 1

        current_messages = self._current_session.history.copy()

        if end_index is not None:
            messages_to_summarize = current_messages[start_index:end_index]
        else:
            messages_to_summarize = (
                current_messages[start_index:]
                if max_history_turns == -1
                else current_messages[start_index : -max_history_turns * 2]
            )
        if not messages_to_summarize:
            return

        recent_history_text = "\n".join(
            [f"{m['role']}:{m['content']}" for m in messages_to_summarize]
        )

        summary_prompt = f"""
Summarize this conversation, incorporating the previous summary if provided.

Previous summary: {self._current_session.summary}

Recent conversation:
{recent_history_text}

Create a concise summary (2-3 sentences) that:
- Incorporates key points from the previous summary
- Adds important new topics and conclusions
- Maintains context needed for future messages

Summary:
"""

        try:
            self._current_session.summary = await llm_client.completion(
                summary_model,
                [{"role": "user", "content": summary_prompt}],
            )
            new_index = start_index + len(messages_to_summarize) - 1
            self._current_session.last_summarized_index = new_index
            self._current_session.updated_at = datetime.now(timezone.utc)
            self._repo.update_session(self._current_session)
        except Exception as e:
            raise RuntimeError(f"Failed to create conversation summary: {e}")

    def _get_or_create_session_meta(self) -> SessionMeta:
        meta = self._repo.get_session_meta()
        if not meta:
            meta = self._repo.create_session_meta(SessionMeta(id=-1))
        return meta

    def _clear_history(self):
        self._current_session.history = []
        self._current_session.summary = ""
        self._current_session.last_summarized_index = -1
        self._current_session.updated_at = datetime.now(timezone.utc)

    def _get_or_create_current_session(self) -> Session:
        if (
            self._session_meta.latest_session_id is not None
            and self._continue_last_session
        ):
            session = self._repo.get_session(
                session_id=self._session_meta.latest_session_id
            )
            if not session:
                raise ValueError(
                    f"Session `{self._session_meta.latest_session_id}` not found"
                )
        else:
            session = self._new_session()
        self._current_session = session
        return session

    def _new_session(self) -> Session:
        session = self._repo.create_session(
            Session(id=-1, model=self._model, title="Untitled")
        )
        self._session_meta.latest_session_id = session.id
        return session


class SessionManagerRepo:
    def __init__(self, db_name: str = "sessions.db"):
        if "XDG_DATA_HOME" in os.environ:
            data_path = Path(os.environ["XDG_DATA_HOME"])
        else:
            data_path = Path.home() / ".local" / "share"
        db_dir = data_path / "mchat"
        db_dir.mkdir(parents=True, exist_ok=True)

        self._db_cache = CachingMiddleware(JSONStorage)
        self._db = TinyDB(db_dir / db_name, storage=self._db_cache)
        self._session_table = self._db.table("sessions")
        self._meta_table = self._db.table("meta")

    def create_session(self, session: Session) -> Session:
        return db.insert(self._session_table, session)

    def get_session(self, session_id: int) -> Session | None:
        return db.select_one(self._session_table, model_cls=Session, doc_id=session_id)

    def get_sessions(self) -> list[Session]:
        return db.select_all(self._session_table, Session)

    def update_session(self, session: Session) -> Session:
        return db.update(self._session_table, session)

    def delete_session(self, session_id: int) -> bool:
        return db.delete_one(self._session_table, session_id)

    def create_session_meta(self, meta: SessionMeta) -> SessionMeta:
        return db.insert(self._meta_table, meta)

    def get_session_meta(self) -> SessionMeta | None:
        return db.select_one(self._meta_table, SessionMeta)

    def update_session_meta(self, meta: SessionMeta) -> SessionMeta:
        return db.update(self._meta_table, meta)

    def flush(self):
        self._db_cache.flush()

    def close(self):
        self._db.close()
