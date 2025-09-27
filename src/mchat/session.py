import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from mchat.llm_client import LLMClient


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
    latest_session_id: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


Message = dict[str, str]


class SessionRepository(Protocol):
    def create_session(self, session: Session) -> Session: ...

    def get_session(self, session_id: int) -> Session | None: ...

    def get_sessions(self) -> list[Session]: ...

    def update_session(self, session: Session) -> Session: ...

    def delete_session(self, session_id: int) -> bool: ...

    def create_session_meta(self, meta: SessionMeta) -> SessionMeta: ...

    def get_session_meta(self) -> SessionMeta | None: ...

    def update_session_meta(self, meta: SessionMeta) -> SessionMeta: ...

    def list_session_messages(self, session_id: int) -> list[Message]: ...

    def append_session_message(self, session_id: int, message: Message) -> int: ...

    def delete_session_messages(self, session_id: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class SessionManagerSQLiteRepo(SessionRepository):
    def __init__(self, db_name: str = "sessions.sqlite3"):
        if "XDG_DATA_HOME" in os.environ:
            data_path = Path(os.environ["XDG_DATA_HOME"])
        else:
            data_path = Path.home() / ".local" / "share"
        db_dir = data_path / "mchat"
        db_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_dir / db_name)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_tables()

    def create_session(self, session: Session) -> Session:
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO sessions (
                    title,
                    model,
                    system_prompt,
                    summary,
                    last_summarized_index,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.title,
                    session.model,
                    session.system_prompt,
                    session.summary,
                    session.last_summarized_index,
                    _serialize_dt(session.created_at),
                    _serialize_dt(session.updated_at),
                ),
            )
        new_session = session.model_copy(update={"id": cursor.lastrowid, "history": []})
        # Persist any existing in-memory history
        for message in session.history:
            self.append_session_message(new_session.id, message)
            new_session.history.append(message)
        return new_session

    def get_session(self, session_id: int) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_session(row)

    def get_sessions(self) -> list[Session]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def update_session(self, session: Session) -> Session:
        if session.id is None or session.id < 0:
            raise ValueError("Session id must be set for update")
        with self._conn:
            self._conn.execute(
                """
                UPDATE sessions
                SET
                    title = ?,
                    model = ?,
                    system_prompt = ?,
                    summary = ?,
                    last_summarized_index = ?,
                    created_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    session.title,
                    session.model,
                    session.system_prompt,
                    session.summary,
                    session.last_summarized_index,
                    _serialize_dt(session.created_at),
                    _serialize_dt(session.updated_at),
                    session.id,
                ),
            )
        return session

    def delete_session(self, session_id: int) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,),
            )
        return cursor.rowcount == 1

    def create_session_meta(self, meta: SessionMeta) -> SessionMeta:
        with self._conn:
            existing = self._conn.execute(
                "SELECT ROWID FROM session_meta LIMIT 1"
            ).fetchone()
            if existing:
                self._conn.execute(
                    """
                    UPDATE session_meta
                    SET latest_session_id = ?, created_at = ?, updated_at = ?
                    """,
                    (
                        meta.latest_session_id,
                        _serialize_dt(meta.created_at),
                        _serialize_dt(meta.updated_at),
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO session_meta (latest_session_id, created_at, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        meta.latest_session_id,
                        _serialize_dt(meta.created_at),
                        _serialize_dt(meta.updated_at),
                    ),
                )
        return meta

    def get_session_meta(self) -> SessionMeta | None:
        row = self._conn.execute("SELECT * FROM session_meta LIMIT 1").fetchone()
        if not row:
            return None
        return self._row_to_meta(row)

    def update_session_meta(self, meta: SessionMeta) -> SessionMeta:
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE session_meta
                SET latest_session_id = ?, created_at = ?, updated_at = ?
                """,
                (
                    meta.latest_session_id,
                    _serialize_dt(meta.created_at),
                    _serialize_dt(meta.updated_at),
                ),
            )
            if cursor.rowcount == 0:
                self._conn.execute(
                    """
                    INSERT INTO session_meta (latest_session_id, created_at, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        meta.latest_session_id,
                        _serialize_dt(meta.created_at),
                        _serialize_dt(meta.updated_at),
                    ),
                )
        return meta

    def flush(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def _ensure_tables(self):
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    last_summarized_index INTEGER NOT NULL DEFAULT -1,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    UNIQUE (session_id, position)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_meta (
                    latest_session_id INTEGER,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (latest_session_id) REFERENCES sessions(id) ON DELETE SET NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
                ON session_messages (session_id)
                """
            )

    def list_session_messages(self, session_id: int) -> list[Message]:
        rows = self._conn.execute(
            """
            SELECT role, content
            FROM session_messages
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_session_message(self, session_id: int, message: Message) -> int:
        now = _serialize_dt(datetime.now(timezone.utc))
        with self._conn:
            row = self._conn.execute(
                "SELECT message_count FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Session `{session_id}` not found")
            position = row["message_count"] or 0
            role = message.get("role", "assistant") or "assistant"
            content = message.get("content", "") or ""
            if not isinstance(role, str):
                role = str(role)
            if not isinstance(content, str):
                content = str(content)
            self._conn.execute(
                """
                INSERT INTO session_messages (
                    session_id,
                    position,
                    role,
                    content,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    position,
                    role,
                    content,
                    now,
                ),
            )
            self._conn.execute(
                """
                UPDATE sessions
                SET message_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (position + 1, now, session_id),
            )
        return position

    def delete_session_messages(self, session_id: int) -> None:
        now = _serialize_dt(datetime.now(timezone.utc))
        with self._conn:
            self._conn.execute(
                "DELETE FROM session_messages WHERE session_id = ?",
                (session_id,),
            )
            self._conn.execute(
                """
                UPDATE sessions
                SET message_count = 0, updated_at = ?
                WHERE id = ?
                """,
                (now, session_id),
            )

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        history = self.list_session_messages(row["id"])
        return Session(
            id=row["id"],
            title=row["title"],
            model=row["model"],
            system_prompt=row["system_prompt"],
            summary=row["summary"],
            history=history,
            last_summarized_index=row["last_summarized_index"],
            created_at=_deserialize_dt(row["created_at"]),
            updated_at=_deserialize_dt(row["updated_at"]),
        )

    def _row_to_meta(self, row: sqlite3.Row) -> SessionMeta:
        return SessionMeta(
            latest_session_id=row["latest_session_id"],
            created_at=_deserialize_dt(row["created_at"]),
            updated_at=_deserialize_dt(row["updated_at"]),
        )


class SessionManager:
    def __init__(
        self,
        default_model: str,
        repo: SessionRepository | None = None,
        continue_last_session: bool = True,
    ):
        self._repo = repo or SessionManagerSQLiteRepo()
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
        position = self._repo.append_session_message(self._current_session.id, message)
        if not isinstance(position, int):
            position = len(self._current_session.history)
        # Ensure in-memory history mirrors persisted order
        if position == len(self._current_session.history):
            self._current_session.history.append(message)
        elif position < len(self._current_session.history):
            self._current_session.history.insert(position, message)
        else:
            # In case of mismatch, reload from storage for consistency
            self._current_session.history = self._repo.list_session_messages(
                self._current_session.id
            )
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
        start_index: int,
        end_index: int,
    ):
        current_messages = self._current_session.history.copy()

        messages_to_summarize = current_messages[start_index:end_index]
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
            self._current_session.last_summarized_index = end_index - 1
            self._current_session.updated_at = datetime.now(timezone.utc)
            self._repo.update_session(self._current_session)
        except Exception as e:
            raise RuntimeError(f"Failed to create conversation summary: {e}")

    def _get_or_create_session_meta(self) -> SessionMeta:
        meta = self._repo.get_session_meta()
        if not meta:
            meta = self._repo.create_session_meta(SessionMeta())
        return meta

    def _clear_history(self):
        self._repo.delete_session_messages(self._current_session.id)
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


def _serialize_dt(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def _deserialize_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp value: {value!r}") from exc
    raise TypeError(f"Unsupported timestamp value: {value!r}")
