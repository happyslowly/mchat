from datetime import datetime, timezone

import pytest

from mchat.session import Session, SessionManagerSQLiteRepo, SessionMeta


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    repository = SessionManagerSQLiteRepo(db_name="test_sessions.sqlite")
    try:
        yield repository
    finally:
        repository.close()


def test_create_session_persists_history(repo):
    session = Session(
        id=-1,
        title="Test Session",
        model="test-model",
        system_prompt="sys",
        summary="previous summary",
        history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
    )

    created = repo.create_session(session)

    assert created.id > 0
    assert created.history == session.history

    fetched = repo.get_session(created.id)
    assert fetched is not None
    assert fetched.history == session.history
    assert repo.list_session_messages(created.id) == session.history


def test_append_session_message_updates_history(repo):
    created = repo.create_session(
        Session(id=-1, title="Empty", model="model", history=[])
    )

    repo.append_session_message(
        created.id, {"role": "user", "content": "first"}
    )
    repo.append_session_message(
        created.id, {"role": "assistant", "content": "reply"}
    )

    fetched = repo.get_session(created.id)
    assert fetched is not None
    assert [m["content"] for m in fetched.history] == ["first", "reply"]


def test_session_meta_upsert(repo):
    repo.create_session_meta(SessionMeta())

    stored = repo.get_session_meta()
    assert stored is not None
    assert stored.latest_session_id is None

    session = repo.create_session(
        Session(id=-1, title="For Meta", model="m", history=[])
    )

    updated = stored.model_copy(
        update={
            "latest_session_id": session.id,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    repo.update_session_meta(updated)

    stored_again = repo.get_session_meta()
    assert stored_again is not None
    assert stored_again.latest_session_id == session.id
    assert stored_again.created_at == stored.created_at
