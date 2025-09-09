from typing import TypeVar

from pydantic import BaseModel
from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.table import Table

TModel = TypeVar("TModel", bound=BaseModel)


def insert(table: Table, data: TModel) -> TModel:
    doc_id = table.insert(data.model_dump(exclude={"id"}, mode="json"))
    data = data.model_copy(update={"id": doc_id})
    return data


def update(table: Table, data: TModel, doc_id: int | None = None) -> TModel:
    if doc_id is None:
        doc_id = getattr(data, "id", None)
        if doc_id is None:
            raise ValueError(f"Row `{data}` doesn't contain an id field")
    row = table.get(doc_id=doc_id)
    if not row:
        raise ValueError(f"Table `{table}` doesn't contain document `{doc_id}`")
    table.update(data.model_dump(exclude={"id"}, mode="json"), doc_ids=[doc_id])
    return data


def select_one(
    table: Table, model_cls: type[TModel], doc_id: int | None = None
) -> TModel | None:
    if doc_id is None:
        docs = table.all()
        if not docs:
            return None
        if len(docs) != 1:
            raise ValueError(f"Table `{table}` contains multiple rows")
        doc = docs[0]
    else:
        doc = table.get(doc_id=doc_id)
        if not doc:
            return None
        if isinstance(doc, list):
            raise ValueError(
                f"Table `{table}` has multiple rows with same the doc id `{doc_id}`"
            )
    return model_cls.model_validate({**doc, "id": doc.doc_id})


def select_all(table: Table, model_cls: type[TModel]) -> list[TModel]:
    result = []
    for doc in table:
        result.append(model_cls.model_validate({**doc, "id": doc.doc_id}))
    return result


def delete_one(table: Table, doc_id: int) -> bool:
    deleted = table.remove(doc_ids=[doc_id])
    return len(deleted) == 1


def flush(db: TinyDB):
    if isinstance(db.storage, CachingMiddleware):
        db.storage.flush()


def close(db: TinyDB):
    db.close()
