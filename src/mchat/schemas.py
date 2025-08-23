from typing import Literal

from pydantic import BaseModel

Role = Literal["user", "assistant", "system"]


class Config(BaseModel):
    base_url: str
    model: str
    api_key: str | None = None


class Message(BaseModel):
    role: Role
    content: str


class History(BaseModel):
    system_prompt: str | None
    messages: list[Message]
