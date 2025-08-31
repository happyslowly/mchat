import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

import httpx
from loguru import logger

from mchat.tools import exec_tool_calls, get_tool_schemas

EventType = Literal["thinking", "content", "tool_call", "tool_complete", "error"]


@dataclass
class ChunkResult:
    thinking: str = ""
    content: str = ""
    error: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    is_done: bool = False


@dataclass
class StreamEvent:
    type: EventType
    data: str


class LLMClient:
    def __init__(self, base_url: str, timeout: int, api_key: str | None = None):
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout if timeout > 0 else None

    def list_models(self) -> list[str]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._base_url + "/models")
            response.raise_for_status()
            data = response.json()["data"]
            models = []
            for entry in data:
                if entry["object"] == "model":
                    models.append(entry["id"])
            return models

    async def stream_completion(
        self, model: str, messages: list[dict]
    ) -> AsyncIterator[StreamEvent]:
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "tools": get_tool_schemas(),
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while True:
                try:
                    async with client.stream(
                        "POST",
                        self._base_url + "/chat/completions",
                        json=body,
                        headers=self._build_headers(),
                    ) as response:
                        tool_calls = []
                        response.raise_for_status()
                        async for data in response.aiter_bytes():
                            result = self._process_chunk(
                                data,
                            )
                            if result.is_done:
                                return
                            if result.error:
                                yield StreamEvent("error", result.error)
                                return

                            if result.thinking:
                                yield StreamEvent("thinking", result.thinking)
                            if result.content:
                                yield StreamEvent("content", result.content)
                            if result.tool_calls:
                                tool_calls.extend(result.tool_calls)

                    tool_calls_complete = self._merge_tool_calls(tool_calls)
                    if tool_calls_complete:
                        body["messages"].append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls_complete,
                            }
                        )
                        for tool_call in tool_calls_complete:
                            yield StreamEvent("tool_call", json.dumps(tool_call))
                        tool_messages = await exec_tool_calls(tool_calls_complete)
                        yield StreamEvent(
                            "tool_complete",
                            f"{len(tool_calls_complete)} tool(s) executed",
                        )
                        body["messages"].extend(tool_messages)
                except httpx.TimeoutException as _:
                    raise TimeoutError("API call timed out")
                except Exception as e:
                    raise RuntimeError(f"API error: {e}")

    async def completion(self, model: str, messages: list[dict]) -> str:
        body = {
            "model": model,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    self._base_url + "/chat/completions",
                    json=body,
                    headers=self._build_headers(),
                )
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.TimeoutException as _:
                raise TimeoutError("API call timed out")
            except Exception as e:
                raise RuntimeError(f"API error: {e}")

    def _process_chunk(
        self,
        data: bytes,
    ) -> ChunkResult:
        result = ChunkResult()

        for chunk in data.decode().split("\n"):
            if not chunk:
                continue
            # special handing the error chunk due to llama.cpp bug
            if chunk.startswith("error: "):
                chunk_data = chunk[7:]
                try:
                    error = json.loads(chunk[7:])["message"]
                except Exception as _:
                    error = chunk
                result.error += error
                return result

            if not chunk.startswith("data: "):
                continue

            chunk = chunk[6:]
            if chunk == "[DONE]":
                break

            try:
                chunk_data = json.loads(chunk)
            except json.JSONDecodeError as _:
                logger.error(f"Failed to parse chunk: `{chunk}`")
                continue
            choices = chunk_data["choices"]
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice["finish_reason"]
            if finish_reason == "stop":
                result.is_done = True
                return result

            delta = choice["delta"]
            if not delta:
                continue

            tool_calls_delta = delta.get("tool_calls")
            if tool_calls_delta:
                for tool_calls_part in tool_calls_delta:
                    result.tool_calls.append(tool_calls_part)

            thinking_delta = delta.get("reasoning_content")
            if thinking_delta:
                result.thinking += thinking_delta
            content_delta = delta.get("content")
            if content_delta:
                result.content += content_delta

        return result

    def _merge_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        merged = []
        for tool_call in tool_calls:
            id = tool_call.get("id")
            if id:
                merged.append(tool_call)
            else:
                if tool_call.get("function"):
                    if tool_call["function"].get("name"):
                        merged[-1]["function"]["name"] += tool_call["function"]["name"]
                    if tool_call["function"].get("arguments"):
                        merged[-1]["function"]["arguments"] += tool_call["function"][
                            "arguments"
                        ]
        return merged

    def _build_headers(self):
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
