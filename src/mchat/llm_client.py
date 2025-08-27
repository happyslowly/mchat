import json
import traceback
from typing import AsyncIterator, Literal

import httpx
from loguru import logger

from mchat.tools import exec_tool_calls, get_tool_schemas

EventType = Literal["thinking", "content", "tool_call", "tool_complete"]


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
    ) -> AsyncIterator[tuple[EventType, str]]:
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "tools": get_tool_schemas(),
        }

        thinking_so_far = ""
        content_so_far = ""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            is_done = False
            while True:
                tool_calls = []
                try:
                    async with client.stream(
                        "POST",
                        self._base_url + "/chat/completions",
                        json=body,
                        headers=self._build_headers(),
                    ) as response:
                        response.raise_for_status()
                        async for data in response.aiter_bytes():
                            (
                                tool_calls,
                                thinking_so_far,
                                content_so_far,
                                is_done,
                            ) = self._process_chunk(
                                data,
                                tool_calls,
                                thinking_so_far,
                                content_so_far,
                            )
                            if thinking_so_far:
                                yield "thinking", thinking_so_far
                            if content_so_far:
                                yield "content", content_so_far
                            if is_done:
                                return

                    if tool_calls:
                        body["messages"].append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls,
                            }
                        )
                        for tool_call in tool_calls:
                            yield "tool_call", json.dumps(tool_call)
                        tool_messages = await exec_tool_calls(tool_calls)
                        yield "tool_complete", f"{len(tool_calls)} tools executed"
                        body["messages"].extend(tool_messages)
                except httpx.TimeoutException as _:
                    raise TimeoutError("API call timed out")
                except Exception as e:
                    traceback.print_exc()
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
        self, data: bytes, tool_calls: list[dict], thinking: str, content: str
    ) -> tuple[list[dict], str, str, bool]:
        for chunk in data.decode().split("\n"):
            if not chunk:
                continue
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
                return tool_calls, thinking, content, True

            delta = choice["delta"]
            if not delta:
                continue

            tool_calls_delta = delta.get("tool_calls")
            if tool_calls_delta:
                for tool_calls_part in tool_calls_delta:
                    index = tool_calls_part["index"]
                    if index >= len(tool_calls):
                        tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    current_tool = tool_calls[index]
                    if tool_calls_part.get("id"):
                        current_tool["id"] = tool_calls_part["id"]
                    if tool_calls_part.get("function"):
                        if tool_calls_part["function"].get("name"):
                            current_tool["function"]["name"] = tool_calls_part[
                                "function"
                            ]["name"]
                        if tool_calls_part["function"].get("arguments"):
                            current_tool["function"]["arguments"] += tool_calls_part[
                                "function"
                            ]["arguments"]

            thinking_delta = delta.get("reasoning_content")
            if thinking_delta:
                thinking += thinking_delta
            content_delta = delta.get("content")
            if content_delta:
                content += content_delta

        return tool_calls, thinking, content, False

    def _build_headers(self):
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
