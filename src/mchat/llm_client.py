import json
from typing import AsyncIterator

import httpx
from loguru import logger


class LLMClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout=60):
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout

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
    ) -> AsyncIterator[tuple[str, str]]:
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        thinking_so_far = ""
        content_so_far = ""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    self._base_url + "/chat/completions",
                    json=body,
                    headers=self._build_headers(),
                ) as response:
                    async for data in response.aiter_bytes():
                        thinking_so_far, content_so_far = self._process_chunk(
                            data, thinking_so_far, content_so_far
                        )
                        yield thinking_so_far, content_so_far
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
        self, data: bytes, thinking: str, content: str
    ) -> tuple[str, str]:
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
            delta = choice["delta"]
            if not delta:
                continue
            thinking_delta = delta.get("reasoning_content")
            if thinking_delta:
                thinking += thinking_delta
            content_delta = delta.get("content")
            if content_delta:
                content += content_delta
        return thinking, content

    def _build_headers(self):
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
