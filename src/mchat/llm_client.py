import json
from dataclasses import dataclass
from typing import AsyncIterator, Literal, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from mchat.tools import exec_tool_calls, get_tool_schemas

EventType = Literal["thinking", "content", "tool_call", "tool_complete", "error"]


@dataclass
class StreamEvent:
    type: EventType
    data: str


class LLMClient:
    def __init__(self, base_url: str, timeout: int, api_key: str | None = None):
        self._openai = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "dummy-key", timeout=float(timeout)
        )

    async def list_models(self) -> list[str]:
        models = await self._openai.models.list()
        return [m.id for m in models.data]

    async def stream_completion(
        self, model: str, messages: list[dict], max_tool_rounds: int = 8
    ) -> AsyncIterator[StreamEvent]:
        local_messages = messages.copy()
        rounds = 0
        while True:
            try:
                if rounds >= max_tool_rounds:
                    yield StreamEvent(
                        "error",
                        f"Maximum tool rounds reached ({max_tool_rounds})",
                    )
                    return

                response = await self._openai.chat.completions.create(
                    model=model,
                    messages=cast(list[ChatCompletionMessageParam], local_messages),
                    tools=get_tool_schemas(),
                    tool_choice="auto",
                    stream=True,
                )
                tool_calls = []
                async for chunk in response:
                    if not chunk.choices:
                        break

                    choice = chunk.choices[0]
                    delta = choice.delta

                    if choice.finish_reason in (
                        "stop",
                        "length",
                        "content_filter",
                    ):
                        if choice.finish_reason == "stop":
                            return
                        detail = ""
                        refusal_text = getattr(delta, "refusal", None)
                        if refusal_text:
                            detail = f": {refusal_text}"
                        else:
                            cfr = getattr(choice, "content_filter_results", None)
                            if cfr:
                                detail = f": {json.dumps(cfr)}"
                        yield StreamEvent(
                            "error",
                            f"Generation stopped ({choice.finish_reason}){detail}",
                        )
                        return

                    if getattr(delta, "reasoning_content", None):
                        yield StreamEvent(
                            "thinking", getattr(delta, "reasoning_content")
                        )

                    if delta.content:
                        yield StreamEvent("content", delta.content)

                    if delta.tool_calls:
                        tool_calls.extend(delta.tool_calls)

                if tool_calls:
                    tool_calls_complete = self._merge_tool_calls(tool_calls)
                    if tool_calls_complete:
                        local_messages.append(
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
                        local_messages.extend(tool_messages)
                        rounds += 1
                else:
                    return
            except Exception as e:
                yield StreamEvent("error", str(e))
                return

    async def completion(self, model: str, messages: list[dict]) -> str:
        response = await self._openai.chat.completions.create(
            model=model, messages=cast(list[ChatCompletionMessageParam], messages)
        )
        return response.choices[0].message.content or ""

    def _merge_tool_calls(self, tool_calls: list[ChoiceDeltaToolCall]) -> list[dict]:
        merged = []
        for tool_call in tool_calls:
            idx = tool_call.index
            while idx >= len(merged):
                merged.append(
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            current = merged[idx]
            if tool_call.id:
                current["id"] = tool_call.id
            if tool_call.function:
                if tool_call.function.name:
                    current["function"]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    current["function"]["arguments"] += tool_call.function.arguments

        return [m for m in merged if m.get("id")]
