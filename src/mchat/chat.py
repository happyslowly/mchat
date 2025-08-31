import asyncio
import json

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from mchat.commands import (
    CommandProcessor,
    create_completer,
    quit_command,
    set_chat_context,
)
from mchat.config import config_manager
from mchat.llm_client import EventType, LLMClient
from mchat.session import ChatSession


class Chat:
    def __init__(self, console: Console):
        self._config = config_manager.config
        self._llm_client = LLMClient(
            self._config.base_url,
            api_key=self._config.api_key,
            timeout=self._config.timeout,
        )
        self._chat_session = ChatSession()
        self._console = console
        self._command_processor = CommandProcessor(self._console)
        self._prompt_session = self._get_prompt_session()

        self._summary_task = None
        self._save_task = None
        self._last_summarized_index = -1

    async def start(self):
        self._save_task = asyncio.create_task(
            _set_interval(self._chat_session.save, self._config.save_interval)
        )
        set_chat_context(
            self._chat_session,
            self._prompt_session,
            self._save_task,
            self._summary_task,
        )
        while True:
            try:
                user_input = await self._prompt_session.prompt_async("» ")
            except (EOFError, KeyboardInterrupt):
                await quit_command()
                exit(0)

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                await self._command_processor.execute(user_input)
                continue

            await self._chat_completion_stream(user_input)
            self._create_summary_task()

    def _build_messages(self, prompt: str) -> list[dict]:
        messages = []
        system_prompt = self._chat_session.system_prompt or ""
        if self._chat_session.summary:
            system_prompt += (
                f"\n\nPrevious conversation summary: {self._chat_session.summary}"
            )
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._chat_session.history[self._last_summarized_index + 1 :])
        messages.append({"role": "user", "content": prompt})
        return messages

    def _process_stream_event(
        self,
        event_type: EventType,
        data: str,
        thinking_content: str,
        content: str,
        tool_content: str,
    ) -> tuple[str, str, str]:
        if event_type == "thinking":
            thinking_content = data
        elif event_type == "content":
            content = data
        elif event_type == "tool_call":
            try:
                tool_call = json.loads(data)
                fn_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                args_str = ", ".join(
                    (
                        f"{k}={repr(v)[:30]}..."
                        if len(repr(v)) > 30
                        else f"{k}={repr(v)}"
                    )
                    for k, v in args.items()
                )
                tool_content += f"🔧 {fn_name}({args_str})\n"
            except (json.JSONDecodeError, KeyError):
                tool_content += f"🔧 Tool call: {data}\n"
        elif event_type == "tool_complete":
            tool_content += f"✅ {data}\n"

        return thinking_content, content, tool_content

    def _build_display_panels(
        self, tool_content: str, thinking_content: str, content: str
    ) -> list:
        panels = []
        if tool_content:
            panels.append(
                Panel(
                    tool_content.strip(),
                    title="🧰",
                    title_align="right",
                    style="dim",
                )
            )
        if thinking_content:
            panels.append(
                Panel(
                    thinking_content,
                    title="🤔",
                    title_align="right",
                    style="dim italic",
                )
            )
        if content:
            panels.append(Panel(Markdown(content), title="📝", title_align="right"))
        return panels

    async def _chat_completion_stream(self, prompt: str):
        messages = self._build_messages(prompt)

        content = ""
        thinking_content = ""
        tool_content = ""

        loading_spinner = Spinner("dots")
        with Live(loading_spinner, console=self._console) as live:
            try:
                waiting = True
                async for event_type, data in self._llm_client.stream_completion(
                    self._chat_session._model, messages
                ):
                    thinking_content, content, tool_content = (
                        self._process_stream_event(
                            event_type,
                            data,
                            thinking_content,
                            content,
                            tool_content,
                        )
                    )

                    panels = self._build_display_panels(
                        tool_content, thinking_content, content
                    )
                    if panels:
                        live.update(Group(*panels))
                        waiting = False
                    elif waiting:
                        live.update(loading_spinner)

            except Exception as e:
                live.update(Text(text=str(e), style="red"))

        self._chat_session.add_to_history({"role": "user", "content": prompt})
        if content:
            self._chat_session.add_to_history({"role": "assistant", "content": content})

    async def _summarize(self):
        start = self._last_summarized_index + 1
        current_messages = self._chat_session.history.copy()
        messages_to_summarize = (
            current_messages[start:]
            if self._config.max_history_turns == -1
            else current_messages[start : -self._config.max_history_turns * 2]
        )
        if not messages_to_summarize:
            return
        recent_history_text = "\n".join(
            [f"{m['role']}:{m['content']}" for m in messages_to_summarize]
        )
        previous_summary = self._chat_session.summary
        summary_prompt = f"""
Summarize this conversation, incorporating the previous summary if provided.

Previous summary: {previous_summary}

Recent conversation:
{recent_history_text}

Create a concise summary that:
- Incorporates key points from the previous summary
- Adds important new topics and conclusions
- Maintains context needed for future messages

Summary:
"""
        try:
            summary_model = self._config.summary_model or self._config.model
            self._chat_session.summary = await self._llm_client.completion(
                summary_model, [{"role": "user", "content": summary_prompt}]
            )
            self._last_summarized_index += len(messages_to_summarize)
        except Exception as e:
            raise RuntimeError(f"Failed to create conversation summary: {e}")

    def _create_summary_task(self):
        if self._summary_task and self._summary_task.done():
            e = self._summary_task.exception()
            if e:
                self._console.print(
                    Panel.fit(
                        str(e), border_style="red", title="📝", title_align="right"
                    )
                )
            self._summary_task = None

        if not self._summary_task:
            self._summary_task = asyncio.create_task(self._summarize())

    def _get_prompt_session(self):
        style = Style.from_dict(
            {
                "completion-menu.completion": "fg:default bg:default",
                "completion-menu.completion.current": "bold",
            }
        )
        return PromptSession(completer=create_completer(), style=style)


async def _set_interval(func, interval):
    while True:
        await func()
        await asyncio.sleep(interval)
