import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from mchat.commands import (
    CommandProcessor,
    create_completer,
    quit_command,
    set_chat_context,
)
from mchat.config import config_manager
from mchat.llm_client import LLMClient
from mchat.session import ChatSession


class Chat:
    def __init__(self):
        self._config = config_manager.config
        self._llm_client = LLMClient(self._config.base_url, self._config.api_key)
        self._chat_session = ChatSession()
        self._console = Console()
        self._command_processor = CommandProcessor(self._console)
        self._prompt_session = self._get_prompt_session()

        self._summary_task = None
        self._save_task = None
        self._last_summarized_index = -1

    async def start(self):
        # save history every 5 minutes
        self._save_task = asyncio.create_task(
            _set_interval(self._chat_session.save, 60 * 5)
        )
        set_chat_context(
            self._chat_session,
            self._prompt_session,
            self._save_task,
            self._summary_task,
        )
        while True:
            try:
                user_input = await self._prompt_session.prompt_async("> ")
            except (EOFError, KeyboardInterrupt):
                await quit_command(self._console, [])
                exit(0)

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                if await self._command_processor.execute(user_input):
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

    async def _chat_completion_stream(self, prompt: str):
        messages = self._build_messages(prompt)
        content = ""
        with Live("", console=self._console) as live:
            try:
                async for (
                    thinking_so_far,
                    content_so_far,
                ) in self._llm_client.stream_completion(self._config.model, messages):
                    content_panel = Panel(
                        Markdown(content_so_far), title="📝", title_align="right"
                    )
                    if thinking_so_far:
                        thinking_panel = Panel(
                            thinking_so_far,
                            title="🤔",
                            title_align="right",
                            style="dim italic",
                        )
                        display = Group(thinking_panel, content_panel)
                    else:
                        display = content_panel
                    live.update(display)
                    content = content_so_far
            except Exception as e:
                self._console.print(str(e), style="red")

        self._chat_session.add_to_history({"role": "user", "content": prompt})
        if content:
            self._chat_session.add_to_history({"role": "assistant", "content": content})

    async def _summarize(self):
        start = self._last_summarized_index + 1
        current_messages = self._chat_session.history.copy()
        messages_to_summarize = (
            current_messages[start:]
            if self._config.history_limit == -1
            else current_messages[start : -self._config.history_limit]
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

Create a concise summary (2-3 sentences) that:
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
            self._console.print(
                f"Failed generate conversation summary: {e}", style="red"
            )

    def _create_summary_task(self):
        if self._summary_task and self._summary_task.done():
            if self._summary_task.exception():
                self._console.print("Conversation summary failed", style="yellow")
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
