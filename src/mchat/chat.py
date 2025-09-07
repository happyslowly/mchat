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

from mchat.commands import Commands, create_completer
from mchat.config import Config, load_config
from mchat.llm_client import LLMClient
from mchat.session import ChatSession


class Chat:
    def __init__(self, console: Console, config: Config | None = None):
        self._config = config or load_config()
        self._llm_client = LLMClient(
            self._config.base_url,
            api_key=self._config.api_key,
            timeout=self._config.timeout,
        )
        self._chat_session = ChatSession(model=self._config.model)
        self._console = console

        self._summary_task = None
        self._save_task = None

        self._prompt_session = PromptSession()
        self._commands = Commands(
            console=self._console,
            llm_client=self._llm_client,
            chat_session=self._chat_session,
            prompt_session=self._prompt_session,
            tasks=[],
        )
        self._setup_prompt_session()

    async def start(self):
        self._save_task = asyncio.create_task(
            _set_interval(self._chat_session.save, self._config.save_interval)
        )
        self._commands._tasks.append(self._save_task)
        while True:
            try:
                user_input = await self._prompt_session.prompt_async("» ")
            except (EOFError, KeyboardInterrupt):
                await self._commands._quit_command()
                exit(0)

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                await self._commands.execute(user_input)
                continue

            await self._chat_completion_stream(user_input)
            self._commands._tasks.append(self._create_summary_task())

    def _build_messages(self, prompt: str) -> list[dict]:
        messages = []
        system_prompt = self._chat_session.system_prompt or ""
        if self._chat_session.summary:
            system_prompt += (
                f"\n\nPrevious conversation summary: {self._chat_session.summary}"
            )
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            self._chat_session.history[self._chat_session.last_summarized_index + 1 :]
        )
        messages.append({"role": "user", "content": prompt})
        return messages

    def _format_tool_call_data(self, data: str) -> str:
        tool_call = json.loads(data)
        fn_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        args_str = ", ".join(
            (f"{k}={repr(v)[:30]}..." if len(repr(v)) > 30 else f"{k}={repr(v)}")
            for k, v in args.items()
        )
        return f"🔧 {fn_name}({args_str})"

    def _format_tool_complete_data(self, data: str) -> str:
        return f"✅ {data}"

    def _build_display_panels(
        self,
        tool_contents: list[str],
        thinking_contents: list[str],
        contents: list[str],
    ) -> list:
        panels = []
        if tool_contents:
            panels.append(
                Panel(
                    "\n".join(tool_contents),
                    title="🧰",
                    title_align="right",
                    style="dim",
                )
            )
        if thinking_contents:
            panels.append(
                Panel(
                    "".join(thinking_contents),
                    title="🤔",
                    title_align="right",
                    style="dim italic",
                )
            )
        if contents:
            panels.append(
                Panel(Markdown("".join(contents)), title="📝", title_align="right")
            )
        return panels

    async def _chat_completion_stream(self, prompt: str):
        messages = self._build_messages(prompt)

        tool_contents = []
        thinking_contents = []
        contents = []
        loading_spinner = Spinner("dots")

        with Live(loading_spinner, console=self._console) as live:
            try:
                waiting = True
                async for event in self._llm_client.stream_completion(
                    self._chat_session._model, messages
                ):

                    if event.type == "error":
                        raise RuntimeError(event.data)

                    if event.type == "thinking":
                        thinking_contents.append(event.data)
                    elif event.type == "content":
                        contents.append(event.data)
                    elif event.type == "tool_call":
                        tool_contents.append(self._format_tool_call_data(event.data))
                    elif event.type == "tool_complete":
                        tool_contents.append(
                            self._format_tool_complete_data(event.data)
                        )

                    panels = self._build_display_panels(
                        tool_contents, thinking_contents, contents
                    )
                    if panels:
                        live.update(Group(*panels))
                        waiting = False
                    elif waiting:
                        live.update(loading_spinner)

            except Exception as e:
                live.update(Text(text=str(e), style="red"))

        if contents:
            self._chat_session.add_to_history({"role": "user", "content": prompt})
            self._chat_session.add_to_history(
                {"role": "assistant", "content": "".join(contents)}
            )

    async def _summarize(self):
        await self._chat_session.create_summary(
            self._llm_client,
            self._chat_session._model,
            self._config.max_history_turns,
        )

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
        return self._summary_task

    def _setup_prompt_session(self):
        style = Style.from_dict(
            {
                "completion-menu.completion": "fg:default bg:default",
                "completion-menu.completion.current": "bold",
            }
        )
        self._prompt_session.completer = create_completer(self._commands)
        self._prompt_session.style = style


async def _set_interval(func, interval):
    while True:
        await func()
        await asyncio.sleep(interval)
