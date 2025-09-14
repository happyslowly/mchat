import getpass
import json
import os
import platform
import socket
from datetime import date, datetime

from prompt_toolkit import PromptSession
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from mchat.commands import CommandManager
from mchat.config import Config
from mchat.llm_client import LLMClient
from mchat.session import SessionManager
from mchat.task import TaskManager


class Chat:
    def __init__(
        self,
        config: Config,
        console: Console,
        llm_client: LLMClient,
        session_manager: SessionManager,
        task_manager: TaskManager,
        command_manager: CommandManager,
        prompt_session: PromptSession,
    ):
        self._config = config
        self._console = console
        self._llm_client = llm_client
        self._session_manager = session_manager
        self._task_manager = task_manager
        self._prompt_session = prompt_session
        self._command_manager = command_manager

        self._summary_task = None
        self._save_task = None

    async def start(self):
        while True:
            try:
                user_input = await self._prompt_session.prompt_async("» ")
            except (EOFError, KeyboardInterrupt):
                await self._command_manager.quit()
                exit(0)

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                await self._command_manager.execute(user_input)
                continue

            await self._chat_completion_stream(user_input)
            self._task_manager.create_task(self._summarize, exclusive=True)
            if self._gen_title not in self._task_manager:
                self._task_manager.create_task(self._gen_title, interval=60)

    def _build_messages(self, prompt: str) -> list[dict]:
        messages = []
        session = self._session_manager.current_session
        system_prompt = session.system_prompt.format(**self._get_variables()) or ""
        if session.summary:
            system_prompt += f"\n\nPrevious conversation summary: {session.summary}"
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(session.history[session.last_summarized_index + 1 :])
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
        session = self._session_manager.current_session
        messages = self._build_messages(prompt)

        tool_contents = []
        thinking_contents = []
        contents = []
        loading_spinner = Spinner("dots")

        with Live(loading_spinner, console=self._console) as live:
            try:
                waiting = True
                async for event in self._llm_client.stream_completion(
                    session.model, messages
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
            self._session_manager.add_to_history({"role": "user", "content": prompt})
            self._session_manager.add_to_history(
                {"role": "assistant", "content": "".join(contents)}
            )

    async def _summarize(self):
        session = self._session_manager.current_session
        await self._session_manager.create_summary(
            self._llm_client,
            summary_model=self._config.summary_model or session.model,
            max_history_turns=self._config.max_history_turns,
        )

    async def _gen_title(self):
        session = self._session_manager.current_session
        await self._session_manager.generate_title(
            self._llm_client,
            summary_model=self._config.summary_model or session.model,
        )

    def _get_variables(self):
        now = datetime.now()
        return {
            "current_timestamp": now.isoformat(),
            "current_date": date.today().isoformat(),
            "current_time": now.strftime("%H:%M:%S"),
            "username": getpass.getuser(),
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "cwd": os.getcwd(),
        }
