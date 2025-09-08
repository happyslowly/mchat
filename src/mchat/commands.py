import json
from typing import Awaitable, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.enums import EditingMode
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from mchat.llm_client import LLMClient
from mchat.session import ChatSession
from mchat.tasks import TaskManager

CommandHandler = Callable[..., Awaitable[str | Text | Markdown | None]]


class CommandManager:
    def __init__(
        self,
        llm_client: LLMClient,
        console: Console,
        chat_session: ChatSession,
        prompt_session: PromptSession,
        task_manager: TaskManager,
    ):
        self._llm_client = llm_client
        self._console = console
        self._chat_session = chat_session
        self._prompt_session = prompt_session
        self._task_manager = task_manager
        self._commands = self._get_commands()

    def _get_commands(self) -> dict[str, tuple[CommandHandler, str]]:
        return {
            "quit": (self._quit_command, "Exit the chat application"),
            "help": (self._help_command, "Show available commands"),
            "system": (self._system_command, "View or set system prompt"),
            "models": (self._models_command, "List available models"),
            "model": (self._switch_model_command, "Switch to specified model"),
            "history": (
                self._history_command,
                "View, clear, or dump conversation history",
            ),
            "summary": (
                self._summary_command,
                "Create summary of entire conversation history",
            ),
            "clear": (self._clear_command, "Clear current chat session"),
            "edit_mode": (
                self._edit_mode_command,
                "Switch between vi/emacs editing mode",
            ),
        }

    async def _quit_command(self, *args) -> None:
        _ = args
        if self._chat_session:
            await self._chat_session.save()
        self._task_manager.cancel_all()
        exit(0)

    async def _help_command(self, *args) -> str:
        _ = args
        commands = self._get_commands()
        max_width = max(len(cmd) for cmd in commands) + 1
        lines = []
        for cmd in commands:
            cmd_text = f"/{cmd}".ljust(max_width + 2)
            lines.append(f"{cmd_text} {commands[cmd][1]}")
        return "\n".join(lines)

    async def _models_command(self, *args) -> Text:
        _ = args
        model_list = await self._llm_client.list_models()
        lines = []
        for m in model_list:
            if self._chat_session and m == self._chat_session._model:
                lines.append(Text(m, style="green"))
            else:
                lines.append(Text(m))
        return Text("\n").join(lines)

    async def _switch_model_command(self, *args):
        if not args:
            return
        model_name = args[0]
        model_list = await self._llm_client.list_models()
        if model_name not in model_list:
            raise ValueError(f"Model `{model_name}` not found.")
        elif self._chat_session:
            self._chat_session._model = model_name

    async def _system_command(self, *args) -> str:
        if not self._chat_session:
            raise ValueError("Chat session is not available")
        elif len(args) == 1 and args[0].lower() == "clear":
            self._chat_session.system_prompt = ""
        else:
            self._chat_session.system_prompt = " ".join(args)
        return self._chat_session.system_prompt

    async def _history_command(self, *args) -> str | Markdown:
        _ = args
        if not self._chat_session:
            raise ValueError("Chat session is not available")

        if args:
            if args[0].lower() == "clear":
                self._chat_session.history.clear()
                self._chat_session.summary = ""
                return "Conversation history cleared"
            if args[0].lower() == "dump":
                if self._chat_session.history:
                    with open("history.jsonl", "w") as f:
                        for message in self._chat_session.history:
                            f.write(json.dumps(message))
                            f.write("\n")
                    return "Conversation history dumped to history.jsonl"
                raise ValueError("No conversation history to dump")
            raise ValueError("Usage: /history [clear|dump]")
        else:
            if self._chat_session.history:
                lines = []
                for message in self._chat_session.history:
                    content = message["content"]
                    if message["role"] == "user":
                        lines.append(f"***You***: *{content}*  ")
                    else:
                        lines.append(f"**AI**: {content}  ")
                lines.append("")
                return Markdown("\n".join(lines))
            raise ValueError("No conversation history")

    async def _clear_command(self, *args):
        _ = args
        if not self._chat_session:
            return
        self._chat_session.clear()

    async def _edit_mode_command(self, *args) -> str:
        if not self._prompt_session:
            raise ValueError("Prompt session is not available")
        if args:
            mode = args[0]
            if mode.lower() == "vi":
                self._prompt_session.editing_mode = EditingMode.VI
            elif mode.lower() == "emacs":
                self._prompt_session.editing_mode = EditingMode.EMACS
        return "vi" if self._prompt_session.editing_mode == EditingMode.VI else "emacs"

    async def _summary_command(self, *args) -> str:
        _ = args

        if not self._chat_session:
            raise ValueError("Chat session is not available")

        if not self._chat_session.history:
            raise ValueError("No conversation history to summarize")

        await self._chat_session.create_summary(
            self._llm_client,
            self._chat_session._model,
            end_index=len(self._chat_session.history),
        )
        return self._chat_session.summary

    async def execute(self, command_line: str):
        parts = command_line[1:].split()
        cmd_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        if cmd_name not in self._commands:
            self._console.print(
                self._create_panel(f"Unknown command: /{cmd_name}", error=True)
            )
            return

        output, error = None, None
        with Live(Spinner("dots"), transient=True) as _:
            handler, _ = self._commands[cmd_name]
            try:
                output = await handler(*args)
            except Exception as e:
                error = str(e)
        if output:
            self._console.print(self._create_panel(output))
        if error:
            self._console.print(self._create_panel(error, error=True))

    def _create_panel(
        self, content: str | Text | Markdown, error: bool = False
    ) -> Panel:
        return Panel.fit(
            content,
            title="⚡",
            title_align="right",
            border_style="red" if error else "default",
        )


class SmartCommandCompleter(Completer):
    def __init__(self, commands):
        self.commands = commands

    def get_completions(self, document, complete_event):
        _ = complete_event
        text = document.text
        if text.startswith("/") and " " not in text:
            word = text[1:]
            for cmd in self.commands:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))


def create_completer(command: CommandManager):
    return SmartCommandCompleter(list(command._get_commands().keys()))
