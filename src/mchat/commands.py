import asyncio
import json
from typing import Awaitable, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.enums import EditingMode
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mchat.config import config_manager
from mchat.llm_client import LLMClient
from mchat.session import ChatSession

CommandHandler = Callable[..., Awaitable[str | Text | None]]

_save_task = None
_summary_task = None
_chat_session = None
_prompt_session = None


def set_chat_context(
    chat_session: ChatSession,
    prompt_session: PromptSession,
    save_task: asyncio.Task | None,
    summary_task: asyncio.Task | None,
):
    global _chat_session, _save_task, _summary_task, _prompt_session
    _chat_session = chat_session

    _prompt_session = prompt_session
    _save_task = save_task
    _summary_task = summary_task


async def quit_command(*args) -> None:
    _ = args
    if _chat_session:
        await _chat_session.save()
    if _save_task and not _save_task.done():
        _save_task.cancel()
    if _summary_task and not _summary_task.done():
        _summary_task.cancel()
    exit(0)


async def help_command(*args) -> str:
    _ = args
    commands = get_commands()
    max_width = max(len(cmd) for cmd in commands) + 1
    lines = []
    for cmd in commands:
        cmd_text = f"/{cmd}".ljust(max_width + 2)
        lines.append(f"{cmd_text} {commands[cmd][1]}")
    return "\n".join(lines)


async def models_command(*args) -> Text:
    _ = args
    config = config_manager.config
    llm_client = LLMClient(
        config.base_url, api_key=config.api_key, timeout=config.timeout
    )
    model_list = llm_client.list_models()
    lines = []
    for m in model_list:
        if _chat_session and m == _chat_session._model:
            lines.append(Text(m, style="green"))
        else:
            lines.append(Text(m))
    return Text("\n").join(lines)


async def switch_model_command(*args):
    if not args:
        return
    model_name = args[0]
    config = config_manager.config
    llm_client = LLMClient(
        config.base_url, api_key=config.api_key, timeout=config.timeout
    )
    model_list = llm_client.list_models()
    if model_name not in model_list:
        raise ValueError(f"Model `{model_name}` not found.")
    elif _chat_session:
        _chat_session._model = model_name


async def system_command(*args) -> str | None:
    if not _chat_session:
        return
    if not args:
        if _chat_session.system_prompt:
            return _chat_session.system_prompt
    elif len(args) == 1 and args[0].lower() == "clear":
        _chat_session.system_prompt = ""
    else:
        _chat_session.system_prompt = " ".join(args)


async def history_command(*args) -> Text | str | None:
    _ = args
    if not _chat_session:
        return

    if args:
        if args[0].lower() == "clear":
            _chat_session.history.clear()
            _chat_session.summary = ""
        elif args[0].lower() == "dump":
            if _chat_session.history:
                with open("history.jsonl", "w") as f:
                    for message in _chat_session.history:
                        f.write(json.dumps(message))
                        f.write("\n")
                return "History dumped to history.jsonl"
    else:
        if _chat_session.history:
            lines = []
            for message in _chat_session.history:
                role = "You" if message["role"] == "user" else "AI"
                content = message["content"]
                lines.append(
                    Text(f"{role}: ", style="magenta").append(
                        Text(content, style="default")
                    )
                )
            return Text("\n").join(lines)


async def edit_mode_command(*args) -> str | None:
    if not _prompt_session:
        return
    if not args:
        current = "vi" if _prompt_session.editing_mode == EditingMode.VI else "emacs"
        return f"{current}"
    mode = args[0]
    if mode.lower() == "vi":
        _prompt_session.editing_mode = EditingMode.VI
    elif mode.lower() == "emacs":
        _prompt_session.editing_mode = EditingMode.EMACS


def get_commands() -> dict[str, tuple[CommandHandler, str]]:
    return {
        "quit": (quit_command, "Exit the chat application"),
        "help": (help_command, "Show available commands"),
        "system": (system_command, "View or set system prompt"),
        "models": (models_command, "List available models"),
        "model": (switch_model_command, "Switch to specified model"),
        "history": (history_command, "View, clear, or dump conversation history"),
        "edit_mode": (edit_mode_command, "Switch between vi/emacs editing mode"),
    }


class CommandProcessor:
    def __init__(self, console: Console):
        self._console = console
        self._commands = get_commands()

    async def execute(self, command_line: str):
        parts = command_line[1:].split()
        cmd_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        if cmd_name in self._commands:
            handler, _ = self._commands[cmd_name]
            try:
                output = await handler(*args)
                if output:
                    self._console.print(
                        Panel.fit(
                            output,
                            title="⚡",
                            title_align="right",
                        )
                    )

            except Exception as e:
                self._console.print(
                    Panel.fit(
                        str(e),
                        border_style="red",
                        title="⚡",
                        title_align="right",
                    )
                )
            return

        self._console.print(
            Panel.fit(
                f"Unknown command: /{cmd_name}",
                border_style="red",
                title="⚡",
                title_align="right",
            )
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


def create_completer():
    return SmartCommandCompleter(list(get_commands().keys()))
