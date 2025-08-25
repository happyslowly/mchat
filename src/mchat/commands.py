import asyncio
from typing import Awaitable, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.enums import EditingMode
from rich.console import Console

from mchat.config import config_manager
from mchat.llm_client import LLMClient
from mchat.session import ChatSession

CommandHandler = Callable[[Console, list[str]], Awaitable[None]]

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


async def quit_command(console: Console, args: list[str]):
    _ = console, args
    if _chat_session:
        await _chat_session.save()
    if _save_task and not _save_task.done():
        _save_task.cancel()
    if _summary_task and not _summary_task.done():
        _summary_task.cancel()
    exit(0)


async def help_command(console: Console, args: list[str]):
    _ = args
    console.print("Available Commands:", style="dim")
    commands = get_commands()
    max_width = max(len(cmd) for cmd in commands) + 1
    for cmd in commands:
        cmd_text = f"/{cmd}".ljust(max_width + 2)
        console.print(f"  {cmd_text} {commands[cmd][1]}", style="dim")
    console.print()


async def models_command(console: Console, args: list[str]):
    _ = args
    config = config_manager.config
    llm_client = LLMClient(config.base_url, config.api_key)
    model_list = llm_client.list_models()
    for m in model_list:
        console.print(f"*{m}" if m == config.model else f" {m}", style="dim")
        console.print()


async def switch_model_command(console: Console, args: list[str]):
    if not args:
        return
    model_name = args[0]
    config = config_manager.config
    llm_client = LLMClient(config.base_url, config.api_key)
    model_list = llm_client.list_models()
    if model_name not in model_list:
        console.print(f"Model `{model_name}` not found!", style="red")
    else:
        config.model = model_name


async def system_command(console: Console, args: list[str]):
    if not _chat_session:
        return
    if not args:
        console.print("System Prompt:", style="dim")
        console.print(_chat_session.system_prompt, style="dim")
    elif len(args) == 1 and args[0] in ("''", '""'):
        _chat_session.system_prompt = ""
    else:
        _chat_session.system_prompt = " ".join(args)


async def clear_history_command(console: Console, args: list[str]):
    _ = console, args
    if _chat_session:
        _chat_session.clear()


async def show_history_command(console: Console, args: list[str]):
    _ = args
    if _chat_session:
        console.print("Conversation history:", style="dim")
        for message in _chat_session.history:
            console.print(f"{message['role']}:{message['content']}", style="dim")


async def search_command(console: Console, args: list[str]):
    _ = args
    if _chat_session:
        if _chat_session.search:
            _chat_session.search = False
            console.print("Search is turned off", style="dim")
        else:
            _chat_session.search = True
            console.print("Search is turned on", style="dim")


async def edit_mode_command(console: Console, args: list[str]):
    if not _prompt_session:
        return
    if not args:
        current = "vi" if _prompt_session.editing_mode == EditingMode.VI else "emacs"
        console.print(f"Current editing mode: {current}", style="dim")
        return
    mode = args[0]
    if mode.lower() == "vi":
        _prompt_session.editing_mode = EditingMode.VI
    elif mode.lower() == "emacs":
        _prompt_session.editing_mode = EditingMode.EMACS
    else:
        console.print("Invalid mode. Use 'vi' or 'emacs'", style="red")


def get_commands() -> dict[str, tuple[CommandHandler, str]]:

    return {
        "quit": (quit_command, "Exit the chat application"),
        "help": (help_command, "Show available commands"),
        "system": (system_command, "View or set system prompt"),
        "models": (models_command, "List available models (* = current)"),
        "model": (switch_model_command, "Switch to specified model"),
        "clear_history": (clear_history_command, "Clear conversation history"),
        "show_history": (show_history_command, "Print conversation history"),
        "edit_mode": (edit_mode_command, "Switch between vi/emacs editing mode"),
        "search": (search_command, "Toggle internet search on and off"),
    }


class CommandProcessor:
    def __init__(self, console: Console):
        self._console = console
        self._commands = get_commands()

    async def execute(self, command_line: str) -> bool:
        if not command_line.startswith("/"):
            return False

        parts = command_line[1:].split()
        cmd_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        if cmd_name in self._commands:
            handler, _ = self._commands[cmd_name]
            try:
                await handler(self._console, args)
            except Exception as e:
                self._console.print(f"Command {cmd_name} failed: {e}", style="red")
            return True

        self._console.print(f"Unknown command: /{cmd_name}", style="red")
        return True


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
