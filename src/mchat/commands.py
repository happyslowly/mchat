import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

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
from mchat.session import SessionManager
from mchat.task import TaskManager

CommandHandler = Callable[..., Awaitable[str | Text | Markdown | None]]
CommandCategory = Literal["General", "Session", "History", "Model", "System Prompt"]


@dataclass
class CommandInfo:
    handler: CommandHandler
    description: str
    usage: str
    category: CommandCategory


_command_registry: dict[str, CommandInfo] = {}


def command(name: str, description: str, usage: str, category: CommandCategory):
    def decorator(func: CommandHandler):
        _command_registry[name] = CommandInfo(
            handler=func, description=description, usage=usage, category=category
        )
        return func

    return decorator


class CommandManager:
    def __init__(
        self,
        llm_client: LLMClient,
        console: Console,
        chat_session_manager: SessionManager,
        prompt_session: PromptSession,
        task_manager: TaskManager,
    ):
        self._llm_client = llm_client
        self._console = console
        self._chat_session_manager = chat_session_manager
        self._prompt_session = prompt_session
        self._task_manager = task_manager

    @command("quit", "Exit application", "/quit", "General")
    async def quit(self, *args) -> None:
        _ = args
        self._chat_session_manager.close()
        self._task_manager.cancel_all()
        exit(0)

    @command("help", "Show this help", "/help", "General")
    async def help(self, *args) -> Text:
        _ = args

        categories = {}
        for cmd_name, cmd_info in _command_registry.items():
            if cmd_info.category not in categories:
                categories[cmd_info.category] = []
            categories[cmd_info.category].append((cmd_name, cmd_info))

        max_usage_width = max(
            len(cmd_info.usage) for cmd_info in _command_registry.values()
        )

        lines = []
        category_order: list[CommandCategory] = [
            "General",
            "Model",
            "System Prompt",
            "Session",
            "History",
        ]

        for category in category_order:
            if category in categories:
                lines.append(Text(f"{category}", style="bold"))
                for _, cmd_info in sorted(categories[category]):
                    usage_text = cmd_info.usage.ljust(max_usage_width)
                    lines.append(
                        Text(f"  {usage_text}  {cmd_info.description}", style="dim")
                    )

        return Text("\n").join(lines)

    @command("models", "List models", "/models", "Model")
    async def list_models(self, *args) -> Text:
        _ = args
        model_list = await self._llm_client.list_models()
        lines = []
        for m in model_list:
            if m == self._chat_session_manager.current_session.model:
                lines.append(Text(m, style="green"))
            else:
                lines.append(Text(m))
        return Text("\n").join(lines)

    @command("use", "Switch model", "/use <model_name>", "Model")
    async def use_model(self, *args):
        if not args:
            return
        model_name = args[0]
        model_list = await self._llm_client.list_models()
        if model_name not in model_list:
            raise ValueError(f"Model `{model_name}` not found.")
        self._chat_session_manager.set_model(model_name)

    @command("sessions", "List sessions", "/sessions", "Session")
    async def list_sessions(self, *args) -> Text:
        def format(data: dict) -> str:
            title = data["title"].ljust(max_width + 2)
            return f"{data['id']}: {title} ({data['updated_at']})"

        _ = args
        session_list = self._chat_session_manager.list_sessions()

        max_width = max(len(s["title"]) for s in session_list) + 1
        lines = []
        for s in session_list:
            if s["id"] == self._chat_session_manager.current_session.id:
                lines.append(Text(format(s), style="green"))
            else:
                lines.append(Text(format(s)))
        return Text("\n").join(lines)

    @command("system", "Show system prompt", "/system", "System Prompt")
    async def show_system(self, *args) -> str:
        _ = args
        return (
            self._chat_session_manager.current_session.system_prompt
            or "(No system prompt set)"
        )

    @command("set-system", "Set system prompt", "/set-system [prompt]", "System Prompt")
    async def set_system(self, *args) -> str:
        if not args:
            self._chat_session_manager.set_system_prompt("")
            return "System prompt cleared"
        else:
            self._chat_session_manager.set_system_prompt(" ".join(args))
            return f"System prompt set to: {self._chat_session_manager.current_session.system_prompt}"

    @command("history", "Show history", "/history", "History")
    async def show_history(self, *args) -> Markdown:
        _ = args
        if self._chat_session_manager.current_session.history:
            lines = []
            for message in self._chat_session_manager.current_session.history:
                content = message["content"]
                if message["role"] == "user":
                    lines.append(f"***You***: *{content}*  ")
                else:
                    lines.append(f"**AI**: {content}  ")
            lines.append("")
            return Markdown("\n".join(lines))
        raise ValueError("No conversation history")

    @command("mode", "Set editing mode", "/mode [vi|emacs]", "General")
    async def set_mode(self, *args) -> str:
        if not self._prompt_session:
            raise ValueError("Prompt session is not available")
        if args:
            mode = args[0]
            if mode.lower() == "vi":
                self._prompt_session.editing_mode = EditingMode.VI
            elif mode.lower() == "emacs":
                self._prompt_session.editing_mode = EditingMode.EMACS
        return "vi" if self._prompt_session.editing_mode == EditingMode.VI else "emacs"

    @command(
        "summary",
        "Create conversation summary",
        "/summary",
        "History",
    )
    async def create_summary(self, *args) -> str:
        _ = args

        if not self._chat_session_manager.current_session.history:
            raise ValueError("No conversation history to summarize")

        await self._chat_session_manager.create_summary(
            self._llm_client,
            self._chat_session_manager.current_session.model,
            end_index=len(self._chat_session_manager.current_session.history),
        )
        return self._chat_session_manager.current_session.summary

    @command("new-session", "Create new session", "/new-session", "Session")
    async def new_session(self, *args) -> str:
        _ = args
        new_session = self._chat_session_manager.create_session()
        return f"Created new session: {new_session.id}"

    @command(
        "switch-session",
        "Switch to session",
        "/switch-session <session_id>",
        "Session",
    )
    async def switch_session(self, *args):
        if not args:
            raise ValueError("Usage: /switch <session_id>")
        self._chat_session_manager.switch_session(int(args[0]))

    @command(
        "delete-session",
        "Delete session",
        "/delete-session <session_id>",
        "Session",
    )
    async def delete_session(self, *args) -> str:
        if not args:
            raise ValueError("Usage: /rm <session_id>")
        session_id = int(args[0])
        session_list = self._chat_session_manager.list_sessions()
        session_to_delete = next(
            (s for s in session_list if s["id"] == session_id), None
        )
        if not session_to_delete:
            raise ValueError(f"Session {session_id} not found")
        self._chat_session_manager.delete_session(session_id)
        return f"Deleted session: {session_to_delete['title']}"

    @command("clear-history", "Clear history", "/clear-history", "History")
    async def clear_history(self, *args) -> str:
        _ = args
        self._chat_session_manager.clear_history()
        return "Conversation history cleared"

    @command("export", "Export history to file", "/export [filename]", "History")
    async def export_history(self, *args) -> str:
        filename = args[0] if args else "history.jsonl"
        if self._chat_session_manager.current_session.history:
            with open(filename, "w") as f:
                for message in self._chat_session_manager.current_session.history:
                    f.write(json.dumps(message))
                    f.write("\n")
            return f"Conversation history exported to {filename}"
        raise ValueError("No conversation history to export")

    @command("reset", "Reset session", "/reset", "Session")
    async def reset_session(self, *args) -> str:
        _ = args
        self._chat_session_manager.clear_session()
        self._chat_session_manager.set_system_prompt("")
        return "Session reset (history and system prompt cleared)"

    async def execute(self, command_line: str):
        parts = command_line[1:].split()
        cmd_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        if cmd_name not in _command_registry:
            self._console.print(
                self._create_panel(f"Unknown command: /{cmd_name}", error=True)
            )
            return

        output, error = None, None
        with Live(Spinner("dots"), transient=True) as _:
            command_info = _command_registry[cmd_name]
            try:
                output = await command_info.handler(self, *args)
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


def create_completer():
    return SmartCommandCompleter(list(_command_registry.keys()))
