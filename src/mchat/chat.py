import asyncio
import json
import os
import tomllib
from pathlib import Path

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.enums import EditingMode
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from mchat.history import HistoryManager
from mchat.schemas import Config, Message


class Chat:
    def __init__(self):
        self._config = self._load_config()
        self._hm = HistoryManager()
        self._console = Console()
        self._model_list = self._get_model_list()
        self._commands = {
            "quit": (self._quit, "Exit the chat application"),
            "help": (self._help, "Show available commands"),
            "system": (self._system, "View or set system prompt"),
            "models": (self._models, "List available models (* = current)"),
            "model": (self._model, "Switch to specified model"),
            "clear_history": (self._clear_history, "Clear conversation history"),
            "show_history": (self._show_history, "Print conversation history"),
            "edit_mode": (self._edit_mode, "Switch between vi/emacs editing mode"),
        }
        self._system_prompt = ""
        self._session = PromptSession()

    async def start(self):
        # save history every 5 minutes
        self._history_task = asyncio.create_task(_set_interval(self._hm.save, 60 * 5))
        while True:
            try:
                user_input = await self._session.prompt_async("> ")
            except (EOFError, KeyboardInterrupt):
                await self._quit()

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                parts = user_input[1:].split()
                cmd_name = parts[0]
                command = self._commands.get(cmd_name)
                if command:
                    fn = command[0]
                    args = parts[1:] if len(parts) > 1 else []
                    try:
                        await fn(*args)
                    except Exception as e:
                        self._console.print(
                            f"Execute {cmd_name} failed: {e}", style="red"
                        )
                    continue

            await self._chat_completion_stream(user_input)

    async def _chat_completion_stream(self, prompt: str):
        messages = []
        if self._system_prompt:
            messages.append(Message(role="system", content=self._system_prompt))
            self._hm.update_system_prompt(self._system_prompt)
        if self._hm.history.messages:
            messages.extend(self._hm.history.messages)
        current_message = Message(role="user", content=prompt)
        messages.append(current_message)

        body = {
            "model": self._config.model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
        }

        thinking = ""
        content = ""
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                async with client.stream(
                    "POST",
                    self._config.base_url + "/chat/completions",
                    json=body,
                    headers=self._build_headers(),
                ) as response:
                    with Live("", console=self._console) as live:
                        async for data in response.aiter_bytes():
                            thinking, content = self._process(data, thinking, content)
                            content_panel = Panel(
                                Markdown(content), title="📝", title_align="right"
                            )
                            if thinking:
                                thinking_panel = Panel(
                                    thinking,
                                    title="🤔",
                                    title_align="right",
                                    style="dim italic",
                                )
                                display = Group(thinking_panel, content_panel)
                            else:
                                display = content_panel
                            live.update(display)
            except httpx.TimeoutException as _:
                self._console.print(
                    f"API call times out, please try again", style="yellow"
                )
            except Exception as e:
                self._console.print(f"API error: {e}", style="red")
        self._hm.add(current_message)
        if content:
            self._hm.add(Message(role="assistant", content=content))

    def _process(self, data: bytes, thinking: str, content: str) -> tuple[str, str]:
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
                self._console.print(f"Failed to parse chunk: `{chunk}`", style="red")
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
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    @staticmethod
    def _load_config() -> Config:
        if "XDG_CONFIG_HOME" in os.environ:
            config_dir = Path(os.environ["XDG_CONFIG_HOME"])
        else:
            config_dir = Path.home() / ".config"
        config_file = config_dir / "mchat" / "config.toml"
        try:
            with open(config_file, "rb") as f:
                config = tomllib.load(f)
            return Config(
                base_url=config["base_url"],
                model=config["model"],
                api_key=config.get("api_key"),
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_file}")

    async def _quit(self):
        await self._hm.save()
        if self._history_task:
            self._history_task.cancel()
        exit(0)

    async def _help(self):
        self._console.print("Available Commands:", style="dim")
        max_width = max(len(cmd) for cmd in self._commands) + 1
        for cmd in self._commands:
            cmd_text = f"/{cmd}".ljust(max_width + 2)
            self._console.print(f"  {cmd_text} {self._commands[cmd][1]}", style="dim")
        self._console.print()

    async def _models(self):
        for m in self._model_list:
            self._console.print(
                f"*{m}" if m == self._config.model else f" {m}", style="dim"
            )
        self._console.print()

    async def _model(self, model_name: str):
        if model_name not in self._model_list:
            self._console.print(f"Model `{model_name}` not found!", style="red")
        else:
            self._config.model = model_name

    async def _system(self, *args):
        if not args:
            self._console.print("System Prompt:", style="dim")
            self._console.print(self._system_prompt, style="dim")
        if len(args) == 1 and args[0] in ("''", '""'):
            self._system_prompt = ""
        else:
            self._system_prompt = " ".join(args)

    async def _clear_history(self):
        self._hm.clear()

    async def _show_history(self):
        self._console.print("Conversation history:", style="dim")
        for message in self._hm.history.messages:
            self._console.print(f"{message.role}:{message.content}", style="dim")

    async def _edit_mode(self, mode: str | None = None):
        if not mode:
            current = "vi" if self._session.editing_mode == EditingMode.VI else "emacs"
            self._console.print(f"Current editing mode: {current}", style="dim")
            return
        if mode.lower() == "vi":
            self._session.editing_mode = EditingMode.VI
        elif mode.lower() == "emacs":
            self._session.editing_mode = EditingMode.EMACS
        else:
            self._console.print("Invalid mode. Use 'vi' or 'emacs'", style="red")

    def _get_model_list(self) -> list[str]:
        with httpx.Client() as client:
            response = client.get(self._config.base_url + "/models")
            response.raise_for_status()
            data = response.json()["data"]
            models = []
            for entry in data:
                if entry["object"] == "model":
                    models.append(entry["id"])
            return models


async def _set_interval(func, interval):
    while True:
        await func()
        await asyncio.sleep(interval)
