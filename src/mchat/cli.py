import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from mchat.chat import Chat
from mchat.commands import CommandManager, create_completer
from mchat.config import get_config
from mchat.llm_client import LLMClient
from mchat.session import SessionManager, SessionManagerRepo
from mchat.task import TaskManager


def bootstrap(console: Console) -> Chat:
    config = get_config()
    llm_client = LLMClient(
        base_url=config.base_url, api_key=config.api_key, timeout=config.timeout
    )
    session_manager = SessionManager(
        repo=SessionManagerRepo(),
        default_model=config.model,
        continue_last_session=config.continue_last_session,
    )
    task_manager = TaskManager()
    prompt_session = PromptSession()
    prompt_session.completer = create_completer()
    prompt_session.style = Style.from_dict(
        {
            "completion-menu.completion": "fg:default bg:default",
            "completion-menu.completion.current": "bold",
        }
    )
    command_manager = CommandManager(
        console=console,
        llm_client=llm_client,
        chat_session_manager=session_manager,
        prompt_session=prompt_session,
        task_manager=task_manager,
    )

    chat = Chat(
        config=config,
        console=console,
        llm_client=llm_client,
        session_manager=session_manager,
        task_manager=task_manager,
        command_manager=command_manager,
        prompt_session=prompt_session,
    )

    return chat


def main():
    console = Console()
    try:
        chat = bootstrap(console)
        asyncio.run(chat.start())
    except Exception as e:
        console.print(Panel.fit(str(e), border_style="red"))
        exit(1)


if __name__ == "__main__":
    main()
