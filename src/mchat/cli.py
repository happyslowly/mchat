import asyncio

from rich.console import Console
from rich.panel import Panel

from mchat.chat import Chat
from mchat.config import load_config


def main():
    console = Console()
    try:
        config = load_config()
        chat = Chat(console, config=config)
        asyncio.run(chat.start())
    except Exception as e:
        console.print(Panel.fit(str(e), border_style="red"))
        exit(1)


if __name__ == "__main__":
    main()
