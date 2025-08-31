import asyncio

from rich.console import Console
from rich.panel import Panel

from mchat.chat import Chat


def main():
    console = Console()
    try:
        chat = Chat(console)
        asyncio.run(chat.start())
    except Exception as e:
        console.print(Panel.fit(str(e), border_style="red"))
        exit(1)


if __name__ == "__main__":
    main()
