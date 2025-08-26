import asyncio

from rich.console import Console

from mchat.chat import Chat


def main():
    console = Console()
    try:
        chat = Chat(console)
        asyncio.run(chat.start())
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        exit(1)


if __name__ == "__main__":
    main()
