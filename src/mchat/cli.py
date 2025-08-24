import asyncio

from mchat.chat import Chat


def main():
    try:
        chat = Chat()
        asyncio.run(chat.start())
    except Exception as e:
        print(f"{e}")
        exit(1)


if __name__ == "__main__":
    main()
