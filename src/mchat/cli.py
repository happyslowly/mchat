from mchat.chat import Chat

if __name__ == "__main__":
    import asyncio

    try:
        chat = Chat()
        asyncio.run(chat.start())
    except Exception as e:
        print(f"{e}")
        exit(1)
