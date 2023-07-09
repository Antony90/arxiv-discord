import os

class CONFIG:
    BOT_TOKEN = os.environ["BOT_TOKEN"]
    PAPER_STORE_PATH = "./paper_store.raw"
    THREAD_CHANNEL = os.environ["THREAD_CHANNEL"]

    MAX_THREAD_SEARCH = 30
    CHAT_WINDOW = 3
    MAX_MSG = 2000 # discord char limit