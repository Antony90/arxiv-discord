import os
import json

class CONFIG:
    BOT_TOKEN = os.environ["BOT_TOKEN"]
    PAPER_STORE_PATH = "./paper_store.raw"
    THREAD_CHANNELS = json.loads(os.environ["THREAD_CHANNELS"]) # TODO: use a database for guild conversation channels
    
    MAX_THREAD_SEARCH = 30
    CHAT_WINDOW = 5
    MAX_MSG = 2000 # discord char limit

    SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]