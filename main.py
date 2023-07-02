__import__("dotenv").load_dotenv()

import asyncio
import logging
import discord
from discord.ext.commands import bot

from langchain.memory.chat_message_histories import ChatMessageHistory

from bot import ArxivBot
from config import CONFIG



handler = logging.FileHandler(filename='logs/bot.log', encoding='utf-8', mode='w')

intents = discord.Intents.default()
intents.message_content = True

bot = ArxivBot()
bot.run(
    token=CONFIG.BOT_TOKEN, 
    log_handler=handler, 
    log_level=logging.DEBUG
)


async def main_test():
    from ai.agent import ArxivAgent
    
    agent = ArxivAgent(verbose=True)
    chat_id = "test"
    message_history = ChatMessageHistory(messages=[])
    
    while True:
        ai_response = await agent.acall(
            input=input(">>> "),
            chat_id=chat_id,
            chat_history=message_history
        )
        message_history.add_ai_message(ai_response)

