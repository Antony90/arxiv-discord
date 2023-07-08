__import__("dotenv").load_dotenv()

import asyncio
import logging
import discord

from langchain.memory.chat_message_histories import ChatMessageHistory
from ai.agent import ArxivAgent

from bot import ArxivBot
from config import CONFIG

import argparse


def run_bot(agent: ArxivAgent):
    handler = logging.FileHandler(filename='logs/bot.log', encoding='utf-8', mode='w')

    intents = discord.Intents.default()
    intents.message_content = True

    bot = ArxivBot(agent)
    bot.run(
        token=CONFIG.BOT_TOKEN, 
        log_handler=handler, 
        log_level=logging.DEBUG
    )


async def run_test(agent: ArxivAgent):
    # terminal input test
    
    chat_id = "test"
    message_history = ChatMessageHistory(messages=[])
    
    while True:
        ai_response = await agent.acall(
            input=input(">>> "),
            chat_id=chat_id,
            chat_history=message_history
        )
        message_history.add_ai_message(ai_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("arXiv AI assistant")
    parser.add_argument("-t", "--test", action="store_true")
    args = parser.parse_args()

    agent = ArxivAgent(verbose=True)

    if args.test:
        print("Starting REPL")
        try:
            asyncio.run(run_test(agent))
        except KeyboardInterrupt:
            pass
    else:
        print("Starting bot")
        run_bot(agent)

    print("\nSaving and exiting")
    agent.save()