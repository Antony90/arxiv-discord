from ast import List
import logging

import discord
from discord import InteractionMessage, Message, MessageReference, app_commands
from discord.ext.commands.bot import Bot, Context

from langchain.memory.chat_message_histories import ChatMessageHistory
from ai.agent import ArxivAgent

from config import CONFIG

def ArxivBot():

    intents = discord.Intents.default()
    intents.message_content = True
    
    bot = Bot(
        command_prefix="!", 
        description="arXiv chatbot", 
        intents=intents
    )

    agent = ArxivAgent(verbose=True)

    @bot.event
    async def on_ready():
        synced = await bot.tree.sync()
        print(f"{len(synced)} command(s) synced")

    # @bot.tree.command(name="chat", description="arXiv chatbot")
    # @app_commands.describe(message="First message of the conversation")

    @bot.command(name="chat", description="arXiv chatbot")
    async def chat(ctx: Context, message: str):
        print("chat command invoked")
        ai_response = message.upper()
        
        # await agent.acall(
        #     input=message,
        #     chat_id=ctx.message.id # user msg that invoked the command
        # )
        await ctx.message.reply(content=ai_response)
            
        
    @bot.event
    async def on_message(message: Message):
        if message.author.id == bot.user.id:
            return
        bot.process_commands(message)
        print("on_message", message.content)
        
        message_history = get_reply_chain(message)
        print()
        print(message_history)
        print("========================")
        # original bot response message id to uniquely id a conversation
        chat_id = message_history[0].id 
        chat_history = to_chat_history(message_history)
        print(chat_history)
        print()

        ai_response = message.content.upper() 
        
        # await agent.acall(
        #     input=message.content,
        #     chat_id=chat_id,
        #     chat_history=chat_history
        # )

        await message.reply(ai_response)

    def to_chat_history(message_history: list[Message]):
        history = ChatMessageHistory()
        for msg in message_history:
            if msg.author.id == bot.user.id:
                add = history.add_ai_message
            else:
                add = history.add_user_message
            add(msg.content)
        return history

    def get_reply_chain(curr_msg: Message):
        messages = [curr_msg]

        while curr_msg is not None and isinstance(curr_msg.reference, MessageReference):
            curr_msg = curr_msg.reference.resolved
            messages.append(curr_msg)

        return messages[::-1]
    async def find_root_msg():
        return Message()


    return bot


