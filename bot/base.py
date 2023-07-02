from ast import List
import logging

import discord
from discord import DeletedReferencedMessage, InteractionMessage, Message, MessageReference, app_commands
from discord.ext.commands.bot import Bot, Context

from langchain.memory.chat_message_histories import ChatMessageHistory
from ai.agent import ArxivAgent

from config import CONFIG

def ArxivBot(agent: ArxivAgent):

    intents = discord.Intents.default()
    intents.message_content = True
    
    bot = Bot(
        command_prefix="!", 
        description="arXiv chatbot", 
        intents=intents
    )

    @bot.event
    async def on_ready():
        synced = await bot.tree.sync()
        print(f"{len(synced)} command(s) synced")

    # @bot.tree.command(name="chat", description="arXiv chatbot")
    # @app_commands.describe(message="First message of the conversation")

    @bot.command(name="chat", description="arXiv chatbot")
    async def chat(ctx: Context, message: str):
        chat_history = ChatMessageHistory()
        chat_history.add_user_message(message)

        async with ctx.channel.typing():
            ai_response = await agent.acall(
                input=message,
                chat_id=ctx.message.id, # user msg that invoked the command
                chat_history=chat_history
            )
        
        await ctx.message.reply(content=ai_response)
            
        
    @bot.event
    async def on_message(message: Message):
        if message.content.startswith(bot.command_prefix):
            return await bot.process_commands(message)
        
        # prevent bot from invoking command
        # only care about replies
        if message.author.id == bot.user.id or message.reference is None:
            return
        
        
        async with message.channel.typing():
            message_history = await get_reply_chain(message)
            
            # original bot response message id to uniquely id a conversation
            chat_id = message_history[0].id 
            chat_history = to_chat_history(message_history)

            ai_response = await agent.acall(
                input=message.content,
                chat_id=chat_id,
                chat_history=chat_history
            )

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

    async def get_reply_chain(curr_msg: Message):
        messages: list[Message] = [curr_msg]
        channel = curr_msg.channel

        # keep on going to the next reply
        while curr_msg.reference is not None:
            next_msg = curr_msg.reference.resolved
            if next_msg is None:
                # wasn't resolved by discord, so fetch it
                curr_msg = await channel.fetch_message(curr_msg.reference.message_id)
            elif isinstance(next_msg, DeletedReferencedMessage):
                # end chain on deleted msg 
                break
            else:
                # already resolved
                curr_msg = next_msg
            messages.append(curr_msg)

        return messages[::-1] # chronological order
    

    return bot


