
from enum import Enum
from typing import Literal

import discord
from discord.app_commands import Choice
from discord import DeletedReferencedMessage, Interaction, Message, MessageReference, app_commands
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

    @bot.tree.command(name="chat", description="Start a conversation with arXiv papers")
    async def chat(interaction: discord.Interaction):
        await interaction.response.send_message("Hello, how can I help you?")
        # await interaction.response.defer(thinking=True)
        # if papers is not None:
        #     initial_msg = f"Load these papers: {papers}"
        # else:
        #     initial_msg = "What can you do?"

        # chat_history = ChatMessageHistory()
        
        # ai_response = await agent.acall(
        #     input=initial_msg,
        #     chat_id=interaction.id, # user msg that invoked the command
        #     chat_history=chat_history
        # )
    
        # await interaction.followup.send(content=ai_response)
            
        
    @bot.event
    async def on_message(message: Message):
        # if message.content.startswith(bot.command_prefix):
        #     return await bot.process_commands(message)
        
        # prevent bot from invoking command
        # only care about replies
        if message.author.id == bot.user.id or message.reference is None:
            return
        
        print(f"on_message: '{message.content}'")
        async with message.channel.typing():
            message_history = await get_reply_chain(message)
            for msg in message_history:
                print(f"{msg.author.name}: {msg.content}")
            
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
                history.add_ai_message(msg.content)
            else:
                # remove chat cmd from message
                chat_cmd = f"{bot.command_prefix}{chat.name}"
                if msg.content.startswith(chat_cmd):
                    history.add_user_message(msg.content[len(chat_cmd):])
                else:
                    history.add_user_message(msg.content)
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
                # end chain on deleted msg, use curr_msg as new chat_id
                chat_id = curr_msg.id
                break
            else:
                # already resolved
                curr_msg = next_msg
    
            messages.append(curr_msg)
        else:
            # exited loop without finding deleted msg, `curr_msg` must be bot's interaction response
            print(type(curr_msg), type(curr_msg.interaction), curr_msg.content)
            chat_id = curr_msg.interaction.id
            print(chat_id, "from interaction")

        return messages[::-1] # chronological order
    
    async def is_bot_owner(interaction: Interaction):
        return False
        # return await bot.is_owner(interaction.message.author)

    @bot.tree.command(name="summarize", description="Summarize a paper and start a conversation")
    @app_commands.describe(paper="ID or URL of an arXiv paper")
    @app_commands.describe(type="Type of summary, default: keypoints")
    @app_commands.check(is_bot_owner)
    async def summarize(interaction: discord.Interaction, paper: str, type: Literal['keypoints', 'laymans', 'comprehensive']='keypoints'):
        chat_history = ChatMessageHistory()
        
        ai_response = await agent.acall(
            input=f"Summarize {paper} with {type}",
            chat_id=interaction.id, # user msg that invoked the command
            chat_history=chat_history
        )
    
        await interaction.followup.send(content=ai_response)

    return bot


