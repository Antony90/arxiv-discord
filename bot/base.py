
from collections import deque
from enum import Enum
from typing import Literal

import discord
from discord.app_commands import Choice, CheckFailure, AppCommandError
from discord import DeletedReferencedMessage, ForumChannel, Interaction, Message, MessageReference, Thread, app_commands
from discord.ext.commands.bot import Bot, Context

from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.schema.messages import HumanMessage, AIMessage

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
        print(f"Guilds: {{{', '.join(map(lambda g: g.name, bot.guilds))}}}")


    def in_thread(interaction: Interaction):
        if not str(interaction.channel.id) == CONFIG.THREAD_CHANNEL:
            raise CheckFailure(f"Must be in the designated thread channel <#{CONFIG.THREAD_CHANNEL}>")
        return True
    
    @bot.tree.command(name="chat", description="Start a conversation with arXiv papers in a new thread")
    @app_commands.describe(title="Thread title")
    @app_commands.check(in_thread)
    async def chat(interaction: Interaction, title: str="Placeholder"):
        await interaction.response.send_message(content="Created thread")
        response_msg = await interaction.original_response()
        await response_msg.create_thread(name=title)

    @bot.tree.error
    async def on_command_error(interaction: Interaction, err: AppCommandError):
        reason = "Unknown error!"
        if isinstance(err, CheckFailure) and err.args:
            reason = err.args[0]

        await interaction.response.send_message(f"Error: {reason}!")

    async def get_thread_msg_history(thread_channel: Thread, max_search=30):
        num_msgs = 0
        messages = []
        async for msg in thread_channel.history(limit=max_search, oldest_first=False):
            # chat window counts number of AI+User message pairs
            if num_msgs > agent.chat_window * 2:
                break
        
            if not msg.content: # `msg` can be a reference with no content
                continue
            
            if bot.user in msg.mentions and not msg.author.bot:
                messages.append(HumanMessage(content=f"{msg.author.display_name}: {msg.clean_content}"))
            elif msg.author == bot.user and msg.reference:
                messages.append(AIMessage(content=msg.clean_content))

            num_msgs += 1

        return messages[::-1]

    @bot.event
    async def on_message(message: Message):
        # only invoke agent chat if in thread, not bot, and mention/reply bot
        if message.author.id == bot.user.id or not isinstance(message.channel, Thread) or bot.user not in message.mentions:
            return
        
        msg_history = get_thread_msg_history(message.channel)
        chat_history = ChatMessageHistory(messages=msg_history)
        for msg in chat_history.messages:
            if msg.content: # `msg` can be a reference with no content
                if isinstance(msg, HumanMessage):
                    print("Human:", msg.content)
                else:
                    print("AI:", msg.content)
        
        async with message.channel.typing():
            ai_response = await agent.acall(
                input=message.content,
                chat_id=str(message.channel.id),
                chat_history=chat_history
            )
        return await message.reply(ai_response)
        
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
        return await bot.is_owner(interaction.message.author)

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


