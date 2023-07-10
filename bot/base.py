
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Union

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


    def conversation_channel(interaction: Interaction):
        if not interaction.channel.id in CONFIG.THREAD_CHANNELS:
            raise CheckFailure(f"Must be in a conversation channel <#{CONFIG.THREAD_CHANNELS[0]}>")
        return True
    
    @bot.tree.command(name="chat", description="Start a conversation with papers in a new thread")
    @app_commands.describe(message="Send a message")
    @app_commands.check(conversation_channel)
    async def chat(inter: Interaction, message: str):
        await start_chat(inter, message)


    @bot.tree.command(name="summarize", description="Summarize a paper and start a conversation in a new thread")
    @app_commands.describe(paper="ID, url or title of an arXiv paper")
    @app_commands.describe(type="Type of summary, default: keypoints")
    @app_commands.check(conversation_channel)
    async def summarize(inter: discord.Interaction, paper: str, type: Literal['keypoints', 'laymans', 'comprehensive']='keypoints'):
        await start_chat(inter, message=f"Summarize {paper} with {type}", title="Summarize {paper}")


    async def start_chat(inter: Interaction, message: str, title: Optional[str]=None):
        user = inter.user
        await inter.response.send_message("Creating a thread")

        # create thread from embed
        title = title or f"{message} - {user.display_name}"

        resp_msg = await inter.original_response()
        
        thread = await resp_msg.create_thread(
            name=title,
            reason="chat-bot",
            auto_archive_duration=4320
        )

        # respond to user's initial msg 
        chat_history = ChatMessageHistory(messages=[])
        async with thread.typing():
            ai_response = await get_completion_response(
                chat_id=str(thread.id), 
                message=message, 
                chat_history=chat_history
            )

        await send_response(msgs=ai_response, root=thread)

    async def get_msg_history(thread: Thread, curr_msg_id: int, max_search=CONFIG.MAX_THREAD_SEARCH):
        num_msgs = 0
        messages = []
        async for msg in thread.history(limit=max_search, oldest_first=False):
            # chat window counts number of AI+User message pairs
            if num_msgs > agent.chat_window * 2:
                break
        
            if not msg.content: # thread starter message has no content
                continue
            
            # user message reply. exclude the current message
            if bot.user in msg.mentions and not msg.author.bot and msg.id != curr_msg_id:
                messages.append(HumanMessage(content=format_user_msg(msg.content, msg.author)))
            elif msg.author == bot.user:
                messages.append(AIMessage(content=msg.clean_content))

            num_msgs += 1
        messages.reverse()
        return messages
    

    def format_user_msg(msg: str, user: discord.User):
        return f"{user.display_name}: {msg}"
    
    async def send_response(msgs: list[str], root: Union[Message, Thread]) -> Optional[Message]:
        """Send a list of messages in order with each message replying to the previous.
        `msgs` (list[str]): messages to send
        `root` (Union[Message, Thread]): message or thread to send message to
        Returns the last message sent."""
        prev: Optional[Message] = None
        for msg in msgs:
            if prev:
                prev = await prev.reply(msg)
            else:
                # first message, either replying to the user Message or start of Thread
                if isinstance(root, Message):
                    prev = await root.reply(msg)
                else: # Thread
                    prev = await root.send(msg)
        return prev

    @bot.event
    async def on_message(message: Message):
        # only invoke agent chat if in thread, not bot, and mention/reply bot
        if (
            message.author.id == bot.user.id 
            or not isinstance(message.channel, Thread) 
            or bot.user not in message.mentions
        ):
            return
        
        thread = message.channel
        if thread.locked:
            return
        
        msg_history = await get_msg_history(thread, curr_msg_id=message.id)
        chat_history = ChatMessageHistory(messages=msg_history)    
        
        async with thread.typing():
            ai_response = await get_completion_response(
                chat_id=str(thread.id),
                message=message.content,
                chat_history=chat_history
            )
        await send_response(msgs=ai_response, root=message)

    async def get_completion_response(chat_id: str, message: str, chat_history: ChatMessageHistory) -> list[str]:
        print()
        print("History:")
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                print("Human:", msg.content)
            else:
                print("AI:", msg.content)
        print(f"Input: {message}")
        print()

        response = await agent.acall(
            input=message,
            chat_id=chat_id,
            chat_history=chat_history
        )

        # split messages into max size chunks
        msgs = []
        while len(response) > CONFIG.MAX_MSG:
            msgs.append(response[:CONFIG.MAX_MSG])
            response = response[CONFIG.MAX_MSG:]
        msgs.append(response) # remainder/original response 
        print("msgs", msgs)
        return msgs

    @bot.tree.error
    async def on_command_error(inter: Interaction, err: AppCommandError):
        reason = "Unknown error"
        if err.args:
            reason = err.args[0]
        print(err.with_traceback())

        if inter.response.is_done:
            resp: discord.InteractionMessage = await inter.original_response()
            send_msg = resp.reply
        else:
            send_msg = inter.response.send_message

        await send_msg(f"Error: {reason}!")




    return bot


