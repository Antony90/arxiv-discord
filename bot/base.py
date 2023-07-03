
from enum import Enum

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
    @app_commands.describe(papers="Optional list of papers to load by ID or URL.")
    async def chat(interaction: discord.Interaction, papers: str = None):
        await interaction.response.defer(thinking=True)
        if papers is not None:
            initial_msg = f"Load these papers: {papers}"
        else:
            initial_msg = "What can you do?"

        chat_history = ChatMessageHistory()
        
        ai_response = await agent.acall(
            input=initial_msg,
            chat_id=interaction.id, # user msg that invoked the command
            chat_history=chat_history
        )
    
        await interaction.followup.send(content=ai_response)
            
        
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
            chat_id = curr_msg.interaction.id
            print(chat_id, "from interaction")

        return messages[::-1] # chronological order
    
    summary = app_commands.Group(name="summary", description="Different types of paper summary methods")
    
    @summary.command(name="laymans", description="A layman's summary of a paper")
    @app_commands.describe(paper="ID or URL of an arXiv paper")
    async def laymans(interaction: discord.Interaction, paper: str):
        await interaction.response.send_message("Placeholder reponse")
    @summary.command(name="keypoints", description="A key points list summary of a paper")
    @app_commands.describe(paper="ID or URL of an arXiv paper")
    async def keypoints(interaction: discord.Interaction, paper: str):
        await interaction.response.send_message("Placeholder reponse")
    @summary.command(name="comprehensive", description="A comprehensive  summary of a paper")
    @app_commands.describe(paper="ID or URL of an arXiv paper")
    async def comprehensive(interaction: discord.Interaction, paper: str):
        await interaction.response.send_message("Placeholder reponse")

    bot.tree.add_command(summary)

    class SummaryType(Enum):
        keypoints = 0
        laymans = 1
        comprehensive = 2

    @bot.tree.command(name="summaryx")
    @app_commands.describe(paper="ID or URL of an arXiv paper")
    async def summaryx(interaction: discord.Interaction, type: SummaryType, paper: str):
        await interaction.response.send_message(f"Summary: type")

    return bot


