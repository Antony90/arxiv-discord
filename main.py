__import__("dotenv").load_dotenv()

import asyncio
from langchain.memory.chat_message_histories import ChatMessageHistory

from ai.agent import ArxivAgent

async def main():
    agent = ArxivAgent(verbose=True)
    chat_id = "test"
    message_history = ChatMessageHistory(messages=[])
    
    while True:
        ai_response = await agent.acall(
            input=input(">>> "),
            chat_id=chat_id,
            message_history=message_history
        )
        message_history.add_ai_message(ai_response)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting\n")
