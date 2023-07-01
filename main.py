__import__("dotenv").load_dotenv()
import asyncio
from ai.agent import ArxivAgent

async def main():
    agent = ArxivAgent(chat_id="test", message_history=[], verbose=True)
    
    while True:
        await agent.acall(input(">>> "))

if __name__ == "__main__":
    asyncio.run(main())
