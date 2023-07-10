import asyncio
import aiohttp
from config import CONFIG
async def scholar_lookup(paper_id: str) -> str:
    """Get Google Scholar `cite_id`"""
    url =  f"https://scholar.google.com/scholar_lookup?arxiv_id={paper_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            print(await resp.text())
            # parse cite_id

async def get_cites(cite_id: str):
    """Citations from a Google Scholar `cite_id`"""
    url = "https://serpapi.com/search?engine=google_scholar&cites={cite_id}&api_key={CONFIG.SERPAPI_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()
        
if __name__ == "__main__":
    asyncio.run(scholar_lookup("2304.03442"))