from dataclasses import dataclass, field
from typing import Dict, List, Optional
from langchain.docstore.document import Document
import asyncio

# wrapper which automatically parses PDF into unicode text
from langchain.document_loaders.arxiv import ArxivAPIWrapper
# base search which returns Result objects
from arxiv import SortCriterion, SortOrder, Search

class ArxivFetch:

    def __init__(self, doc_content_chars_max: Optional[int] = None):
        # load 1 doc max since only care about 1st result
        self._search = ArxivAPIWrapper(
            doc_content_chars_max=doc_content_chars_max, 
            load_all_available_meta=True,
            load_max_docs=1
        )

    @staticmethod
    def _short_id(url: str):
        """Convert https://arxiv.org/abs/XXXX.YYYYYvZ to XXXX.YYYYY"""
        return url.split("/")[-1].split("v")[0]

    def get_doc_sync(self, paper_id: str) -> Document:
        docs = self._search.load(query=paper_id)
        doc = docs[0]
        doc.metadata = {
            "source": self._short_id(doc.metadata["entry_id"]),
            "title": doc.metadata["Title"]
        }
        print(f"Downloaded {doc.metadata}")
        return doc
    
    async def get_doc_async(self, paper_id: str):
        return await asyncio.get_event_loop().run_in_executor(None, self.get_doc_sync, paper_id)
    
    def _search_papers(self, query: str):
        search = Search(
            query, 
            max_results=5, 
            sort_by=SortCriterion.Relevance,
            sort_order=SortOrder.Descending
        )
        return search.results()
    
    
    def search_sync(self, query: str) -> List[str]:
        results = self._search_papers(query)
        
        output = []
        for r in results:
            output.append(f"{r.title} - {self._short_id(r.entry_id)}")
        return output
    
    async def search_async(self, query: str):
        loop = asyncio.get_event_loop()
        # use default executor (thread pool)
        results = await loop.run_in_executor(None, self._search_papers, query)
        
        output = []
        for r in results:
            output.append(f"{r.title} - {self._short_id(r.entry_id)}")
        return output
    

@dataclass
class PaperMetadata:
    title: str
    source: str

    def short_repr(self):
        return f"{self.title} - {self.source}"

@dataclass
class LoadedPapersStore:
    _to_papers: Dict[str, list[PaperMetadata]] = field(default_factory=dict)

    def get(self, chat_id: str):
        if chat_id not in self._to_papers:
            self._to_papers[chat_id] = []
        return self._to_papers[chat_id]

    def add_papers(self, chat_id: str, paper_metas: List[PaperMetadata]):
        self._to_papers[chat_id].extend(paper_metas)
    