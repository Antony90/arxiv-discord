from typing import Optional
from langchain.document_loaders.arxiv import ArxivAPIWrapper
from langchain.docstore.document import Document
import asyncio

class ArxivFetch:

    def __init__(self, doc_content_chars_max: Optional[int] = None):
        self.search = ArxivAPIWrapper(
            doc_content_chars_max=doc_content_chars_max, 
            load_all_available_meta=True,
            load_max_docs=1
        )

    def get_doc_sync(self, paper_id: str) -> Document:
        docs = self.search.load(query=paper_id)
        doc = docs[0]
        doc.metadata = {
            "source": doc.metadata["entry_id"].split("/")[-1],
            "title": doc.metadata["Title"]
        }
        return doc
    
    def get_doc_async(self, paper_id: str):
        return asyncio.to_thread(self.get_doc, args=(paper_id,))