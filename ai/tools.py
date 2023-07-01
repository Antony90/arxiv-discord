import asyncio

from langchain.tools import BaseTool, ArxivQueryRun
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.callbacks.base import BaseCallbackHandler
from langchain.base_language import BaseLanguageModel
from langchain.vectorstores.base import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ai.arxiv import ArxivFetch

class RelatedArxiv(BaseTool):
    """Gets list of related arXiv paper IDs for a given paper"""
    pass


class ArxivSearch(BaseTool):
    """Search arXiv with a text query and get paper IDs"""
    pass


class PaperQASchema(BaseModel):
    query: str = Field(description="A question or a keyword describing what you need and the ID of the paper to query. Cannot be empty.")

class PaperQATool(BaseTool):
    name = "arXiv-Paper-Query"
    description = "Primary source of factual information. Query the contents of the currently loaded papers."
    args_schema: Type[PaperQASchema] = PaperQASchema

    qa_func: Any
    
    def __init__(self, llm: BaseLanguageModel, vectorstore: VectorStore, *args, **kwargs):
        """
        `llm`: for self query prompting

        `vectorstore`: embeddings of currently loaded PDFs, must support self query retrieval
        """
        super().__init__(*args, **kwargs)
        self.qa_func = self._make_qa_func(llm, vectorstore)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.qa_func(query)
    
    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.qa_func(query)

    def _make_qa_func(self, llm, vectorstore):
        metadata_info = [
            AttributeInfo(
                name="source",
                description="arXiv paper ID",
                type="string"
            ),
            AttributeInfo(
                name="title",
                description="paper title",
                type="string"
            )
        ]
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            metadata_field_info=metadata_info,
            document_contents="arXiv PDF texts",
            verbose=True
        )
        qa: RetrievalQAWithSourcesChain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        def func(query):
            result = qa({"question": query})
            print(f"{result['sources']=}")
            return result["answer"]
        
        return func
    

class AddPapersSchema(BaseModel):
    query: List[str] = Field(description="List of arXiv paper ids")
    
    
class AddPapersTool(BaseTool):
    name = "Add-arXiv-papers"
    description = "Downloads arXiv papers into memory. Must be done before a paper can be queried"
    args_schema: Type[AddPapersSchema] = AddPapersSchema

    # private attrs
    vectorstore: VectorStore # Document embedding store
    search = ArxivFetch()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0
    )
    pdf_download_callback: Callable[[List[Dict[str, str]]], None]
    

    def _run(self, query: List[str]):
        # TODO: use threading map
        raise NotImplementedError("Use async version `_arun`")
    

    async def _arun(self, query: List[str]):
        docs: List[Document] = await asyncio.gather(*[self.search.get_doc_async(paper_id) for paper_id in query])
        self.pdf_download_callback([doc.metadata for doc in docs])

        print(f"Got {sum([len(doc.page_content) for doc in docs])} chars from {len(query)} PDFs")
        split_docs = self.text_splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} docs")
        
        self.vectorstore.add_documents(split_docs)
        return f"Added {len(query)} papers to memory"
        
class AddPapersCallback:
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running, to store paper title and id."""
        output

class SummarizePaperTool(BaseTool):
    name = "Summarize-arXiv-Paper"
    # description