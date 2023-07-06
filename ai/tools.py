import asyncio
from pydantic import BaseModel, Extra, Field

from langchain import LLMChain, PromptTemplate
from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from langchain.base_language import BaseLanguageModel
from langchain.vectorstores.base import VectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever, LineListOutputParser
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


from typing import Any, Callable, Coroutine, Dict, List, Optional, Type


import logging

from ai.store import PaperStore
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

from ai.arxiv import ArxivFetch, LoadedPapersStore, PaperMetadata
from ai.prompts import MAP_PROMPT, MULTI_QUERY_PROMPT, REDUCE_COMPREHENSIVE_PROMPT, REDUCE_KEYPOINTS_PROMPT, REDUCE_LAYMANS_PROMPT, SEARCH_TOOL

class BasePaperTool(BaseTool):
    """Base class for tools which may want to load a paper before running their function."""

    vectorstore: Chroma
    paper_store: PaperStore
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    ) 

    def load_paper(self, paper_id: str) -> bool:
        """Load a paper. Will download if it doesn't exist in vectorstore.
        return: Whether it was already in the vectorstore."""
        
        # check for existing Docs of this paper
        result = self.vectorstore.get(where={"source":paper_id})
        if len(result["documents"]) != 0: # any key can be checked
            print(f"{paper_id} already in db")
            return True # already in db

        doc, abstract = arxiv_fetch.get_doc_sync(paper_id)
        self.paper_store.save_title_abstract(paper_id, doc.metadata["title"], abstract)

        # split and embed docs in vectorstore
        split_docs = self.text_splitter.split_documents([doc])
        self.vectorstore.add_documents(split_docs)
        return False
    
    async def aload_paper(self, paper_id: str) -> bool:
        """Load a paper. Will download if it doesn't exist in vectorstore.
        return: Whether it was already in the vectorstore."""
        
        # check for existing Docs of this paper
        result = self.vectorstore.get(where={"source":paper_id})
        if len(result["documents"]) != 0: # any key can be checked
            print(f"{paper_id} already in db")
            return True # already in db

        doc, abstract = await arxiv_fetch.get_doc_async(paper_id)
        self.paper_store.save_title_abstract(paper_id, doc.metadata["title"], abstract)
        
        # split and embed docs in vectorstore
        split_docs = self.text_splitter.split_documents([doc])
        self.vectorstore.add_documents(split_docs) # TODO: find store with async implementation
        return False


arxiv_fetch = ArxivFetch()

class ArxivSearchSchema(BaseModel):
    query: str = Field(description="arXiv search query. Specific queries are preferred.")


class ArxivSearchTool(BaseTool):
    name = "arXiv-Search"
    description = SEARCH_TOOL
    args_schema: Type[ArxivSearchSchema] = ArxivSearchSchema

    def _run(self, query: str) -> List[str]:
        return arxiv_fetch.search_sync(query)
    
    async def _arun(self, query: str):
        try:
            return await arxiv_fetch.search_async(query)
        except IndexError:
            raise ToolException("No results found")

class PaperQASchema(BaseModel):
    question: str = Field(description="A question to ask about a paper. Cannot be empty. Do not include the paper ID")
    paper_id: str = Field(description="ID of paper to query")

class PaperQATool(BasePaperTool):
    name = "arXiv-Paper-Question"
    description = "Ask a question about the contents of a paper. Source of factual information for an arXiv paper. Don't include paper ID/URL in the question"
    args_schema: Type[PaperQASchema] = PaperQASchema

    # private attrs
    llm: BaseChatModel # for QA retrieval chain prompting
    vectorstore: Chroma # embeddings of currently loaded PDFs, must support self query retrieval

    class Config:
        extra = Extra.allow
    
    def _run(self, question: str, paper_id: str) -> str:
        self.load_paper(paper_id)
        qa = self._make_qa_chain(paper_id)
        return qa.run(question)


    async def _arun(self, question: str, paper_id: str) -> str:
        await self.aload_paper(paper_id)
        qa = self._make_qa_chain(paper_id)
        return qa.run(question)


    def _make_qa_chain(self, paper_id: str):
        """Make a RetrievalQA chain which filters by this paper_id"""
        filter = {
            "source": paper_id
        }
        paper_title = self.paper_store.get_title(paper_id)
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"filter": filter})
        # generate multiple queries from different perspectives to pull a richer set of Documents
        output_parser = LineListOutputParser()
        llm_chain = LLMChain(llm=self.llm, prompt=MULTI_QUERY_PROMPT(paper_title), output_parser=output_parser)

        # TODO: implement async get_relevant_docs with subclass
        mq_retriever = MultiQueryRetriever(
            retriever=retriever,
            llm_chain=llm_chain,
            parser_key="lines"
        )
            
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=mq_retriever
        )
        return qa
    

class AddPapersSchema(BaseModel):
    query: List[str] = Field(description="List of arXiv paper ids")
    chat_id: str = Field(description="Chat ID to add the papers to")
    
class AddPapersTool(BaseTool):
    name = "Add-arXiv-papers"
    description = "Loads arXiv papers into your memory. Must be done before any function can be applied to a paper."
    args_schema: Type[AddPapersSchema] = AddPapersSchema

    # private attrs
    vectorstore: Chroma # Document embedding store
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    paper_load_callback: Callable[[List[PaperMetadata], str], None]
    

    def _run(self, query: List[str]):
        # TODO: use threading map
        raise NotImplementedError("Use async version `_arun`")
    

    async def _arun(self, query: List[str], chat_id: str):
        # arXiv paper ids to download PDF Documents for
        to_download: List[str] = []
        # papers already in vectorstore
        existing_metas: List[PaperMetadata] = []

        for paper_id in query:
            # check for existing Docs of this paper
            result = self.vectorstore._collection.get(where={"source":paper_id})
            if len(result["documents"]) == 0: # any key can be checked
                # not in db
                to_download.append(paper_id)
            else:
                # take the first result's metadata
                existing_metas.append(PaperMetadata(**result["metadatas"][0]))

        # notify callback with existing papers
        self.paper_load_callback(existing_metas, chat_id)
            
        docs = await asyncio.gather(*[arxiv_fetch.get_doc_async(paper_id) for paper_id in to_download])
        # also notify on newly fetched papers
        downloaded_metas = [PaperMetadata(**doc.metadata) for doc in docs]
        
        if len(docs) > 0:
            # text spltter will error on empty list
            self.paper_load_callback(downloaded_metas, chat_id)
            # split and embed docs in vectorstore
            split_docs = self.text_splitter.split_documents(docs)
            self.vectorstore.add_documents(split_docs)
        
        paper_metas = existing_metas + downloaded_metas
        output_str = "\n".join([meta.short_repr() for meta in paper_metas])

        return f"Papers loaded:\n{output_str}"
        

class SummarizePaperSchema(BaseModel):
    paper_id: str = Field(description="arXiv paper id")
    # Need to mention default values in natural language
    # since OpenAI function calling JSON does not specify exactly what the default value is.
    # This means the LLM cannot reason about whether the default value or another value is more appopriate
    type: str = Field(description="Type of summary. One of: {keypoints, laymans, comprehensive}, default to keypoints")

class SummarizePaperTool(BasePaperTool):
    name = "Summarize-arXiv-Paper"
    description = "Summarizes a paper given its ID."

    args_schema: Type[SummarizePaperSchema] = SummarizePaperSchema

    # private attrs
    vectorstore: Chroma 
    llm: BaseChatModel

    _summary_prompt = {
        "keypoints": REDUCE_KEYPOINTS_PROMPT, 
        "laymans": REDUCE_LAYMANS_PROMPT,
        "comprehensive": REDUCE_COMPREHENSIVE_PROMPT
    }

    def _run(self, paper_id: str):
        try:
            combine_prompt = self._summary_prompt[type]
        except KeyError:
            raise ToolException(f"Unknown summary type: \"{type}\"")
        self.load_paper(paper_id)

        existing_summary = self.paper_store.get_summary(paper_id, type)
        if existing_summary:
            return existing_summary

        map_reduce_chain = load_summarize_chain(
            llm=self.llm, 
            chain_type="map_reduce", 
            map_prompt=MAP_PROMPT,
            combine_prompt=combine_prompt
        )
            
        result = self.vectorstore.get(where={"source": paper_id})
        chunks = result["documents"]
        if len(chunks) == 0:
            raise ToolException("Document not loaded or does not exist")
        docs = [Document(page_content=chunk) for chunk in chunks]
        summary = map_reduce_chain.run(docs)

        self.paper_store.save_summary(paper_id, type, summary)

    async def _arun(self, paper_id: str, type="keypoints"):
        try:
            combine_prompt = self._summary_prompt[type]
        except KeyError:
            raise ToolException(f"Unknown summary type: \"{type}\"")
        await self.aload_paper(paper_id)

        existing_summary = self.paper_store.get_summary(paper_id, type)
        if existing_summary:
            return existing_summary

        map_reduce_chain = load_summarize_chain(
            llm=self.llm, 
            chain_type="map_reduce", 
            map_prompt=MAP_PROMPT,
            combine_prompt=combine_prompt
        )
            
        result = self.vectorstore.get(where={"source": paper_id})
        chunks = result["documents"]
        if len(chunks) == 0:
            raise ToolException("Document not loaded or does not exist")
        
        docs = [Document(page_content=chunk) for chunk in chunks]
        summary = await map_reduce_chain.arun(docs)

        self.paper_store.save_summary(paper_id, type, summary)


class FiveKeywordsTool(BaseTool):
    pass