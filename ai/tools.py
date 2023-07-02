import asyncio
from pydantic import BaseModel, Extra, Field, PrivateAttr

from langchain import PromptTemplate
from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.base_language import BaseLanguageModel
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


from typing import Any, Callable, Coroutine, Dict, List, Optional, Type

from ai.arxiv import ArxivFetch, LoadedPapersStore, PaperMetadata




arxiv_fetch = ArxivFetch()

class ArxivSearchSchema(BaseModel):
    query: str = Field(description="arXiv search query. Example: \"Large language models medical diagnosis\"")


class ArxivSearchTool(BaseTool):
    name = "arXiv-Search"
    description = "Search arXiv and get a list of relevant papers (title and ID). Only use if user specifically wants you to search arXiv."
    args_schema: Type[ArxivSearchSchema] = ArxivSearchSchema

    def _run(self, query: str) -> List[str]:
        return arxiv_fetch.search_sync(query)
    
    async def _arun(self, query: str):
        try:
            return await arxiv_fetch.search_async(query)
        except IndexError:
            raise ToolException("No results found")

class PaperQASchema(BaseModel):
    query: str = Field(description="A question to ask about a currently loaded paper. Cannot be empty.")
    paper_id: str = Field(description="ID of paper to query")
    chat_id: str = Field(description="Chat ID of current conversation")

class PaperQATool(BaseTool):
    name = "arXiv-Paper-Query"
    description = "Source of factual information for a loaded paper. The paper must be loaded. When using the output of this tool, retain as much information as possible, do not shorten."
    args_schema: Type[PaperQASchema] = PaperQASchema

    # private attrs
    llm: BaseLanguageModel # for QA retrieval chain prompting
    vectorstore: Chroma # embeddings of currently loaded PDFs, must support self query retrieval
    _user_paper_store: LoadedPapersStore

    class Config:
        extra = Extra.allow
    
    def _validate(self, paper_id: str, chat_id: str) -> None:
        # check that the paper is already loaded for this chat
        # prevents vague queries which cause SelfQueryRetriever to not filter by paper ids
        is_loaded = paper_id in map(lambda meta: meta.source, self._user_paper_store.get(chat_id))
        if not is_loaded:
            raise ToolException(f"Paper {paper_id} is not loaded for this chat, would you like to load it?")
        if not len(paper_id) > 0: # TODO: check if valid arXiv ID
            raise ToolException(f"\"{paper_id}\" is not a valid arXiv ID.")


    def _run(self, query: str, paper_id: str, chat_id: str) -> str:
        self._validate(paper_id, chat_id)
        qa = self._make_qa_chain(paper_id)
        return qa.run(query)


    async def _arun(self, query: str, paper_id: str, chat_id: str) -> str:
        self._validate(paper_id, chat_id)
        qa = self._make_qa_chain(paper_id)
        return await qa.arun(query)


    def _make_qa_chain(self, paper_id: str):
        """Make a RetrievalQA chain which filters by this paper_id"""
        filter = {
            "source": paper_id
        }
        retriever = self.vectorstore.as_retriever(search_kwargs={"filter": filter})
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )
        return qa
    

class AddPapersSchema(BaseModel):
    query: List[str] = Field(description="List of arXiv paper ids")
    chat_id: str = Field(description="Chat ID to add the papers to")
    
class AddPapersTool(BaseTool):
    name = "Add-arXiv-papers"
    description = "Loads arXiv papers into your memory. Must be done before a paper can be queried."
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
        print(f"{to_download=}")
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
    query: str = Field(description="arXiv paper id")

class SummarizePaperTool(BaseTool):
    name = "Summarize-arXiv-Paper"
    description = "Summarizes a loaded paper, given its paper id."

    args_schema: Type[SummarizePaperSchema] = SummarizePaperSchema
    chain: BaseCombineDocumentsChain = Field(exclude=True, default=None)
    vectorstore: Chroma = Field(exclude=True, default=None)


    def __init__(self, llm: BaseLanguageModel, vectorstore: Chroma, **data) -> None:
        super().__init__(**data)
        prompt = PromptTemplate(
            input_variables=["text"],
            template=\
"""Write a concise summary of the following paper, focussing on objective fact:


"{text}"


CONCISE SUMMARY:""")

        self.chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        self.vectorstore = vectorstore

    def _run(self, query):
        raise Exception("Bonk! Too many tokens here!")
        
        result = self.vectorstore._collection.get(where={"source": query})
        docs = result["documents"]
        return self.chain.run(docs)

    async def _arun(self, query):
        return "This is a placeholder summary."
        # return await self.chain.arun(query)