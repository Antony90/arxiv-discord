import asyncio
from pydantic import BaseModel, Extra, Field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type
import logging

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

from arxiv import Result

from ai.store import PaperStore
from ai.arxiv import ArxivFetch, LoadedPapersStore, PaperMetadata
from ai.prompts import ABSTRACT_QS_PROMPT, ABSTRACT_SUMMARY_PROMPT, MAP_PROMPT, MULTI_QUERY_PROMPT, REDUCE_COMPREHENSIVE_PROMPT, REDUCE_KEYPOINTS_PROMPT, REDUCE_LAYMANS_PROMPT, SEARCH_TOOL



class BasePaperTool(BaseTool):
    """Base class for tools which may want to load a paper before running their function."""

    vectorstore: Chroma
    paper_store: PaperStore
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    ) 

    def load_paper(self, paper_id: str, chat_id: str) -> bool:
        """Load a paper. Will download if it doesn't exist in vectorstore.
        return: Whether it was already in the vectorstore."""
        
        # check for existing Docs of this paper
        result = self.vectorstore.get(where={"source":paper_id})
        if len(result["documents"]) != 0: # any key can be checked
            found = True # already in db
        else:
            doc, abstract = arxiv_fetch.get_doc_sync(paper_id)
            self.paper_store.save_title_abstract(paper_id, doc.metadata["title"], abstract)

            # split and embed docs in vectorstore
            split_docs = self.text_splitter.split_documents([doc])
            self.vectorstore.add_documents(split_docs)
            found = False

        self.paper_store.add_paper_to_chat(paper_id, chat_id)
        return found
    async def aload_paper(self, paper_id: str, chat_id: str) -> bool:
        """Load a paper. Will download if it doesn't exist in vectorstore.
        return: Whether it was already in the vectorstore."""
        
        # check for existing Docs of this paper
        result = self.vectorstore.get(where={"source":paper_id})
        if len(result["documents"]) != 0: # any key can be checked
            found = True # already in db
        else:
            doc, abstract = await arxiv_fetch.get_doc_async(paper_id)
            self.paper_store.save_title_abstract(paper_id, doc.metadata["title"], abstract)
            
            # split and embed docs in vectorstore
            split_docs = self.text_splitter.split_documents([doc])
            self.vectorstore.add_documents(split_docs) # TODO: find store with async implementation
            found = False

        self.paper_store.add_paper_to_chat(paper_id, chat_id)
        return found

arxiv_fetch = ArxivFetch()


class ArxivSearchSchema(BaseModel):
    query: str = Field(description="arXiv search query. Refuse user queries which are to vague.")


class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = SEARCH_TOOL
    args_schema: Type[ArxivSearchSchema] = ArxivSearchSchema

    def format_result(self, result: Result):
        abstract = result.summary[:200].replace('\n', '')
        return f"- [`{ArxivFetch._short_id(result.entry_id)}`] - `{result.title}`\n    - {abstract}..."

    def _run(self, query: str) -> List[str]:
        return "\n".join([self.format_result(r) for r in arxiv_fetch.search_async(query)])
    
    async def _arun(self, query: str):
        return "\n".join([self.format_result(r) for r in await arxiv_fetch.search_async(query)])

class PaperQASchema(BaseModel):
    question: str = Field(description="A question to ask about a paper. Cannot be empty. Do not include the paper ID")
    paper_id: str = Field(description="ID of paper to query")
    chat_id: str = Field(description="Chat ID")

class PaperQATool(BasePaperTool):
    name = "paper_question_answering"
    description = "Ask a question about the contents of a paper. Primary source of factual information for a paper. Don't include paper ID/URL in the question."
    args_schema: Type[PaperQASchema] = PaperQASchema

    # private attrs
    llm: BaseChatModel # for QA retrieval chain prompting
    vectorstore: Chroma # embeddings of currently loaded PDFs, must support self query retrieval

    class Config:
        extra = Extra.allow
    
    def _run(self, question, paper_id, chat_id) -> str:
        self.load_paper(paper_id, chat_id)
        qa = self._make_qa_chain(paper_id)
        return qa.run(question)


    async def _arun(self, question, paper_id, chat_id) -> str:
        await self.aload_paper(paper_id, chat_id)
        qa = self._make_qa_chain(paper_id)
        return qa.run(question)


    def _make_qa_chain(self, paper_id: str):
        """Make a RetrievalQA chain which filters by this paper_id"""
        filter = {
            "source": paper_id
        }
        # paper_title = self.paper_store.get_title(paper_id)
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"filter": filter})
        # # generate multiple queries from different perspectives to pull a richer set of Documents
        # output_parser = LineListOutputParser()
        # llm_chain = LLMChain(llm=self.llm, prompt=MULTI_QUERY_PROMPT(paper_title), output_parser=output_parser)

        # # TODO: implement async get_relevant_docs with subclass
        # mq_retriever = MultiQueryRetriever(
        #     retriever=retriever,
        #     llm_chain=llm_chain,
        #     parser_key="lines"
        # )
            
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )
        return qa
    

class AbstractSummarySchema(BaseModel):
    paper_id: str = Field(description="arXiv paper ID")
    chat_id: str = Field(description="Chat ID")
    
class AbstractSummaryTool(BasePaperTool):
    name = "abstract_summary"
    description = "Returns a bullet point summary of the abstract. Do not modify this tool's output in your response. Use specifically when a short summary is needed."
    args_schema: Type[AbstractSummarySchema] = AbstractSummarySchema
    
    llm: BaseLanguageModel

    def _run(self, paper_id, chat_id):
        # TODO: use threading map
        raise NotImplementedError("Use async version `_arun`")
    

    async def _arun(self, paper_id, chat_id):
        await self.aload_paper(paper_id, chat_id)

        abstract = self.paper_store.get_abstract(paper_id)
        title = self.paper_store.get_title(paper_id)

        # TODO: should be an instance variable since unchanged
        # summarize the abstract highlighting key points
        abs_summary_chain = LLMChain(prompt=ABSTRACT_SUMMARY_PROMPT, llm=self.llm)
        abs_summary = abs_summary_chain.arun(title=title, abstract=abstract)
        
        return abs_summary
        

class SummarizePaperSchema(BaseModel):
    paper_id: str = Field(description="arXiv paper id")
    # Need to mention default values in natural language
    # since OpenAI function calling JSON does not specify exactly what the default value is.
    # This means the LLM cannot reason about whether the default value or another value is more appopriate
    type: str = Field(description="Type of summary. One of: {keypoints, laymans, comprehensive}, default to keypoints")
    chat_id: str = Field(description="Chat ID")


class SummarizePaperTool(BasePaperTool):
    name = "summarize_paper_full"
    description = "Summarizes a paper in full, with significant detail."

    args_schema: Type[SummarizePaperSchema] = SummarizePaperSchema

    # private attrs
    vectorstore: Chroma 
    llm: BaseChatModel

    _summary_prompt = {
        "keypoints": REDUCE_KEYPOINTS_PROMPT, 
        "laymans": REDUCE_LAYMANS_PROMPT,
        "comprehensive": REDUCE_COMPREHENSIVE_PROMPT
    }

    def _run(self, paper_id, type, chat_id):
        try:
            combine_prompt = self._summary_prompt[type]
        except KeyError:
            raise ToolException(f"Unknown summary type: \"{type}\"")
        self.load_paper(paper_id, chat_id)

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

    async def _arun(self, paper_id, type, chat_id):
        try:
            combine_prompt = self._summary_prompt[type]
        except KeyError:
            raise ToolException(f"Unknown summary type: \"{type}\"")
        await self.aload_paper(paper_id, chat_id)

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


class AbstractQuestionsSchema(BaseModel):
    paper_id: str = Field(description="ID of paper.")
    chat_id: str = Field(description="Chat ID")

class AbstractQuestionsTool(BasePaperTool):
    name = "get_abstract_questions"
    description = "Generates a set of questions to jump start discussion of a paper. Uses the paper's abstract."

    args_schema: Type[AbstractQuestionsSchema] = AbstractQuestionsSchema

    paper_store: PaperStore
    llm: BaseLanguageModel

    def _run(self, paper_id, chat_id):
        self.load_paper(paper_id, chat_id)
        abstract = self.paper_store.get_abstract(paper_id)
        title = self.paper_store.get_title(paper_id)

        # TODO: should be an instance variable since unchanged
        llm_chain = LLMChain(prompt=ABSTRACT_QS_PROMPT, llm=self.llm)
        
        return llm_chain.run(title=title, abstract=abstract)
    
    async def _arun(self, paper_id, chat_id):
        await self.aload_paper(paper_id, chat_id)
        abstract = self.paper_store.get_abstract(paper_id)
        title = self.paper_store.get_title(paper_id)

        # TODO: should be an instance variable since unchanged
        llm_chain = LLMChain(prompt=ABSTRACT_QS_PROMPT, llm=self.llm)
        
        return await llm_chain.arun(title=title, abstract=abstract)

class FiveKeywordsTool(BaseTool):
    pass