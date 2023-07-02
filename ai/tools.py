import asyncio
from langchain import PromptTemplate

from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from langchain.chains import RetrievalQAWithSourcesChain
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

from pydantic import BaseModel, Field, PrivateAttr

from ai.arxiv import ArxivFetch

arxiv_fetch = ArxivFetch()

class ArxivSearchSchema(BaseModel):
    query: str = Field(description="arXiv search query. Example: \"Large language models medical diagnosis\"")


class ArxivSearchTool(BaseTool):
    name = "arXiv-Search"
    description = "Search arXiv and get a list of relevant papers (title and ID)."
    args_schema: Type[ArxivSearchSchema] = ArxivSearchSchema

    def _run(self, query: str) -> List[str]:
        return arxiv_fetch.search_sync(query)
    
    async def _arun(self, query: str):
        try:
            return await arxiv_fetch.search_async(query)
        except IndexError:
            raise ToolException("No results found")

class PaperQASchema(BaseModel):
    query: str = Field(description="A question or a keyword to query loaded papers. Cannot be empty.")

class PaperQATool(BaseTool):
    name = "arXiv-Paper-Query"
    description = "Source of factual information only for loaded papers. Query the contents of the currently loaded papers."
    args_schema: Type[PaperQASchema] = PaperQASchema

    qa_func: Any
    
    def __init__(self, llm: BaseLanguageModel, vectorstore: VectorStore, *args, **kwargs):
        """
        `llm`: for self query prompting

        `vectorstore`: embeddings of currently loaded PDFs, must support self query retrieval
        """
        super().__init__(*args, **kwargs)
        self.qa_func = self._make_qa_func(llm, vectorstore)

    def _run(self, query: str) -> str:
        return self.qa_func(query)
    
    async def _arun(self, query: str) -> str:
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
        def func(query: str):
            if query.strip() == '':
                raise Exception("Query cannot be empty")
            result = qa({"question": query})
            return result
        
        return func
    

class AddPapersSchema(BaseModel):
    query: List[str] = Field(description="List of arXiv paper ids")
    
    
class AddPapersTool(BaseTool):
    name = "Add-arXiv-papers"
    description = "Downloads arXiv papers into memory. Must be done before a paper can be queried"
    args_schema: Type[AddPapersSchema] = AddPapersSchema

    # private attrs
    vectorstore: Chroma # Document embedding store
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    pdf_download_callback: Callable[[List[Dict[str, str]]], None]
    

    def _run(self, query: List[str]):
        # TODO: use threading map
        raise NotImplementedError("Use async version `_arun`")
    

    async def _arun(self, query: List[str]):
        to_download = []
        for paper_id in query:
            # check for existing Docs
            result = self.vectorstore._collection.get(where={"source":paper_id})
            if len(result["documents"]) == 0: # any key can be checked
                # not in db
                to_download.append(paper_id)
            
        docs = await asyncio.gather(*[arxiv_fetch.get_doc_async(paper_id) for paper_id in to_download])
        self.pdf_download_callback([doc.metadata for doc in docs])
        print(f"Downloaded {len(to_download)} PDFs")

        split_docs = self.text_splitter.split_documents(docs)
        
        self.vectorstore.add_documents(split_docs)
        doc_str = "\n".join([f"{doc.metadata['title']} | {doc.metadata['source']}" for doc in docs])
        return f"Success, papers loaded:\n{doc_str}"
        

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