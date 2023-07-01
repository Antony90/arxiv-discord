import asyncio
from langchain import PromptTemplate

from langchain.tools import BaseTool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.base_language import BaseLanguageModel
from langchain.vectorstores.base import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document



from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr

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
    description = "Source of factual information specifically for loaded papers. Query the contents of the currently loaded papers."
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
        

class SummarizePaperSchema(BaseModel):
    query: str = Field(description="arXiv paper id")

class SummarizePaperTool(BaseTool):
    name = "Summarize-arXiv-Paper"
    description = "Summarizes a loaded paper, given its paper id."

    args_schema: Type[SummarizePaperSchema] = SummarizePaperSchema
    chain: BaseCombineDocumentsChain = Field(exclude=True, default=None)
    vectorstore: VectorStore = Field(exclude=True, default=None)


    def __init__(self, llm: BaseLanguageModel, vectorstore: VectorStore, **data) -> None:
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
        return "This is a placeholder summary."
        # sreturn elf.chain.run(query)

    async def _arun(self, query):
        return "This is a placeholder summary."
        # return await self.chain.arun(query)