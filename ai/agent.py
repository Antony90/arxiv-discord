import asyncio
from typing import Dict, List
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.agents import AgentExecutor, Agent, initialize_agent, AgentType
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage

from ai.tools import AddPapersTool, PaperQATool

class ArxivAgent:
    def __init__(self, chat_id: str, message_history=[], verbose=False):
        self.llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(
            # chat_memory=message_history,
            return_messages=True,
            memory_key="memory"
        ) # TODO: load chat_memory from discord msg history

        self.sources = [] # list of arXiv paper ids loaded in vectorstore

        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # use discord message id as collection name
        # will persist documents loaded in a reply chain
        self.vectorstore = Chroma(
            collection_name=chat_id, 
            embedding_function=self.embeddings,
            persist_directory="vectorstore"
        )

        self.agent = self._init_agent(verbose)
        self.loaded_docs = [] # metadata of loaded docs, to be inserted in prompt

    async def acall(self, input):
        print(self.memory.dict())
        await self.agent.arun(input=input, papers=self._get_loaded_papers_msg())


    def _init_agent(self, verbose):
        papers_prompt = PromptTemplate(
            input_variables=["papers"], 
            template="Here are the papers you can query in <title> | <arxiv id> format:\n{papers}\n"
        )
        message_history = MessagesPlaceholder(variable_name="memory")
        extra_prompt_messages = [message_history]
        system_message = SystemMessage(
            content="You are an expert research assistant with access to a PDF papers."
        )
        prompt = OpenAIFunctionsAgent.create_prompt(
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message
        )
        agent = OpenAIFunctionsAgent(
            llm=self.llm,
            tools=self._get_tools(),
            prompt=prompt,
            memory=self.memory,
            verbose=True
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=agent.tools,
            memory=self.memory
        )
        # return initialize_agent()
        #     llm=self.llm, 
        #     tools=self._get_tools(), 
        #     agent=AgentType.OPENAI_FUNCTIONS,
        #     agent_kwargs=agent_kwargs,
        #     memory=self.memory,
        #     verbose=verbose
        # )
    
    def _get_loaded_papers_msg(self):
        """Format metadata list for system prompt"""
        if len(self.loaded_docs) > 0:
            return "\n".join([f"{metadata['title'] | metadata['source']}" for metadata in self.loaded_docs])
        else:
            return "NONE"

    def _on_pdf_download(self, doc_metadatas: List[Dict[str, str]]) -> None:
        """Update doc metadata list"""
        self.loaded_docs.extend(doc_metadatas)
    
    def _get_tools(self):
        return [
            PaperQATool(llm=self.llm, vectorstore=self.vectorstore),
            AddPapersTool(vectorstore=self.vectorstore, pdf_download_callback=self._on_pdf_download)
        ]

