from typing import List
from langchain import LLMChain
from langchain.agents import AgentExecutor, Agent, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory.chat_memory import BaseChatMessageHistory

from langchain.tools import Tool

class ArxivAgent:
    def __init__(self, chat_id: str, message_history: BaseChatMessageHistory = []):
        self.llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(
            chat_memory=message_history,
            memory_key="history"
        ) # TODO: load chat_memory from discord msg history

        self.sources = [] # list of arXiv paper ids loaded in vectorstore

        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # use discord message id as collection name
        # will persist documents loaded in a reply chain
        self.vectorstore = Chroma(
            collection_name=chat_id, 
            embedding_function=self.embeddings,
            persist_directory=".chroma"
        )

        self.agent = self._init_agent()

    def __call__(self):
        pass




    def _init_agent(self):
        return initialize_agent(
            llm=self.llm, 
            tools=self._get_tools(), 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory
        )
    
    def _get_tools(self):
        return [
            Tool.from_function(
                self._paper_qa_tool(),

            )
        ]


