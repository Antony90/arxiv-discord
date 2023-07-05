from typing import List
from pydantic import Extra

from langchain.callbacks import StdOutCallbackHandler, OpenAICallbackHandler
from langchain.prompts import MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage, BaseChatMessageHistory
from langchain.tools.base import ToolException, BaseTool

from ai.arxiv import LoadedPapersStore, PaperMetadata
from ai.prompts import AGENT_PROMPT
from ai.store import PaperStore
from ai.tools import AddPapersTool, ArxivSearchTool, PaperQATool, SummarizePaperTool
from config import CONFIG


class ArxivAgent:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    llm = OpenAI(temperature=0.0, model="gpt-3.5-turbo-0613", callbacks=[OpenAICallbackHandler()])
    chat_llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0613", callbacks=[OpenAICallbackHandler()])
    
    def __init__(self, verbose=False):
        self.verbose = verbose

        # use discord message id as collection name
        # will persist documents loaded in a reply chain
        self.vectorstore = Chroma(
            collection_name="main", 
            embedding_function=self.embeddings,
            persist_directory="vectorstore"
        )
        num_vects = len(self.vectorstore.get()["documents"])
        print(f"Loaded collection `{self.vectorstore._collection.name}` from directory `{self.vectorstore._persist_directory}` with {num_vects} vector(s)")

        self.paper_store = PaperStore(CONFIG.PAPER_STORE_PATH)        
        self.agent = self._init_agent()
    

    async def acall(self, input, chat_id: str, chat_history: BaseChatMessageHistory):
        """Call the model with a new user message and its message history.
        Loaded papers will be automatically fetched.

        Args:
            input (str): next user message
            chat_id (str): client provided unique identifier of conversation
            chat_history (BaseChatMessageHistory): message history
        """
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
            input_key="input",
            memory_key="memory"
        )
        
        exec_chain = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.agent.tools,
            memory=memory,
            verbose=self.verbose,
            callbacks=[StdOutCallbackHandler("red")]
        )

        return await exec_chain.arun(
            input=input
        )

    def _init_agent(self):
        message_history = MessagesPlaceholder(variable_name="memory")
        extra_prompt_messages = [message_history]
        system_message = SystemMessage(
            content=AGENT_PROMPT
        )
        prompt = OpenAIFunctionsAgent.create_prompt(
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message
        )
        return OpenAIFunctionsAgent(
            llm=self.chat_llm,
            tools=self._get_tools(),
            prompt=prompt,
            verbose=self.verbose
        )
        
    def _get_loaded_papers_msg(self, paper_metas: List[PaperMetadata]):
        """Format metadata list for system prompt"""
        if len(paper_metas) > 0:
            return "\n".join([metadata.short_repr() for metadata in paper_metas])
        else:
            return "NONE"

    def _parse_tool_error(self, err: ToolException):
        return f"An error occurred: {err.args[0]}"
    
    def _get_tools(self) -> List[BaseTool]:
        arxiv_search = ArxivSearchTool()

        paper_qa = PaperQATool(
            return_direct=True,
            paper_store=self.paper_store,
            llm=self.chat_llm, 
            vectorstore=self.vectorstore,
            handle_tool_error=self._parse_tool_error
        )

        paper_summarize = SummarizePaperTool(
            paper_store=self.paper_store,
            llm=self.chat_llm,
            vectorstore=self.vectorstore,
            handle_tool_error=self._parse_tool_error
        )

        return [arxiv_search, paper_qa, paper_summarize]
    
    def save(self):
        """Must call before exiting to save vectorstore and paper store"""
        self.paper_store.save()
    


