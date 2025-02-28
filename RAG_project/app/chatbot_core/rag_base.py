
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.schema import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import SystemMessage
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema.runnable import RunnablePassthrough
import sys
sys.path.append("./app/rag")
from prompt import CHITCHAT_SYSTEM, CHITCHAT_HUMAN_MESSAGE, REWRITE_PROMPT, REWRITE_HUMAN_MESAGE, ROUTING_PROMPT, HUMAN_MESSAGE

import sys
sys.path.append(".")
from configs.config import Load_config
CONFIG = Load_config()

from models.models import Model_loader
CONFIG_MODEL = Model_loader()

class DictState(TypedDict):
    human_input: Annotated[Sequence[BaseMessage], add_messages]

class RAG:
    def __init__(self, user_infor, session_id):
        self.model_name = user_infor
        self.session_id = session_id
        self.llm = CONFIG_MODEL.load_gemini()

        self.prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(CHITCHAT_SYSTEM),
            HumanMessagePromptTemplate.from_template(CHITCHAT_HUMAN_MESSAGE)
        ])
        runnable = self.prompt | self.llm

        workflow = StateGraph(state_schema=DictState)

        def call_model(state: DictState):
            response = runnable.invoke(state)
            # Update message history with response:
            return {"human_input": [response]}

        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        self.config = {"configurable": {"thread_id": session_id}}
        
    def run(self, query):

        input_dict = {
            "human_input": [HumanMessage(query)],
        }

        try:
            output = self.app.invoke(input_dict, self.config)
            response = output["human_input"][-1].content # last message = last AI response
            return response
        except Exception as e:
            print(e)



if __name__ == "__main__":
    rag = RAG(user_infor="thao", session_id="1234")
    while (1):
        query = input("Enter your query: ")
        response = rag.run(query)
        print(response)














