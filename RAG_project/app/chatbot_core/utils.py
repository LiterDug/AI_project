from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.schema import HumanMessage

import sys
sys.path.append(".")
from configs.config import Load_config
CONFIG = Load_config()



from models.models import Model_loader
CONFIG_MODEL = Model_loader()

class Router:
    def __call__(self, session_id, user_infor):
        
    

        




