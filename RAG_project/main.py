import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import base64
import httpx

from Chain import *
from Retriever import *

loader = TextLoader("./content/nike_shoes.txt") # đọc file

def load_model(model_name, GOOGLE_API_KEY):
  llm = ChatGoogleGenerativeAI(model=model_name)
  os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
  return llm

def create_chunks(doc_path):
    # Load document
    loader=PyPDFLoader(doc_path)
    docs=loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    return chunks

def create_message(question, image_url):
    image_url = "https://static.nike.com/a/images/t_PDP_1728_v1/f_auto,q_auto:eco/252f2db6-d426-4931-80a0-8b7f8f875536/calm-slides-K7mr3W.png"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
    )
    return message

GOOGLE_API_KEY = "..."
llm_vision = load_model("gemini-1.5-flash",GOOGLE_API_KEY)
llm_text = load_model("gemini-pro",GOOGLE_API_KEY)

loader = TextLoader("./content/nike_shoes.txt")
text=loader.load()[0].page_content
retriever = create_retriver(text)

full_chain = create_full_chain(retriever)

image_url = "https://static.nike.com/a/images/t_PDP_1728_v1/f_auto,q_auto:eco/252f2db6-d426-4931-80a0-8b7f8f875536/calm-slides-K7mr3W.png"
question = "Provide information on given image "
message = create_message(question, image_url)
response = full_chain.invoke(message)













