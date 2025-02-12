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
from file_embedding import *
from model import *

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
sentence_transformer = load_flan_t5_model()

# 1
doc_path = "./content/nike_shoes.pdf"
image_url = "https://static.nike.com/a/images/t_PDP_1728_v1/f_auto,q_auto:eco/252f2db6-d426-4931-80a0-8b7f8f875536/calm-slides-K7mr3W.png"
question = "Provide information on given image "

chunks = create_chunks(doc_path)
retriever = create_retriver(chunks)
full_chain = create_full_chain(retriever)

message = create_message(question, image_url)
response = full_chain.invoke(message, llm_text, llm_vision)
print(response)

# 2
ensemble_retriever = create_ensemble_retriever(chunks)
compression_chain = create_compression_chain(sentence_transformer, ensemble_retriever)
question = "What is the product name?"
response = compression_chain.invoke(question)
print(response)













