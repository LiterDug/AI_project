from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS

from RAG.file_embedding import *

def create_retriver(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks,embedding=embeddings) 
    retriever=vectorstore.as_retriever() 
    return retriever

def create_vectorstore_retriever(chunks):
    # Load a pretrained Sentence Transformer model
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

    # Create a vectorstore
    vectorstore=Chroma.from_documents(chunks,embedding_model)
    vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})
    return vectorstore_retreiver

def create_keyword_retriever(chunks):
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k =  3
    return keyword_retriever

def create_ensemble_retriever(chunks):
    vectorstore_retreiver = create_vectorstore_retriever(chunks)
    keyword_retriever = create_keyword_retriever(chunks)
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.3, 0.7])
    return ensemble_retriever