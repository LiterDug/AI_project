from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
import os

# Khai bao bien
dir = os.path.dirname(__file__) + "/"
pdf_data_path = dir + "data"
vector_db_path = dir + "vectorstores/db_faiss"

def create_chunks(doc_path):
    # Load document
    loader=PyPDFLoader(doc_path)
    docs=loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    return chunks

def create_db_from_files(pdf_data_path, vector_db_path):
    # Khai bao loader de quet toan bo thu muc data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return chunks

# db = create_db_from_files(pdf_data_path, vector_db_path)


def read_vectors_db(vector_db_path):
    embedding_model = GPT4AllEmbeddings(model_file='models/all-MiniLM-L6-v2-f16')
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db   

# db = read_vectors_db(vector_db_path)
# db.index.ntotal








