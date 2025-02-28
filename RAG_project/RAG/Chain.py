from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# Multi-modal chain
def create_full_chain(retriever, llm_text, llm_vision):

    template = """ ```{context} ```{query}Provide brief information and store location."""
    prompt = ChatPromptTemplate.from_template(template) # tạo promp từ template

    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm_text
        | StrOutputParser() # trích xuất câu trả lời dưới dạng văn bản.
    )
    full_chain = (
        RunnablePassthrough() | llm_vision | StrOutputParser() | rag_chain
    )
    return full_chain

#  Compressor chain: Hybrid Retriever + reranking 
def create_compression_chain(llm, ensemble_retriever,cohere_api_key):
    compressor = CohereRerank(cohere_api_key=cohere_api_key)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever= ensemble_retriever, base_compressor= compressor
    )

    compression_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=compression_retriever
    )
    return compression_chain