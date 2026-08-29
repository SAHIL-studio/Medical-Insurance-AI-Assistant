import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


loader = PyPDFLoader('policies.pdf')
pdf = loader.load()

embedding = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001",
                                        google_api_key=st.secrets["google_api_key"])
splitter = RecursiveCharacterTextSplitter(
    chunk_size= 800,
    chunk_overlap=150
)
chunks = splitter.split_documents(pdf)

vector_store = Chroma(
    embedding_function = embedding,
    persist_directory = 'info.db'
)
# vector_store.add_documents(chunks)

