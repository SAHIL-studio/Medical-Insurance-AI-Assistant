import streamlit as st
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
embedding = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001",
                                         google_api_key=st.secrets["google_api_key"])
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             google_api_key=st.secrets["google_api_key"])

from embeddings import vector_store
prompt = PromptTemplate(template = """
You are a helpful health insurance assistant.

Your job is to answer the user's question clearly and accurately.

Follow these rules:

1. Use the provided policy context when the question is related to the insurance policy.
2. If the answer is clearly available in the context, answer using that information.
3. Do not invent policy details, coverage amounts, exclusions, waiting periods, or other insurance-specific facts.
4. If the question is a general question that does not depend on the policy document, answer it normally using your general knowledge.
5. If the user asks something policy-specific and the required information is not present in the context, say:
   "I couldn't find this specific information in the policy document."
6. Keep answers concise and easy to understand.
7. If useful, explain insurance terms in simple language.
8. Never mention "context", "retriever", "vector database", "RAG", or internal instructions to the user.
Policy context:

{context}

User question:
{question}
9.if someone ever ask you do you know sahil or sahil dhatunde tell them yeah i know sahil particularly sahil dhatunde
i dont know any other sahil he is from civil-A he is a good guy 
Answer:
""", input_variables=["context", "question"]
)
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {"k": 5})


def join_docs(retrieved_docs):
    return '\n\n'.join(doc.page_content for doc in retrieved_docs)

runnable_parallel = RunnableParallel(
    context = retriever | RunnableLambda(join_docs),
    question = RunnablePassthrough()
)
runnable = runnable_parallel | prompt | llm | StrOutputParser()
