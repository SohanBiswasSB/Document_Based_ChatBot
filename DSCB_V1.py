#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The imports for Document Ingestion + Vectorization
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# The imports for RAG Chain with GROQ Integration
import requests
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM

# The imports for Streamlit
import streamlit as st

# The imports for the Credentials and Environment Variables
from dotenv import load_dotenv
import os


# In[2]:


# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
HF_MODEL = os.getenv("HF_EMBEDDING_MODEL")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR")


# In[3]:


# Load and split PDF from local file
loader = PyPDFLoader(r"C:\Users\sohan\Desktop\Dev\Personal\Document_Based_ChatBot\pdf_documents\Sohan_Biswas_CV.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed and store in Chroma

# This section is for the Local based embedding model from Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name=HF_MODEL)
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=CHROMA_DB_DIR)

vectordb.persist()

# # This section is for the API Key based embedding model from Hugging Face
# embedding_model = HuggingFaceInferenceAPIEmbeddings(
#     api_key=os.getenv("HF_API_KEY"),
#     model_name=os.getenv("HF_EMBEDDING_MODEL")
# )


# In[4]:


class GroqLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        return response.json()['choices'][0]['message']['content']


# In[5]:


# Loading the vector store
embedding = HuggingFaceEmbeddings()
retriever = vectordb.as_retriever()

# Setting up the LLM from GROQ
llm = GroqLLM()

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# In[ ]:


# Streamlit App for Document ChatBot
st.title("Document ChatBot with GROQ + Chroma + Langchain")

query = st.text_input("Ask a question about the document:")
if query:
    response = qa_chain.run(query)
    st.write("**Answer:**", response)

# Run this script to conver the Jupyter Notebook to a Python script
# jupyter nbconvert --to script DSCB_V1.ipynb
# streamlit run Version.py

