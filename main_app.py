import streamlit as st
import os
from PyPDF2 import PdfReader
import pandas as pd
import openpyxl  
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


# OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]  

def split_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts

def perform_query(query, document_search, chain):
    docs = document_search.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

st.title("Document Analyzer")

uploaded_file = st.file_uploader("Upload a document (CSV, Excel, TXT, Word, or PDF)", type=["csv", "xlsx", "txt", "docx", "pdf"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    raw_text = ""

    if file_extension == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    elif file_extension in ["csv", "txt"]:
        raw_text = uploaded_file.read().decode("utf-8")
    elif file_extension == "xlsx":
        df = pd.read_excel(uploaded_file)
        raw_text = df.to_string()  
    
    texts = split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)

  
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

  
    query = st.text_input("Ask a question:")

    if st.button("Search"):
        
        result = perform_query(query, document_search, chain)
        st.write("Answer:", result)


        st.session_state.query_history.append({"query": query, "answer": result})

    st.text("*Query History:*")
    for item in st.session_state.query_history:
        st.text(f"Query: {item['query']}\nAnswer: {item['answer']}")
