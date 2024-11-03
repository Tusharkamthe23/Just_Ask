import os
from PyPDF2 import PdfReader
import pandas as pd
import openpyxl
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Set up the Groq API key
os.environ["groq_api_key"] = st.secrets["groq_api_key"]
groq_api_key = os.environ.get("groq_api_key")

def split_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts

def perform_query(query, document_search, chain, chat_history):
    # Perform a similarity search
    docs = document_search.similarity_search(query)

    # Use the run method with named arguments
    return chain.run(input_documents=docs, question=query, chat_history=chat_history)

uploaded_file = st.file_uploader("Upload a document (CSV, Excel, TXT, Word, or PDF)", type=["csv", "xlsx", "txt", "docx", "pdf"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    raw_text = ""

    # Process uploaded file based on its extension
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

    # Initialize embeddings and FAISS vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    document_search = FAISS.from_texts(texts, embeddings)

    # Initialize the ChatGroq model
    model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    
    # Initialize the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(model, retriever=document_search.as_retriever(), chain_type="stuff")

    # Initialize session state for query history and chat history
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history

    st.title("PDF Query App")
    query = st.text_input("Ask a question:")

    if st.button("Search"):
        # Call perform_query with keyword arguments only
        result = perform_query(query=query, document_search, chain=chain, chat_history=st.session_state.chat_history)
        st.write("Answer:", result)
        
        # Append current query and answer to session state
        st.session_state.chat_history.append((query, result))  # Add current question and answer to chat history
        st.session_state.query_history.append({"query": query, "answer": result})

    st.text("*Query History:*")
    for item in st.session_state.query_history:
        st.text(f"Query: {item['query']}\nAnswer: {item['answer']}")
