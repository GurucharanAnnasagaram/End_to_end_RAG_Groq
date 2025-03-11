 Web Scraping & Document Processing – Extracts content from a given webpage.
✅ Text Chunking & Vector Embeddings – Splits the text into 1000-token chunks and converts them into vector embeddings using OllamaEmbeddings().
✅ FAISS Vector Storage – Stores embeddings for fast document retrieval.
✅ Retrieval-Augmented Generation (RAG) – Retrieves relevant document snippets before answering queries.
✅ Groq's Gemma2-9B-IT Model – Generates responses based on the retrieved context.
✅ Streamlit UI – Provides an interactive interface for user queries.

🔹 Code Breakdown
1️⃣ Importing Required Libraries
python
Copy
Edit
import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
streamlit → Provides an interactive UI
os → Handles environment variables
ChatGroq → Loads Groq’s Gemma2-9B-IT model
WebBaseLoader → Scrapes content from a given webpage
OllamaEmbeddings → Converts text into vector embeddings
RecursiveCharacterTextSplitter → Splits documents into manageable chunks
FAISS → Stores and retrieves relevant text embeddings
load_dotenv → Loads API keys from environment variables
2️⃣ Load API Key from Environment Variables
python
Copy
Edit
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']
This ensures secure API access by loading the GROQ_API_KEY from the .env file.
3️⃣ Web Scraping & Vector Embedding (Session Initialization)
python
Copy
Edit
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
✅ Loads the webpage (WebBaseLoader)
✅ Extracts the text and splits it into 1000-token chunks
✅ Converts text into embeddings (OllamaEmbeddings())
✅ Stores the embeddings in FAISS for quick retrieval

4️⃣ Initialize the Chat Model (Groq’s Gemma2-9B-IT)
python
Copy
Edit
st.title("CHAT GROQ DEMO")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
Uses Groq's Gemma2-9B-IT model for answering user queries.
5️⃣ Define Prompt Template for RAG
python
Copy
Edit
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions : {input}
"""
)
Ensures the model generates answers ONLY from the retrieved webpage data.
6️⃣ Create RAG Pipeline (Retrieval-Augmented Generation)
python
Copy
Edit
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
retriever → Fetches relevant document chunks using FAISS
retrieval_chain → Feeds the retrieved text into Gemma2-9B-IT to generate an answer
7️⃣ User Input & Response Generation
python
Copy
Edit
prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])
Waits for user input in Streamlit
Retrieves relevant webpage content using FAISS
Generates an AI response using Gemma2-9B-IT
8️⃣ Display Document Context for Transparency
python
Copy
Edit
with st.expander("Document similarity search"):
    for i, doc in enumerate(response["context"]):
        st.write(doc.page_content)
        st.write("-----------------------------")
Shows retrieved document excerpts for transparency
Helps users verify where the AI is pulling its answers from
🔹 Workflow
1️⃣ Extracts webpage data → Loads https://docs.smith.langchain.com/
2️⃣ Processes & Stores Text → Splits content, converts it into embeddings, and saves it in FAISS
3️⃣ User Inputs a Question → The system retrieves relevant sections from the webpage
4️⃣ AI Generates an Answer → Groq’s Gemma2-9B-IT uses the retrieved context to generate a response
5️⃣ Displays the Answer & Source → Users can see the response along with the original document excerpts
