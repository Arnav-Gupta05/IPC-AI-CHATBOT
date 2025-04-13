import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# PDF File Path
PDF_PATH = "IPC.pdf"

# Extract PDF Text
def get_pdf_text(pdf_path):
    raw_text = ""
    pdfreader = PdfReader(pdf_path)
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Split into Chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Create Vector Store
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# Load Chatbot Chain
def get_conversation_chain():
    prompt_template = """You are a helpful assistant. Answer based on the provided context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    return load_qa_chain(model, chain_type="stuff", prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"]))

# Streamlit UI
st.set_page_config(page_title="LegallyHer AI", page_icon="⚖️")
st.title("⚖️ Indian Penal Code AI Chatbot")
st.markdown("Ask any legal questions based on the Indian Penal Code (IPC). The bot will retrieve answers from the IPC document.")

# Process PDF on Startup
if not os.path.exists("faiss_index"):
    with st.spinner("Processing Indian Penal Code... 📄🔍"):
        raw_text = get_pdf_text(PDF_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vectorstore(text_chunks)
        st.success("✅ IPC Processed! You can now ask questions.")

# Load Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = new_db.as_retriever()

# Question Input
user_question = st.text_input("🔎 Ask a question about the IPC:")
if user_question and st.button("Get Answer"):
    with st.spinner("Generating response... 🤖"):
        docs = retriever.get_relevant_documents(user_question)
        chain = get_conversation_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question})
        
        st.success("✅ Answer Generated!")
        st.write("**Reply:**", response["output_text"])

# Footer
st.markdown("---")
st.markdown("🔹 *Made for her <3*")




# """ Names for this bot:-
# LegallyHer AI
# JustHer AI

# """   