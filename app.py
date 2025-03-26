import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import dotenv
import uuid

from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_rag_response,
)

# Load environment variables
dotenv.load_dotenv()

def init_groq_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0.2,
        max_tokens=2500
    )

# Initialize GROQ model
llm_groq = init_groq_model()

# Streamlit app configuration
st.set_page_config(
    page_title="RAG-Xpert", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i>RAG-Xpert : An Enhanced Retrieval-Augmented Generation Framework for Intelligent Document Processing and Knowledge Synthesis</i> 🤖💬</h2>""")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

# --- Sidebar ---
with st.sidebar:
    # Dynamic Color Theme for Light & Dark Modes
    st.markdown("""
        <style>
            .dev-credit {
                text-align: center;
                padding: 12px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                background-color: #007BFF;  /* Blue Background */
                color: white;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            @media (prefers-color-scheme: dark) {
                .dev-credit {
                    background-color: #1E1E1E; /* Dark Gray Background */
                    color: #FFD700; /* Gold Text */
                }
            }
        </style>
        <div class="dev-credit">👨‍💻 Developed by <br> <span>Uditanshu Pandey</span></div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # RAG Sources Section
    st.header("📂 RAG Sources")
    
    # File Upload
    st.file_uploader(
        "📄 Upload a document", 
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )

    # URL Input
    st.text_input(
        "🌐 Enter a URL", 
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )

    # Toggle for RAG Usage
    is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)

    st.toggle(
        "🔍 Use RAG", 
        value=is_vector_db_loaded, 
        key="use_rag", 
        disabled=not is_vector_db_loaded,
    )

    # Clear Chat Button
    st.button("🗑️ Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

    # Documents in DB
    with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
        st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

# --- Main Chat App ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        if not st.session_state.use_rag:
            st.write_stream(llm_groq.invoke(messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_groq, messages))
