from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.document_processor import process_uploaded_file
from src.vector_store import build_vector_store
from src.rag_chain import answer_question


# ── Page config ────────────────────────────────────────────
st.set_page_config(page_title="DocChat", page_icon="📄")
st.title("📄 DocChat")
st.caption("Upload a PDF and ask questions about it — powered by RAG + Qwen")


# ── Session state ──────────────────────────────────────────
# These persist across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload your document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("Reading and embedding your document…"):
            chunks = process_uploaded_file(uploaded_file)
            collection = build_vector_store(chunks)
            st.session_state.collection = collection
            st.session_state.doc_name = uploaded_file.name
            st.session_state.messages = []  # reset chat on new doc
        st.success(f"✓ {len(chunks)} chunks ready from **{uploaded_file.name}**")

    if st.session_state.doc_name:
        st.info(f"Active: **{st.session_state.doc_name}**")
        if st.button("Clear & upload new", use_container_width=True):
            st.session_state.collection = None
            st.session_state.doc_name = None
            st.session_state.messages = []
            st.rerun()


# ── Main chat area ─────────────────────────────────────────
if not st.session_state.collection:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask something about your document…"):

        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            for token in answer_question(st.session_state.collection, question):
                full_response += token
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })