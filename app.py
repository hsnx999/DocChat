from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.document_processor import process_uploaded_file
from src.vector_store import (
    get_or_create_collection,
    add_documents,
    remove_document,
    get_indexed_files,
)
from src.rag_chain import answer_question


st.set_page_config(page_title="DocChat", page_icon="📄")
st.title("📄 DocChat")
st.caption("Upload PDFs and ask questions across all of them — powered by RAG + LLaMA 3")


# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = get_or_create_collection()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Documents")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        indexed = get_indexed_files(st.session_state.collection)
        if uploaded_file.name not in indexed:
            with st.spinner(f"Embedding {uploaded_file.name}…"):
                chunks = process_uploaded_file(uploaded_file)
                add_documents(
                    st.session_state.collection,
                    chunks,
                    filename=uploaded_file.name,
                )
                st.session_state.messages = []  # reset chat on new doc
            st.success(f"✓ {uploaded_file.name} added")
        else:
            st.info(f"{uploaded_file.name} is already indexed.")

    # Show all indexed files with a remove button for each
    st.divider()
    indexed_files = get_indexed_files(st.session_state.collection)

    if indexed_files:
        st.markdown("**Indexed documents:**")
        for fname in indexed_files:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"📄 {fname}")
            if col2.button("✕", key=f"remove_{fname}"):
                remove_document(st.session_state.collection, fname)
                st.session_state.messages = []
                st.rerun()
    else:
        st.info("No documents indexed yet.")

    # Clear everything button
    if indexed_files:
        st.divider()
        if st.button("Clear all documents", use_container_width=True):
            for fname in indexed_files:
                remove_document(st.session_state.collection, fname)
            st.session_state.messages = []
            st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────
indexed_files = get_indexed_files(st.session_state.collection)

if not indexed_files:
    st.info("👈 Upload one or more PDFs in the sidebar to get started.")
else:
    # Show which docs are active
    st.caption(f"Searching across: {', '.join(indexed_files)}")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask something about your documents…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            for token in answer_question(
                st.session_state.collection,
                question,
                st.session_state.messages,
            ):
                full_response += token
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })