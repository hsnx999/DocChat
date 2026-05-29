from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.document_processor import process_uploaded_file
from src.vector_store import (
    get_or_create_collection,
    add_documents,
    remove_document,
    get_indexed_files,
    query_vector_store,
)
from src.rag_chain import answer_question
import os
import json
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

CHAT_HISTORY_FILE = Path("data/chat_history.json")


def _load_chat():
    if CHAT_HISTORY_FILE.exists():
        try:
            data = json.loads(CHAT_HISTORY_FILE.read_text())
            if isinstance(data, list):
                st.session_state.messages = data
        except Exception:
            pass


def _save_chat():
    CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHAT_HISTORY_FILE.write_text(json.dumps(st.session_state.messages, indent=2))


st.set_page_config(page_title="DocChat", page_icon="📄")
st.title("📄 DocChat")
st.caption("Upload PDFs and ask questions across all of them — powered by RAG + LLaMA 3")

# ── Startup validation ─────────────────────────────────────────────────────
if "GROQ_API_KEY" not in os.environ:
    st.error(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file or environment variables, then restart the app."
    )
    st.stop()


# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_loaded" not in st.session_state:
    _load_chat()
    st.session_state.chat_loaded = True
if "collection" not in st.session_state:
    st.session_state.collection = get_or_create_collection()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = get_indexed_files(st.session_state.collection)
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = None


def refresh_indexed_files():
    st.session_state.indexed_files = get_indexed_files(st.session_state.collection)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Documents")

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])

    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner(f"Embedding {uploaded_file.name}…"):
                try:
                    chunks = process_uploaded_file(uploaded_file)
                    add_documents(
                        st.session_state.collection,
                        chunks,
                        filename=uploaded_file.name,
                    )
                    st.session_state.messages = []
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"✓ {uploaded_file.name} added")
                    refresh_indexed_files()
                except ValueError as e:
                    logger.exception("Upload validation failed")
                    st.error(str(e))
                except Exception as e:
                    logger.exception("Upload processing failed")
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
        else:
            st.info(f"{uploaded_file.name} is already indexed.")

    st.divider()
    indexed_files = st.session_state.indexed_files

    if indexed_files:
        st.markdown("**Indexed documents:**")
        for fname in indexed_files:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"📄 {fname}")
            if col2.button("✕", key=f"remove_{fname}"):
                remove_document(st.session_state.collection, fname)
                st.session_state.messages = []
                st.session_state.processed_files.discard(fname)
                refresh_indexed_files()
                st.rerun()
    else:
        st.info("No documents indexed yet.")

    if indexed_files:
        st.divider()
        with st.popover("Clear all documents", use_container_width=True):
            st.warning("This will permanently delete all indexed documents.")
            if st.button("Yes, clear everything"):
                for fname in indexed_files:
                    remove_document(st.session_state.collection, fname)
                st.session_state.messages = []
                st.session_state.processed_files.clear()
                refresh_indexed_files()
                st.rerun()

        st.divider()
        selected = st.multiselect(
            "Search only in:",
            options=indexed_files,
            default=indexed_files,
            label_visibility="collapsed",
        )
        st.session_state.selected_docs = selected if selected else None


# ── Main chat area ─────────────────────────────────────────────────────────
indexed_files = st.session_state.indexed_files

if not indexed_files:
    st.info("👈 Upload one or more PDFs in the sidebar to get started.")
else:
    st.caption(f"Searching across: {', '.join(indexed_files)}")

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                col1, col2 = st.columns([10, 1])
                with col1:
                    if st.session_state.get(f"editing_{i}"):
                        new_content = st.text_area(
                            "Edit message",
                            msg["content"],
                            key=f"edit_input_{i}",
                            label_visibility="collapsed",
                        )
                        save_col, cancel_col = st.columns([1, 1])
                        with save_col:
                            if st.button("Save", key=f"save_{i}"):
                                st.session_state.messages[i]["content"] = new_content
                                st.session_state[f"editing_{i}"] = False
                                _save_chat()
                                st.rerun()
                        with cancel_col:
                            if st.button("Cancel", key=f"cancel_{i}"):
                                st.session_state[f"editing_{i}"] = False
                                st.rerun()
                    else:
                        st.markdown(msg["content"])
                with col2:
                    if st.button("✎", key=f"edit_btn_{i}"):
                        st.session_state[f"editing_{i}"] = True
                        st.rerun()
            else:
                st.markdown(msg["content"])
                feedback = msg.get("feedback")
                fcol1, fcol2, fcol3 = st.columns([1, 1, 10])
                with fcol1:
                    up_disabled = feedback is not None
                    if st.button("👍", key=f"up_{i}", disabled=up_disabled):
                        st.session_state.messages[i]["feedback"] = "up"
                        _save_chat()
                        st.rerun()
                with fcol2:
                    down_disabled = feedback is not None
                    if st.button("👎", key=f"down_{i}", disabled=down_disabled):
                        st.session_state.messages[i]["feedback"] = "down"
                        _save_chat()
                        st.rerun()
                with fcol3:
                    if st.button("Delete", key=f"del_{i}"):
                        st.session_state.messages = st.session_state.messages[:i]
                        _save_chat()
                        st.rerun()

    if question := st.chat_input("Ask something about your documents…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            placeholder.markdown("Thinking...")

            try:
                for token in answer_question(
                    st.session_state.collection,
                    question,
                    st.session_state.messages,
                    filter_sources=st.session_state.selected_docs,
                ):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            except Exception as e:
                logger.exception("Question answering failed")
                full_response = f"Sorry, something went wrong: {e}"
            finally:
                placeholder.markdown(full_response)

            if indexed_files:
                with st.expander("Sources"):
                    try:
                        source_docs = query_vector_store(
                            st.session_state.collection, question, k=3
                        )
                        for i, doc in enumerate(source_docs):
                            filename = doc.metadata.get("source", "document")
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**{i+1}. {filename} (page {page})**")
                            st.caption(doc.page_content[:300])
                    except Exception:
                        pass

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })
        _save_chat()

        if len(st.session_state.messages) > 100:
            st.session_state.messages = st.session_state.messages[-100:]
