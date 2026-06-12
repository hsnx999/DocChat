from dotenv import load_dotenv
load_dotenv()

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import streamlit as st
from src.document_processor import process_uploaded_file, load_url, chunk_documents
from src.vector_store import (
    get_or_create_collection,
    add_documents,
    remove_document,
    get_indexed_files,
    delete_collection,
    list_session_collections,
)
from src.rag_chain import answer_question, generate_document_summary, get_installed_models, list_provider_models
from src.settings import SESSION_CLEANUP_AGE_HOURS, PROVIDER_CONFIG, DEFAULT_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, RETRIEVE_K
import json
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

BASE_DATA_DIR = Path("data")


def _chat_history_file() -> Path:
    return BASE_DATA_DIR / st.session_state.session_id / "chat_history.json"


def _last_active_file() -> Path:
    return BASE_DATA_DIR / st.session_state.session_id / ".last_active"


def _touch_last_active():
    _last_active_file().parent.mkdir(parents=True, exist_ok=True)
    _last_active_file().write_text(datetime.now(timezone.utc).isoformat())


def _load_chat():
    f = _chat_history_file()
    if f.exists():
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                st.session_state.messages = data
        except Exception:
            pass


def _save_chat():
    f = _chat_history_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(st.session_state.messages, indent=2))
    _touch_last_active()


def _cleanup_stale_sessions():
    """Delete sessions inactive for more than SESSION_CLEANUP_AGE_HOURS hours."""
    if not BASE_DATA_DIR.exists():
        return
    cutoff = datetime.now(timezone.utc).timestamp() - SESSION_CLEANUP_AGE_HOURS * 3600
    for d in BASE_DATA_DIR.iterdir():
        if not d.is_dir():
            continue
        last_active_file = d / ".last_active"
        if not last_active_file.exists():
            continue
        try:
            ts = datetime.fromisoformat(last_active_file.read_text().strip()).timestamp()
            if ts < cutoff:
                sid = d.name
                logger.info("Cleaning up stale session '%s'", sid)
                delete_collection(f"session_{sid}")
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            continue
    # Orphan ChromaDB collections with no matching data dir
    for col_name in list_session_collections():
        sid = col_name.removeprefix("session_")
        if not (BASE_DATA_DIR / sid).exists():
            logger.info("Cleaning up orphan collection '%s'", col_name)
            delete_collection(col_name)


st.set_page_config(page_title="DocChat", page_icon="📄")
st.title("📄 DocChat")
st.caption("Upload PDFs and ask questions across all of them — powered by RAG + LLaMA 3")

st.markdown(
    """
<style>
    /* Responsive viewport */
    .main > div { max-width: 100%; padding: 0 0.5rem; }

    /* Prevent collapsed sidebar toggle from covering title */
    .stApp .main .block-container { padding-left: 4rem !important; }

        /* Sidebar: collapse on narrow screens */
    @media (max-width: 768px) {
        /* Prevent horizontal scroll that shifts sidebar */
        html, body, .stApp { overflow-x: hidden !important; }

        /* Pull main content left — sidebar toggle is smaller on mobile */
        .stApp .main .block-container { padding-left: 3.5rem !important; }

        /* Lock sidebar in place */
        section[data-testid="stSidebar"] {
            overflow-x: hidden !important;
            min-width: 0 !important;
        }

        /* Main content padding */
        .main .block-container {
            padding: 0.5rem 0.5rem !important;
        }

        /* Smaller title on mobile */
        h1 { font-size: 1.3rem !important; }
        .stCaption { font-size: 0.75rem !important; }

        /* Touch-friendly buttons */
        button[kind="secondary"], button[kind="primary"] {
            min-height: 44px !important;
            font-size: 16px !important;
        }

        /* Prevent iOS zoom on text inputs */
        input, textarea, select { font-size: 16px !important; }

        /* Readable font sizes */
        div[data-testid="stMarkdownContainer"] {
            font-size: 15px !important;
        }

        /* Larger chat input on mobile */
        .stChatInput textarea {
            font-size: 16px !important;
            padding: 0.75rem !important;
        }

        /* Compact button columns on mobile */
        div[data-testid="column"] button {
            min-height: 44px !important;
            min-width: 36px !important;
            padding: 0.25rem 0.5rem !important;
        }

        /* Chat message bubbles — reduce padding */
        .stChatMessage {
            padding: 0.5rem 0.75rem !important;
        }

        /* Source citation box full-width */
        .stMarkdown div[style*="border-radius"] {
            font-size: 12px !important;
            padding: 4px 8px !important;
        }
    }

    /* Touch-friendly button sizing */
    button {
        min-height: 38px;
    }
    .stButton button {
        width: 100%;
    }
    div[data-testid="column"] button {
        min-width: 44px;
    }

    /* Source expander readable on mobile */
    .streamlit-expanderContent {
        font-size: 14px;
    }
    .streamlit-expanderContent p {
        font-size: 14px;
    }

    /* Hide Streamlit default header bar */
    header[data-testid="stHeader"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session ID ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    sid = st.query_params.get("session_id", "")
    if not sid or not sid.strip():
        sid = str(uuid.uuid4())
    st.session_state.session_id = sid
    st.query_params["session_id"] = sid
    _cleanup_stale_sessions()

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_loaded" not in st.session_state:
    _load_chat()
    st.session_state.chat_loaded = True
if "collection" not in st.session_state:
    st.session_state.collection = get_or_create_collection(st.session_state.session_id)
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = get_indexed_files(st.session_state.collection)
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = None
if "installed_models" not in st.session_state:
    st.session_state.installed_models = get_installed_models()
if "selected_model" not in st.session_state:
    models = st.session_state.installed_models
    st.session_state.selected_model = models[0] if models else LLM_MODEL
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = DEFAULT_PROVIDER
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}
if "provider_models" not in st.session_state:
    st.session_state.provider_models = {}


def refresh_indexed_files():
    st.session_state.indexed_files = get_indexed_files(st.session_state.collection)


@st.dialog("Settings", width="large")
def _render_settings_dialog():
    provider_names = list(PROVIDER_CONFIG.keys())
    current = provider_names.index(st.session_state.selected_provider) if st.session_state.selected_provider in provider_names else 0
    sel_prov = st.selectbox(
        "Select provider",
        provider_names,
        index=current,
        format_func=lambda p: PROVIDER_CONFIG[p]["name"],
        label_visibility="collapsed",
    )
    st.session_state.selected_provider = sel_prov
    cfg = PROVIDER_CONFIG[sel_prov]

    if cfg["key_env"]:
        st.markdown(f"#### {cfg['name']} API Key")
        current_key = st.session_state.api_keys.get(sel_prov, "")
        api_key = st.text_input(
            "Enter your API key",
            type="password",
            value=current_key,
            placeholder=f"sk-...",
            label_visibility="collapsed",
        )
        st.session_state.api_keys[sel_prov] = api_key

        if st.button(
            "Fetch Models",
            key=f"fetch_{sel_prov}",
            use_container_width=True,
            disabled=not api_key,
        ):
            with st.spinner("Fetching models..."):
                models = list_provider_models(sel_prov, api_key)
            st.session_state.provider_models[sel_prov] = models
            if not models:
                st.error("No models found. Check your API key.")

        if api_key and st.session_state.provider_models.get(sel_prov):
            st.success(f"{len(st.session_state.provider_models[sel_prov])} models available")
        elif api_key:
            st.caption("Click Fetch Models to load available models")

    if sel_prov == "ollama":
        current_base = st.session_state.api_keys.get("ollama_base_url", "http://localhost:11434")
        base_url = st.text_input("Ollama Base URL", value=current_base, label_visibility="collapsed")
        st.session_state.api_keys["ollama_base_url"] = base_url

    st.divider()
    st.markdown("#### Model")
    if sel_prov == "ollama":
        model_options = st.session_state.installed_models
    else:
        model_options = st.session_state.provider_models.get(sel_prov, [])

    if model_options:
        mi = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        selected = st.selectbox(
            "Choose a model",
            model_options,
            index=mi,
            label_visibility="collapsed",
        )
        st.session_state.selected_model = selected
    elif cfg["key_env"] and not st.session_state.api_keys.get(sel_prov):
        st.caption("Enter an API key and click Fetch Models")
    else:
        st.caption("No models available")

    st.divider()
    st.markdown("#### Pipeline")
    st.number_input("Retrieve K", value=RETRIEVE_K, min_value=1, max_value=20, key="override_k")
    st.slider("Temperature", 0.0, 2.0, LLM_TEMPERATURE, 0.1, key="override_temp")


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("⚙️ Settings", use_container_width=True):
        _render_settings_dialog()

    # ── Upload ──
    st.markdown("#### 📤 Upload")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"],
                                     label_visibility="collapsed")

    url_input = st.text_input(
        "Or paste a URL",
        key="url_input",
        placeholder="https://...",
        label_visibility="collapsed",
    )

    if url_input and url_input not in st.session_state.processed_files:
        url_display = url_input.split("/")[-1][:50] or url_input.split("/")[-2][:50]
        with st.spinner(f"Loading {url_display}…"):
            try:
                pages = load_url(url_input)
                chunks = chunk_documents(pages)
                url_name = url_input.split("//")[-1].split("/")[0][:50]
                add_documents(st.session_state.collection, chunks, filename=url_name)
                st.session_state.messages = []
                st.session_state.processed_files.add(url_input)
                st.success(f"✓ {url_name} indexed")
                refresh_indexed_files()

                with st.spinner("Generating summary…"):
                    summary = generate_document_summary(
                        chunks,
                        model_name=st.session_state.selected_model,
                        provider=st.session_state.selected_provider,
                        api_key=st.session_state.api_keys.get(st.session_state.selected_provider, ""),
                    )
                    if summary:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"📄 **{url_name}**\n\n{summary}"
                        })
                        _save_chat()
            except Exception as e:
                st.error(f"Failed to load URL: {e}")

    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner(f"Embedding {uploaded_file.name}…"):
                try:
                    chunks = process_uploaded_file(uploaded_file, session_id=st.session_state.session_id)
                    add_documents(st.session_state.collection, chunks, filename=uploaded_file.name)
                    st.session_state.messages = []
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"✓ {uploaded_file.name} added")
                    refresh_indexed_files()

                    with st.spinner("Generating summary…"):
                        summary = generate_document_summary(
                            chunks,
                            model_name=st.session_state.selected_model,
                            provider=st.session_state.selected_provider,
                            api_key=st.session_state.api_keys.get(st.session_state.selected_provider, ""),
                        )
                        if summary:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"📄 **{uploaded_file.name}**\n\n{summary}"
                            })
                            _save_chat()
                except ValueError as e:
                    logger.exception("Upload validation failed")
                    st.error(str(e))
                except Exception as e:
                    logger.exception("Upload processing failed")
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
        else:
            st.info(f"{uploaded_file.name} is already indexed.")

    # ── Documents ──
    indexed_files = st.session_state.indexed_files
    if indexed_files:
        st.divider()
        st.markdown("#### 📚 Your Documents")
        for fname in indexed_files:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {fname}")
            if col2.button("✕", key=f"remove_{fname}"):
                remove_document(st.session_state.collection, fname)
                st.session_state.messages = []
                st.session_state.processed_files.discard(fname)
                refresh_indexed_files()
                st.rerun()

        with st.popover("Clear all documents", use_container_width=True):
            st.warning("This will permanently delete all indexed documents.")
            if st.button("Yes, clear everything"):
                for fname in indexed_files:
                    remove_document(st.session_state.collection, fname)
                st.session_state.messages = []
                st.session_state.processed_files.clear()
                if url_input:
                    st.session_state.processed_files.add(url_input)
                refresh_indexed_files()
                st.rerun()

        # ── Filter ──
        st.divider()
        st.markdown("#### 🔍 Search Filter")
        selected = st.multiselect(
            "Only search in:",
            options=indexed_files,
            default=indexed_files,
            label_visibility="visible",
        )
        st.session_state.selected_docs = selected if selected else None

    # ── Feedback stats ──
    rated_msgs = [m for m in st.session_state.messages if m.get("feedback") and m["role"] == "assistant"]
    if rated_msgs:
        st.divider()
        st.markdown("#### 📊 Feedback")
        ups = sum(1 for m in rated_msgs if m["feedback"] == "up")
        downs = sum(1 for m in rated_msgs if m["feedback"] == "down")
        total = ups + downs
        pct = round(ups / total * 100) if total > 0 else 0
        st.caption(f"👍 {ups}  ·  👎 {downs}  —  {pct}% helpful")


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
                                st.session_state.messages = st.session_state.messages[:i+1]
                                st.session_state[f"editing_{i}"] = False
                                st.session_state.pending_question = new_content
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
                sources = msg.get("sources")
                if sources:
                    source_html = '<div style="margin-top: 8px; margin-bottom: 8px; padding: 6px 10px; background: rgba(128,128,128,0.1); border-radius: 8px; font-size: 13px;">'
                    source_html += '<strong>Sources:</strong><br>'
                    for j, s in enumerate(sources, start=1):
                        source_html += f'<sup>{j}</sup> {s["filename"]} (page {s["page"]})<br>'
                    source_html += '</div>'
                    st.markdown(source_html, unsafe_allow_html=True)
                feedback = msg.get("feedback")
                fcol_spacer, fcol1, fcol2, fcol3 = st.columns([8, 1, 1, 1])
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
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.messages = st.session_state.messages[:i-1]
                        _save_chat()
                        st.rerun()

    pending = st.session_state.pop("pending_question", None)
    question = pending or st.chat_input("Ask something about your documents…")

    if question:
        if not pending:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            source_docs = []
            full_response = ""
            placeholder.markdown("Thinking...")

            try:
                for item in answer_question(
                    st.session_state.collection,
                    question,
                    st.session_state.messages,
                    filter_sources=st.session_state.selected_docs,
                    model_name=st.session_state.selected_model,
                    provider=st.session_state.selected_provider,
                    api_key=st.session_state.api_keys.get(st.session_state.selected_provider, ""),
                ):
                    if item.get("type") == "sources":
                        source_docs = item.get("data", [])
                        continue
                    token = item.get("data", "")
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            except Exception as e:
                logger.exception("Question answering failed")
                full_response = f"Sorry, something went wrong: {e}"
            finally:
                placeholder.markdown(full_response)

            # Save the message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_docs,
            })
            _save_chat()

            if len(st.session_state.messages) > 100:
                st.session_state.messages = st.session_state.messages[-100:]

            # Render sources inline inside the same chat bubble
            if source_docs:
                source_html = '<div style="margin-top: 8px; margin-bottom: 8px; padding: 6px 10px; background: rgba(128,128,128,0.1); border-radius: 8px; font-size: 13px;">'
                source_html += '<strong>Sources:</strong><br>'
                for j, s in enumerate(source_docs, start=1):
                    source_html += f'<sup>{j}</sup> {s["filename"]} (page {s["page"]})<br>'
                source_html += '</div>'
                st.markdown(source_html, unsafe_allow_html=True)

            # Render feedback and delete buttons inline
            idx = len(st.session_state.messages) - 1
            try:
                fcol_spacer, fcol1, fcol2, fcol3 = st.columns([8, 1, 1, 1])
                with fcol1:
                    up_disabled = st.session_state.messages[idx].get("feedback") is not None
                    if st.button("👍", key=f"up_inline_{idx}", disabled=up_disabled):
                        st.session_state.messages[idx]["feedback"] = "up"
                        _save_chat()
                        st.rerun()
                with fcol2:
                    down_disabled = st.session_state.messages[idx].get("feedback") is not None
                    if st.button("👎", key=f"down_inline_{idx}", disabled=down_disabled):
                        st.session_state.messages[idx]["feedback"] = "down"
                        _save_chat()
                        st.rerun()
                with fcol3:
                    if st.button("🗑️", key=f"del_inline_{idx}"):
                        st.session_state.messages = st.session_state.messages[:idx]
                        _save_chat()
                        st.rerun()
            except RuntimeError:
                pass  # SessionInfo race on Streamlit Cloud, state persisted anyway
