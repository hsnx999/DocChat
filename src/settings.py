# ── Document processing ─────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# ── Vector store ────────────────────────────────────────────
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── RAG chain ───────────────────────────────────────────────
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.0
RETRIEVE_K = 4
MAX_HISTORY_TURNS = 6
