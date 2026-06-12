import os

# ── Provider configuration ─────────────────────────────────────────
PROVIDER_CONFIG = {
    "openai": {
        "name": "OpenAI",
        "key_env": "OPENAI_API_KEY",
        "base_url_default": "",
        "list_endpoint": "https://api.openai.com/v1/models",
        "class_path": ("langchain_openai", "ChatOpenAI"),
    },
    "groq": {
        "name": "Groq (Free Tier)",
        "key_env": "GROQ_API_KEY",
        "base_url_default": "https://api.groq.com/openai/v1",
        "list_endpoint": "https://api.groq.com/openai/v1/models",
        "class_path": ("langchain_groq", "ChatGroq"),
    },
    "opencode": {
        "name": "OpenCode Zen",
        "key_env": "OPENCODE_API_KEY",
        "base_url_default": "https://opencode.ai/zen/v1",
        "list_endpoint": "https://opencode.ai/zen/v1/models",
        "class_path": ("langchain_openai", "ChatOpenAI"),
    },
    "deepseek": {
        "name": "DeepSeek",
        "key_env": "DEEPSEEK_API_KEY",
        "base_url_default": "https://api.deepseek.com/v1",
        "list_endpoint": "https://api.deepseek.com/v1/models",
        "class_path": ("langchain_openai", "ChatOpenAI"),
    },
    "claude": {
        "name": "Anthropic Claude",
        "key_env": "ANTHROPIC_API_KEY",
        "base_url_default": "",
        "list_endpoint": "https://api.anthropic.com/v1/models",
        "class_path": ("langchain_anthropic", "ChatAnthropic"),
    },
    "gemini": {
        "name": "Google Gemini",
        "key_env": "GEMINI_API_KEY",
        "base_url_default": "",
        "list_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "class_path": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    },
    "ollama": {
        "name": "Ollama (Local)",
        "key_env": None,
        "base_url_default": "http://localhost:11434",
        "list_endpoint": None,
        "class_path": ("langchain_ollama", "ChatOllama"),
    },
}

DEFAULT_PROVIDER = "ollama"
LLM_MODEL = os.environ.get("DOCCHAT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")

# ── Document processing ────────────────────────────────────────────
CHUNK_SIZE = int(os.environ.get("DOCCHAT_CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("DOCCHAT_CHUNK_OVERLAP", 200))
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".txt": "Text",
    ".docx": "Word",
}

# ── Vector store ───────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = os.environ.get("DOCCHAT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── RAG chain ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = float(os.environ.get("DOCCHAT_TEMPERATURE", "0.0"))
RETRIEVE_K = int(os.environ.get("DOCCHAT_RETRIEVE_K", 6))
MAX_HISTORY_TURNS = 6
RERANKER_MODEL = os.environ.get("DOCCHAT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = 4

# ── Hybrid search ──────────────────────────────────────────────────
BM25_K = 10
HYBRID_ALPHA = 0.5

# ── Session isolation ──────────────────────────────────────────────
SESSION_CLEANUP_AGE_HOURS = 24
