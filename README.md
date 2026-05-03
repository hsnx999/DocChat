# DocChat — RAG-Powered Document Q&A

A conversational AI app that lets you upload any PDF and ask
questions about it in plain English. Built from scratch using
Retrieval-Augmented Generation (RAG).

## Demo
Upload a PDF → ask questions → get grounded, accurate answers
streamed in real time.

## How it works
1. **Load** — PDF is parsed into text page by page
2. **Chunk** — text is split into 1000-character overlapping segments
3. **Embed** — each chunk is converted to a vector using nomic-embed-text
4. **Store** — vectors saved in ChromaDB on disk
5. **Retrieve** — user question is embedded, top matching chunks found
6. **Generate** — chunks injected into prompt, Qwen streams the answer

## Tech stack
- LangChain — pipeline orchestration
- ChromaDB — local vector database
- Ollama (nomic-embed-text + Qwen2.5) — fully local, no API costs
- Streamlit — web UI
- PyPDF — PDF parsing

## Run it yourself

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:3b
streamlit run app.py
```

## What I learned
- How RAG works at the implementation level, not just conceptually
- Why chunk overlap matters for retrieval quality
- How vector similarity search finds semantically related text
- How to stream LLM responses token by token in a web UI