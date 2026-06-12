# DocChat: RAG-Powered Document Q&A Chatbot

DocChat is a production-grade RAG chatbot built on a hybrid retrieval
pipeline (BM25 + vector search + RRF fusion) with cross-encoder re-ranking,
conversation memory, and automated RAGAS evaluation.

A conversational AI app that lets you upload PDFs, text files, and Word
documents and ask questions across all of them in plain English. Answers
are grounded strictly in the documents, no hallucination from general
training data, and the pipeline is backed by automated RAGAS evaluation
scores.

🔗 **[Live Demo →](https://hsnx999-rag-chatbot.streamlit.app)**

---

## Features

    Multi-format upload        Upload PDF, TXT, and DOCX files or paste a URL
                               to fetch web pages and PDFs. Query across all of
                               them in a single chat session.

    Document subset filter     Choose which indexed documents to search against
                               for each question using a sidebar multiselect.

    Cross-encoder re-ranker    Retrieved chunks are re-scored by a cross-encoder
                               model, improving answer quality by surfacing the
                               most relevant passages first.

    Hybrid retrieval           BM25 keyword search fuses with vector similarity
                               via Reciprocal Rank Fusion, improving accuracy on
                               documents with specific names, numbers, or dates.

    Document TL;DR             Every uploaded document gets an instant 5-bullet
                               AI summary so you know what was indexed.

    URL ingestion              Paste any webpage or PDF URL and chat with it
                               instantly, no file download needed.

    Conversation memory        Follow-up questions work in context. A condensing
                               step rewrites vague follow-ups like "tell me more
                               about the second one" into standalone queries before
                               hitting the vector store.

    Message management         Edit your own messages, delete individual assistant
                               responses, and rate answers with thumbs up/down.

    Conversation persistence   Chat history survives page refreshes, stored as
                               a JSON file in the data/ directory.

    Streaming responses        Answers stream token by token in real time,
                               exactly like a chat interface.

    RAGAS evaluation           Automated pipeline scoring on faithfulness,
                               answer relevancy, and context precision.

---

## Architecture decisions

- **Hybrid retrieval (BM25 + vector + RRF) over pure vector search**: keyword
  matches for names, dates, and IDs that embeddings miss
- **Cross-encoder re-ranking as a second pass**: cheaper than re-embedding,
  meaningfully improves precision on top-k results
- **Question condensing before retrieval**: follow-up questions like
  "what about the second clause?" resolve correctly against the vector store
- **Chunk overlap (200 chars)**: prevents answers spanning chunk boundaries
  from being silently missed

---

## How it works

At upload time (runs once per document, including URLs):

    1. Load    PyPDF / python-docx / plain-text reader for files, or WebBaseLoader /
               urllib for URLs (PDF and HTML supported)
    2. Chunk   Text split into 1000-char segments with 200-char overlap
    3. Embed   Each chunk converted to a vector using all-MiniLM-L6-v2
    4. Summarize  A 5-bullet TL;DR is generated and shown in the chat
    5. Store   Vectors saved to ChromaDB on disk, tagged with source filename

At query time (runs on every question):

     6. Condense   Chat history + question rewritten as a standalone query
     7. Retrieve   Condensed query embedded, top-6 chunks found via cosine similarity
                   combined with BM25 keyword search (RRF fusion). Searches across
                   selected documents only (or all if none selected).
     8. Re-rank    Cross-encoder scores each retrieved chunk against the query;
                   only the top 4 most relevant pass to generation
     9. Generate   Selected chunks injected into prompt, LLaMA 3.1 streams the answer

The 200-character overlap between chunks ensures answers that span chunk
boundaries are never missed.

---

## Evaluation Results

| Metric | Ideal Range | What it measures |
|---|---|---|
| **Faithfulness** | ≥ 0.80 | Whether answers stay true to the source document |
| **Answer Relevancy** | ≥ 0.70 | Whether answers actually address the question asked |
| **Context Precision** | ≥ 0.70 | Whether the retrieved context is relevant and concise |

<!-- EVAL_SCORES -->
**RAGAS Evaluation**, updated 2026-06-12 05:49 UTC

| Metric | Score |
|---|---|
| Faithfulness | 0.880 |
| Answer Relevancy | 0.713 |
| Context Precision | 0.857 |

_Higher is better (0.0 – 1.0). Run `python evaluate.py` to re-score on your own document._
<!-- END_EVAL_SCORES -->

Run the evaluation yourself:

    python evaluate.py

Test cases cover RAG architecture concepts. Swap the PDF and questions to score your own document.

[![CI](https://github.com/hsnx999/rag-chatbot/actions/workflows/eval.yml/badge.svg)](https://github.com/hsnx999/rag-chatbot/actions/workflows/eval.yml)


---

## Tech stack

    Library                       Role
    LangChain                     Pipeline orchestration
    ChromaDB                      Local persistent vector database
    rank-bm25                     BM25 keyword search for hybrid retrieval
    beautifulsoup4 / lxml         HTML parsing for URL web page ingestion
    HuggingFace all-MiniLM-L6-v2  Lightweight local embeddings (no API cost)
    Cross-Encoder MiniLM-L6-v2    Re-ranks retrieved chunks for precision
    Groq LLaMA 3.1 8B             Fast, free LLM inference
    Streamlit                     Web UI with session state
    PyPDF / python-docx           PDF and Word text extraction
    RAGAS                         Automated RAG evaluation metrics
    pytest                        Unit and integration test suite

---

## Run it locally

Prerequisites: Python 3.10+ and a free Groq API key from console.groq.com

Clone and set up:

    git clone https://github.com/hsnx999/rag-chatbot.git
    cd rag-chatbot
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Create a .env file from the template:

    cp .env.example .env

Run the app:

    streamlit run app.py

Open http://localhost:8501, upload one or more documents, and start asking questions.

---

## Run with Docker or Podman

    docker compose up --build          # Docker
    podman compose up --build          # Podman

Open http://localhost:8501. The `data/` and `chroma_db/` directories are
persisted as Docker volumes.

---

## Run tests

    source .venv/bin/activate
    python -m pytest tests/ -v

---

## Project structure

    rag-chatbot/
    ├── app.py                      Streamlit UI, session state, multi-doc sidebar
    ├── evaluate.py                 RAGAS evaluation script
    ├── Dockerfile                  Container image definition
    ├── docker-compose.yml          Local deployment with volumes
    ├── tests/
    │   ├── conftest.py             Shared test fixtures
    │   ├── test_document_processor.py
    │   ├── test_rag_chain.py
    │   └── test_vector_store.py
    ├── src/
    │   ├── settings.py             Centralized configuration constants
    │   ├── document_processor.py   PDF/TXT/DOCX/URL loading, chunking, upload handler
    │   ├── vector_store.py         ChromaDB operations: add, remove, query, BM25+RRF hybrid
    │   └── rag_chain.py            Question condensing, re-ranking, prompt templates, LLM streaming
    ├── scripts/
    │   └── update_readme_scores.py CI helper for auto-updating evaluation scores
    ├── .github/workflows/eval.yml  CI pipeline: tests → RAGAS eval → update README
    ├── requirements.txt
    └── .env.example
