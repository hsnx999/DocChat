# DocChat — RAG-Powered Document Q&A Chatbot

A conversational AI app that lets you upload PDFs, text files, and Word
documents and ask questions across all of them in plain English. Answers
are grounded strictly in the documents — no hallucination from general
training data — and the pipeline is backed by automated RAGAS evaluation
scores.

Built from scratch as a portfolio project to learn production RAG architecture.

🔗 **[Live Demo →](https://hsnx999-rag-chatbot.streamlit.app)**

![CI](https://github.com/hsnx999/rag-chatbot/actions/workflows/eval.yml/badge.svg)
---

## Features

    Multi-format upload        Upload PDF, TXT, and DOCX files. Query across
                               all of them in a single chat session.

    Document subset filter     Choose which indexed documents to search against
                               for each question using a sidebar multiselect.

    Cross-encoder re-ranker    Retrieved chunks are re-scored by a cross-encoder
                               model, improving answer quality by surfacing the
                               most relevant passages first.

    Conversation memory        Follow-up questions work in context. A condensing
                               step rewrites vague follow-ups like "tell me more
                               about the second one" into standalone queries before
                               hitting the vector store.

    Message management         Edit your own messages, delete individual assistant
                               responses, and rate answers with thumbs up/down.

    Conversation persistence   Chat history survives page refreshes — stored as
                               a JSON file in the data/ directory.

    Streaming responses        Answers stream token by token in real time,
                               exactly like a chat interface.

    RAGAS evaluation           Automated pipeline scoring on faithfulness,
                               answer relevancy, and context precision.

---

## How it works

At upload time (runs once per document):

    1. Load    PyPDF / python-docx / plain-text reader parses the document
    2. Chunk   Text split into 1000-char segments with 200-char overlap
    3. Embed   Each chunk converted to a vector using all-MiniLM-L6-v2
    4. Store   Vectors saved to ChromaDB on disk, tagged with source filename

At query time (runs on every question):

    5. Condense   Chat history + question rewritten as a standalone query
    6. Retrieve   Condensed query embedded, top-k chunks found via cosine similarity
                  across selected documents (or all if none selected)
    7. Re-rank    Cross-encoder scores each retrieved chunk against the query;
                  only the top 4 most relevant pass to generation
    8. Generate   Selected chunks injected into prompt, LLaMA 3.1 streams the answer

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
| Metric | Score |
|---|---|---|
| Faithfulness | 0.889 |
| Answer Relevancy | 0.734 |
| Context Precision | 0.571 |

_Latest run: 2026-05-29 23:39 UTC_
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

Create a .env file:

    GROQ_API_KEY=your_key_here

Run the app:

    streamlit run app.py

Open http://localhost:8501, upload one or more documents, and start asking questions.

---

## Run with Docker

    docker compose up --build

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
    │   ├── document_processor.py   PDF/TXT/DOCX loading, chunking, upload handler
    │   ├── vector_store.py         ChromaDB operations: add, remove, query, list
    │   └── rag_chain.py            Question condensing, re-ranking, prompt templates, LLM streaming
    ├── requirements.txt
    └── .env.example

---

## What I learned building this

- How RAG works at the implementation level, not just conceptually
- Why chunk overlap matters for retrieval quality at chunk boundaries
- How vector similarity search finds semantically related text
  even when the exact words do not match
- How cross-encoder re-ranking improves retrieval precision over pure vector search
- How to implement conversation memory using a question condensing step
  so follow-up questions resolve correctly
- How to support multiple document formats (PDF, TXT, DOCX) with format-specific
  parsing and validation
- How to build multi-document retrieval with per-file metadata tagging
  and subset filtering at query time
- How to quantitatively evaluate a RAG pipeline using RAGAS metrics
- How to stream LLM responses token by token in a Streamlit UI
- How to write unit tests for RAG components with mocked dependencies
- Debugging Python 3.13 dependency conflicts across a complex
  ML library ecosystem
