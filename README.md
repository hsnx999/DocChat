# DocChat — RAG-Powered Document Q&A Chatbot

A conversational AI app that lets you upload multiple PDFs and ask questions
across all of them in plain English. Answers are grounded strictly in the
documents — no hallucination from general training data — and the pipeline
is backed by automated RAGAS evaluation scores.

Built from scratch as a portfolio project to learn production RAG architecture.

🔗 **[Live Demo →](https://hsnx999-rag-chatbot.streamlit.app)**


---

## Features

    Multi-document support   Upload several PDFs and query across all of them.
                             Each answer cites which file it came from.
                             Remove individual documents without clearing others.

    Conversation memory      Follow-up questions work in context. A condensing
                             step rewrites vague follow-ups like "tell me more
                             about the second one" into standalone queries before
                             hitting the vector store.

    Streaming responses      Answers stream token by token in real time,
                             exactly like a chat interface.

    RAGAS evaluation         Automated pipeline scoring on faithfulness,
                             answer relevancy, and context precision.

---

## Evaluation Results (RAGAS)

The pipeline was evaluated across 7 test questions on a real resume document.

    Metric               Score    What it measures
    Faithfulness         0.950    Answers grounded in context, not hallucinated
    Answer Relevancy     0.862    Responses directly address what was asked
    Context Precision    0.885    Retrieval surfaces the right chunks

Run the evaluation yourself:

    python evaluate.py

---

## How it works

At upload time (runs once per document):

    1. Load    PyPDF parses the document page by page
    2. Chunk   Text split into 1000-char segments with 200-char overlap
    3. Embed   Each chunk converted to a vector using all-MiniLM-L6-v2
    4. Store   Vectors saved to ChromaDB on disk, tagged with source filename

At query time (runs on every question):

    5. Condense   Chat history + question rewritten as a standalone query
    6. Retrieve   Condensed query embedded, top-k chunks found via cosine similarity
                  across all indexed documents
    7. Generate   Chunks injected into prompt, LLaMA 3.1 streams the answer

The 200-character overlap between chunks ensures answers that span chunk
boundaries are never missed.

---

## Tech stack

    Library                       Role
    LangChain                     Pipeline orchestration
    ChromaDB                      Local persistent vector database
    HuggingFace all-MiniLM-L6-v2  Lightweight local embeddings (no API cost)
    Groq LLaMA 3.1 8B             Fast, free LLM inference
    Streamlit                     Web UI with session state
    PyPDF                         PDF text extraction
    RAGAS                         Automated RAG evaluation metrics

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

Open http://localhost:8501, upload one or more PDFs, and start asking questions.

---

## Project structure

    rag-chatbot/
    ├── app.py                      Streamlit UI, session state, multi-doc sidebar
    ├── evaluate.py                 RAGAS evaluation script
    ├── src/
    │   ├── document_processor.py   PDF loading, chunking, Streamlit upload handler
    │   ├── vector_store.py         ChromaDB operations: add, remove, query, list
    │   └── rag_chain.py            Question condensing, prompt templates, LLM streaming
    ├── requirements.txt
    └── .env.example

---

## What I learned building this

- How RAG works at the implementation level, not just conceptually
- Why chunk overlap matters for retrieval quality at chunk boundaries
- How vector similarity search finds semantically related text
  even when the exact words do not match
- How to implement conversation memory using a question condensing step
  so follow-up questions resolve correctly
- How to build multi-document retrieval with per-file metadata tagging
- How to quantitatively evaluate a RAG pipeline using RAGAS metrics
- How to stream LLM responses token by token in a Streamlit UI
- Debugging Python 3.13 dependency conflicts across a complex
  ML library ecosystem