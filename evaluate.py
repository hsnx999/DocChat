import os

# evaluate.py
# Run with: python evaluate.py
# Automatically picks the first PDF found in the data/ folder.

from dotenv import load_dotenv
load_dotenv()

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from datasets import Dataset
from src.settings import LLM_MODEL, EMBEDDING_MODEL, RETRIEVE_K
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.document_processor import load_pdf, chunk_documents
from src.vector_store import (
    get_or_create_collection,
    add_documents,
    get_indexed_files,
    query_vector_store,
)
from src.rag_chain import answer_question


# ── Test questions ─────────────────────────────────────────────────────────────
# Test questions covering the content in the sample "RAG Architecture" PDF.
# Update these if you change the document used for evaluation.
TEST_CASES = [
    {
        "question": "What programming languages are used in the tech stack?",
        "ground_truth": "The tech stack includes Python, TypeScript, and uses LangChain, HuggingFace, and Streamlit.",
    },
    {
        "question": "What is the background of the team that developed the project?",
        "ground_truth": "The project was developed by a team at the AI Research Lab, established in 2022, which has published papers on efficient RAG retrieval.",
    },
    {
        "question": "What project is described in the document?",
        "ground_truth": "The document describes a RAG-based document Q&A system that uses vector search with LLM generation.",
    },
    {
        "question": "What AI frameworks are mentioned?",
        "ground_truth": "The document mentions LangChain, ChromaDB, HuggingFace, Sentence Transformers, Cross-Encoders, and PyTorch.",
    },
    {
        "question": "How does the document indexing process work?",
        "ground_truth": "Documents are chunked into 1000-character segments with 200-character overlap, embedded using all-MiniLM-L6-v2 into a 384-dimensional vector, and stored in ChromaDB.",
    },
    {
        "question": "What databases are mentioned?",
        "ground_truth": "The databases mentioned are ChromaDB for vector storage, PostgreSQL for metadata, MongoDB for logging, and Redis for caching.",
    },
    {
        "question": "What is the tech stack for the evaluation tool?",
        "ground_truth": "The evaluation tool uses FastAPI for the backend API, React for the frontend dashboard, Streamlit for prototyping, and Python.",
    },
]


def find_pdf() -> tuple[str, str]:
    """
    Auto-detect the first PDF in the data/ folder.
    Returns (file_path, filename).
    Raises FileNotFoundError if no PDFs are found.
    """
    data_dir = Path("data")
    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            "No PDF files found in data/ folder. "
            "Upload a document via the Streamlit app first, "
            "or copy a PDF into the data/ directory manually."
        )
    # Use the most recently modified PDF if there are multiple
    pdfs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = pdfs[0]
    return str(chosen), chosen.name


def run_pipeline(collection, question: str) -> tuple[str, list[str]]:
    """Run one question through the full RAG pipeline."""
    docs = query_vector_store(collection, question, k=RETRIEVE_K)
    contexts = [doc.page_content for doc in docs]
    full_answer = ""
    for item in answer_question(collection, question, chat_history=[]):
        if item.get("type") == "token":
            full_answer += item.get("data", "")
    return full_answer, contexts


def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    logger.info("=" * 55)
    logger.info("  DocChat — RAGAS Evaluation")
    logger.info("=" * 55)

    # ── Auto-detect PDF ────────────────────────────────────
    logger.info("1. Finding document...")
    try:
        pdf_path, filename = find_pdf()
        logger.info("   Using: %s", filename)
    except FileNotFoundError as e:
        logger.error("   ERROR: %s", e)
        return

    # ── Load into collection ───────────────────────────────
    collection = get_or_create_collection()
    indexed = get_indexed_files(collection)
    if filename not in indexed:
        pages  = load_pdf(pdf_path)
        chunks = chunk_documents(pages)
        add_documents(collection, chunks, filename=filename)
        logger.info("   Indexed %d chunks", len(chunks))
    else:
        logger.info("   Already indexed — skipping embedding")

    # ── Run test cases ─────────────────────────────────────
    logger.info("2. Running %d test questions through pipeline...", len(TEST_CASES))
    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    for i, case in enumerate(TEST_CASES, start=1):
        logger.info("   [%d/%d] %s...", i, len(TEST_CASES), case['question'][:55])
        answer, context = run_pipeline(collection, case["question"])
        questions.append(case["question"])
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(case["ground_truth"])

    # ── Score with RAGAS ───────────────────────────────────
    logger.info("3. Scoring with RAGAS (this takes ~8 minutes)...")
    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    llm = LangchainLLMWrapper(
        ChatGroq(model=LLM_MODEL, temperature=0.0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=1, timeout=120),
    )

    # ── Print report ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RAGAS Evaluation Results")
    print("=" * 55)
    df = results.to_pandas()

    faith_col     = "faithfulness"      if "faithfulness"      in df.columns else None
    relevancy_col = "answer_relevancy"  if "answer_relevancy"  in df.columns else None
    precision_col = "context_precision" if "context_precision" in df.columns else None

    def fmt(col):
        val = df[col].mean()
        return f"{val:.3f}" if str(val) != "nan" else "n/a (rate limited)"

    print(f"\n  Faithfulness:      {fmt(faith_col) if faith_col else 'n/a'}")
    print(f"  Answer Relevancy:  {fmt(relevancy_col) if relevancy_col else 'n/a'}")
    print(f"  Context Precision: {fmt(precision_col) if precision_col else 'n/a'}")

    print("\n  Per-question breakdown:")
    print("-" * 55)
    for i, row in df.iterrows():
        print(f"\n  Q{i+1}: {questions[i][:52]}...")
        for label, col in [
            ("Faithfulness     ", faith_col),
            ("Answer Relevancy ", relevancy_col),
            ("Context Precision", precision_col),
        ]:
            if col:
                val = row[col]
                print(f"       {label}  {f'{val:.2f}' if str(val) != 'nan' else 'n/a'}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
