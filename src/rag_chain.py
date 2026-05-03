from typing import Generator, List
import chromadb
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from src.vector_store import query_vector_store


SYSTEM_PROMPT = """You are a helpful assistant that answers questions
based strictly on the provided context below.

Rules:
- Only use information from the context to answer.
- If the answer is not in the context, say "I couldn't find that in the document."
- Be concise and direct.
- If asked to summarize, give a clear structured response.

Context:
{context}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_context(docs: List[Document]) -> str:
    """
    Join retrieved chunks into a single string for the prompt.
    Each chunk is numbered so the model can reference them.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} - page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    collection: chromadb.Collection,
    question: str,
) -> Generator[str, None, None]:
    """
    Full RAG pipeline: retrieve → format → prompt → stream.
    Yields string tokens as they come from the LLM.
    """
    # Step 1 — retrieve relevant chunks
    docs = query_vector_store(collection, question, k=4)

    # Step 2 — format them into a context string
    context = format_context(docs)

    # Step 3 — build the prompt
    prompt = PROMPT_TEMPLATE.format_messages(
        context=context,
        question=question,
    )

    # Step 4 — stream the response from Qwen
    llm = ChatOllama(model="qwen2.5-coder:3b", temperature=0.0)
    for chunk in llm.stream(prompt):
        yield chunk.content