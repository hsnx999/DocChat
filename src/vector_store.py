from typing import List, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import logging
from src.settings import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVE_K

logger = logging.getLogger(__name__)
_embeddings = None
_chroma_client = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def get_or_create_collection() -> chromadb.Collection:
    """
    Get existing collection or create a fresh one.
    Unlike before, we no longer delete it on every upload —
    we accumulate documents across multiple uploads.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return collection


def add_documents(
    collection: chromadb.Collection,
    chunks: List[Document],
    filename: str,
) -> int:
    """
    Embed chunks and add them to the existing collection.
    Each chunk is tagged with the source filename so answers
    can cite which document they came from.

    Returns the number of chunks added.
    """
    texts     = [chunk.page_content for chunk in chunks]
    # Use filename as the source tag — clean and readable
    metadatas = [{**chunk.metadata, "source": filename} for chunk in chunks]
    # IDs must be unique across the whole collection
    # Prefix with filename to avoid collisions across docs
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

    try:
        embeddings_model = get_embeddings()
        vectors = embeddings_model.embed_documents(texts)
        collection.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as e:
        logger.exception("Failed to index document '%s'", filename)
        raise RuntimeError(f"Failed to index document '{filename}': {e}")

    return len(chunks)


def remove_document(collection: chromadb.Collection, filename: str):
    """
    Delete all chunks belonging to a specific file.
    ChromaDB supports filtering deletes by metadata.
    """
    try:
        collection.delete(where={"source": filename})
    except Exception as e:
        logger.exception("Failed to remove document '%s'", filename)
        raise RuntimeError(f"Failed to remove document '{filename}': {e}")


def get_indexed_files(collection: chromadb.Collection) -> list[str]:
    """
    Return a list of unique filenames currently in the collection.
    """
    try:
        if collection.count() == 0:
            return []
        results = collection.get(include=["metadatas"])
    except Exception as e:
        logger.exception("Failed to list indexed files")
        raise RuntimeError(f"Failed to list indexed files: {e}")

    seen = set()
    for meta in results["metadatas"]:
        if "source" in meta:
            seen.add(meta["source"])
    return sorted(seen)


def query_vector_store(
    collection: chromadb.Collection,
    query: str,
    k: int = RETRIEVE_K,
    filter_sources: Optional[list[str]] = None,
) -> List[Document]:
    """
    Embed the query and find the k most similar chunks across all documents.
    If filter_sources is provided, only return chunks from those source filenames.
    """
    try:
        embeddings_model = get_embeddings()
        query_vector = embeddings_model.embed_query(query)

        # Cap k at the number of chunks in the collection
        n = min(k, collection.count())
        if n == 0:
            return []

        where_filter = None
        if filter_sources:
            where_filter = {"source": {"$in": filter_sources}}

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            where=where_filter,
        )
    except Exception as e:
        logger.exception("Failed to query vector store")
        raise RuntimeError(f"Failed to query vector store: {e}")

    docs = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

