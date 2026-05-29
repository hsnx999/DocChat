from typing import List, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import logging
from src.settings import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVE_K, BM25_K, HYBRID_ALPHA

logger = logging.getLogger(__name__)
_embeddings = None
_chroma_client = None
_bm25_index = None
_bm25_cache_key = None


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


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    import re
    return re.findall(r"\w+", text.lower())


def _get_bm25_cache_key(collection: chromadb.Collection) -> str:
    """Derive a cache key from collection name + count to detect changes."""
    return f"{collection.name}_{collection.count()}"


def _build_bm25_index(collection: chromadb.Collection):
    """Build or rebuild the BM25 index from all documents in the collection."""
    global _bm25_index, _bm25_cache_key
    key = _get_bm25_cache_key(collection)
    if _bm25_index is not None and _bm25_cache_key == key:
        return
    try:
        from rank_bm25 import BM25Okapi
        all_docs = collection.get(include=["documents", "metadatas"])
        if not all_docs["documents"]:
            _bm25_index = None
            _bm25_cache_key = key
            return
        tokenized_corpus = [_tokenize(doc) for doc in all_docs["documents"]]
        _bm25_index = {
            "bm25": BM25Okapi(tokenized_corpus),
            "documents": all_docs["documents"],
            "metadatas": all_docs["metadatas"],
        }
        _bm25_cache_key = key
        logger.info("Rebuilt BM25 index (%d docs)", len(tokenized_corpus))
    except ImportError:
        logger.warning("rank_bm25 not installed -- hybrid search unavailable")
        _bm25_index = None
        _bm25_cache_key = key
    except Exception:
        logger.exception("Failed to build BM25 index")
        _bm25_index = None
        _bm25_cache_key = key


def _rrf_fuse(
    vector_docs: list,
    bm25_docs: list,
    alpha: float = HYBRID_ALPHA,
) -> list:
    """Fuse two ranked lists using weighted Reciprocal Rank Fusion."""
    scores = {}
    for rank, doc in enumerate(vector_docs):
        doc_id = doc.page_content[:80]
        scores[doc_id] = scores.get(doc_id, 0.0) + alpha * (1.0 / (rank + 1))
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.page_content[:80]
        scores[doc_id] = scores.get(doc_id, 0.0) + (1 - alpha) * (1.0 / (rank + 1))
    seen = set()
    merged = []
    for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1]):
        for doc in vector_docs + bm25_docs:
            if doc.page_content[:80] == doc_id and doc_id not in seen:
                merged.append(doc)
                seen.add(doc_id)
                break
    return merged


def query_bm25(
    collection: chromadb.Collection,
    query: str,
    k: int = BM25_K,
    filter_sources: Optional[list[str]] = None,
) -> list:
    """Retrieve documents using BM25 keyword search."""
    _build_bm25_index(collection)
    if _bm25_index is None:
        return []
    tokenized_query = _tokenize(query)
    scores = _bm25_index["bm25"].get_scores(tokenized_query)
    indexed = list(zip(scores, _bm25_index["documents"], _bm25_index["metadatas"]))
    indexed.sort(key=lambda x: -x[0])
    docs = []
    for score, text, meta in indexed:
        if filter_sources and meta.get("source") not in filter_sources:
            continue
        docs.append(Document(page_content=text, metadata=meta))
        if len(docs) >= k:
            break
    return docs


def get_or_create_collection(session_id: str = "") -> chromadb.Collection:
    """
    Get existing collection or create a fresh one.
    If a session_id is provided, use a session-scoped collection name.
    Otherwise, use the default COLLECTION_NAME.
    """
    client = get_chroma_client()
    name = f"session_{session_id}" if session_id else COLLECTION_NAME
    collection = client.get_or_create_collection(name)
    return collection


def delete_collection(name: str):
    """Delete a collection by name. Silently ignore if not found."""
    try:
        client = get_chroma_client()
        client.delete_collection(name)
    except Exception:
        logger.warning("Collection '%s' not found for deletion", name)


def list_session_collections() -> List[str]:
    """Return all collection names starting with 'session_'."""
    client = get_chroma_client()
    try:
        return [c.name for c in client.list_collections() if c.name.startswith("session_")]
    except Exception:
        logger.exception("Failed to list collections")
        return []


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
    # Use filename as the source tag -- clean and readable
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

    global _bm25_cache_key
    _bm25_cache_key = None
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
    use_hybrid: bool = True,
) -> List[Document]:
    """
    Embed the query and find the k most similar chunks across all documents.
    If filter_sources is provided, only return chunks from those source filenames.
    If use_hybrid is True, also run BM25 search and fuse results with RRF.
    """
    try:
        embeddings_model = get_embeddings()
        query_vector = embeddings_model.embed_query(query)

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

    vector_docs = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        vector_docs.append(Document(page_content=text, metadata=metadata))

    if not use_hybrid:
        return vector_docs

    bm25_docs = query_bm25(collection, query, k=k, filter_sources=filter_sources)
    if not bm25_docs:
        return vector_docs

    return _rrf_fuse(vector_docs, bm25_docs)
