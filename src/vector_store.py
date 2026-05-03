import os
from typing import List
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
import chromadb

CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")


def get_chroma_client():
    """Get a persistent ChromaDB client that saves to disk."""
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def build_vector_store(chunks: List[Document]) -> chromadb.Collection:
    """
    Embed chunks and store in ChromaDB.
    Deletes any existing collection first so re-uploads are clean.
    """
    client = get_chroma_client()

    # Delete old collection if it exists (fresh start on new upload)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    embeddings_model = get_embeddings()

    # Embed all chunks in one API call
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    vectors = embeddings_model.embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB")
    return collection


def query_vector_store(
    collection: chromadb.Collection,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Embed the query and find the k most similar chunks.
    Returns LangChain Document objects so the rest of the app stays consistent.
    """
    embeddings_model = get_embeddings()
    query_vector = embeddings_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
    )

    # Repack into LangChain Document objects
    docs = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(Document(page_content=text, metadata=metadata))

    return docs