from unittest.mock import MagicMock

from src.vector_store import add_documents, remove_document, get_indexed_files


def test_get_indexed_files_empty():
    collection = MagicMock()
    collection.count.return_value = 0
    assert get_indexed_files(collection) == []


def test_get_indexed_files_returns_sorted_names():
    collection = MagicMock()
    collection.count.return_value = 3
    collection.get.return_value = {
        "metadatas": [
            {"source": "c.pdf"},
            {"source": "a.pdf"},
            {"source": "b.pdf"},
        ]
    }
    assert get_indexed_files(collection) == ["a.pdf", "b.pdf", "c.pdf"]


def test_remove_document_calls_delete():
    collection = MagicMock()
    remove_document(collection, "test.pdf")
    collection.delete.assert_called_once_with(where={"source": "test.pdf"})


def test_add_documents_returns_chunk_count(mocker, sample_chunks):
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 384 for _ in sample_chunks]
    mocker.patch("src.vector_store.get_embeddings", return_value=mock_embeddings)
    collection = MagicMock()
    result = add_documents(collection, sample_chunks, "test.pdf")
    assert result == len(sample_chunks)
