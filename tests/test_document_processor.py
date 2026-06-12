import pytest
from langchain_core.documents import Document
from src.document_processor import chunk_documents

class TestChunkDocuments:

    def test_chunk_documents_single_small_doc(self):
        doc = Document(page_content="Hello world. This is a test.")
        result = chunk_documents([doc], chunk_size=1000, chunk_overlap=0)
        assert len(result) == 1
        assert result[0].page_content == doc.page_content

    def test_chunk_documents_splits_large_doc(self):
        text = "word " * 500
        doc = Document(page_content=text)
        result = chunk_documents([doc], chunk_size=200, chunk_overlap=0)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk.page_content) <= 250

    def test_chunk_overlap_produces_shared_text(self):
        text = "one two three four five six seven eight nine ten " * 20
        doc = Document(page_content=text)
        result = chunk_documents([doc], chunk_size=50, chunk_overlap=10)
        assert len(result) >= 2
        for i in range(len(result) - 1):
            c1 = result[i].page_content
            c2 = result[i + 1].page_content
            assert len(set(c1.split()) & set(c2.split())) > 0

    def test_chunk_documents_empty_list(self):
        result = chunk_documents([], chunk_size=1000, chunk_overlap=200)
        assert result == []

    def test_chunk_documents_preserves_metadata(self, sample_chunks):
        result = chunk_documents(sample_chunks, chunk_size=1000, chunk_overlap=0)
        assert len(result) == len(sample_chunks)
        for original, chunk in zip(sample_chunks, result):
            assert chunk.metadata == original.metadata
