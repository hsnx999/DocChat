import pytest
from typing import List
from langchain.schema import Document


@pytest.fixture
def sample_doc() -> Document:
    return Document(
        page_content="Python is a programming language. It is widely used for AI.",
        metadata={"source": "test.pdf", "page": 1},
    )


@pytest.fixture
def sample_chunks() -> List[Document]:
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test.pdf", "page": 1},
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"source": "test.pdf", "page": 2},
        ),
        Document(
            page_content="Natural language processing enables computers to understand text.",
            metadata={"source": "test.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def chat_history() -> list[dict]:
    return [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What is it used for?"},
    ]
