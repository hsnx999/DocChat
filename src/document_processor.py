from pathlib import Path
from typing import List
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.settings import CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF from disk.
    Return a list of Document objects — one per page.
    Each Document has two things:
      - page_content: the raw text of that page
      - metadata: dict with info like {"page": 3, "source": "paper.pdf"}
    """
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return pages
    except Exception as e:
        msg = str(e).lower()
        if "encrypted" in msg:
            logger.warning("Encrypted PDF attempted: %s", file_path)
            raise ValueError("The PDF is encrypted and cannot be read.")
        if "cannot read" in msg or "no such file" in msg:
            logger.warning("Unreadable file: %s", file_path)
            raise ValueError(f"Cannot read the file: {file_path}")
        logger.exception("Failed to load PDF: %s", file_path)
        raise ValueError(f"Failed to load PDF: {e}")


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on:
      1. Paragraphs (\n\n)  ← preferred, keeps ideas together
      2. Lines (\n)
      3. Sentences (". ")
      4. Words (" ")
      5. Characters ("")    ← last resort
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def process_uploaded_file(uploaded_file, save_dir: str = "data") -> List[Document]:
    """
    Accept a Streamlit UploadedFile, save it to disk, then load and chunk it.
    """
    if uploaded_file.size > MAX_FILE_SIZE:
        raise ValueError(
            f"File size ({uploaded_file.size / 1024 / 1024:.1f} MB) exceeds "
            f"the maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
        )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / uploaded_file.name
    raw_bytes = uploaded_file.read()

    if raw_bytes[:4] != b"%PDF":
        raise ValueError("The uploaded file is not a valid PDF (missing %PDF header).")

    file_path.write_bytes(raw_bytes)

    try:
        pages = load_pdf(str(file_path))
        chunks = chunk_documents(pages)
    except Exception:
        logger.exception("Failed to process uploaded file, cleaning up")
        file_path.unlink()
        raise

    return chunks
