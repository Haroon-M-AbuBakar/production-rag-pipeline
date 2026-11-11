from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_pdf(filepath) -> list[Document]:
    """Read a PDF and return one Document per non-empty page."""
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc_id = file_path.stem.lower().replace(" ", "_")
    documents = []

    with pymupdf.open(file_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if not text.strip():
                continue  # skip blank / image-only pages
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": file_path.name,
                        "page": page_num,
                        "doc_id": doc_id,
                    },
                )
            )
    return documents


def chunker(
    documents: list[Document],
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Document]:
    """Split page Documents into smaller chunks. Metadata is copied to each chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)