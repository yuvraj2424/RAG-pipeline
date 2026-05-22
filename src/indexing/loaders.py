from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}


def load_from_directory(directory: str | Path) -> list[Document]:
    """
    Recursively scan a directory and load all supported files
    (PDF, DOCX, XLSX). Metadata is derived from the file path.
    """
    directory = Path(directory)
    docs = []

    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # Derive category from the immediate parent folder name
        category = file_path.parent.name if file_path.parent != directory else "general"

        source = {
            "type": file_path.suffix.lower().lstrip("."),
            "name": file_path.stem,
            "path": str(file_path),
            "category": category,
        }
        loaded = _load_file(source)
        docs.extend(loaded)

    print(f"[Indexing] Loaded {len(docs)} documents from {directory}")
    return docs


def load_documents(sources: list[dict]) -> list[Document]:
    """Load from a list of source dicts (pdf/docx/xlsx/web/text) with metadata tagging."""
    docs = []
    for src in sources:
        loaded = _load_file(src)
        docs.extend(loaded)

    print(f"[Indexing] Loaded {len(docs)} documents")
    return docs


def _load_file(src: dict) -> list[Document]:
    """Dispatch to the correct loader and attach source metadata."""
    file_type = src["type"].lower()

    if file_type == "pdf":
        loaded = PyPDFLoader(src["path"]).load()
    elif file_type == "docx":
        loaded = Docx2txtLoader(src["path"]).load()
    elif file_type == "xlsx":
        loaded = UnstructuredExcelLoader(src["path"], mode="elements").load()
    elif file_type == "web":
        loaded = WebBaseLoader(src["url"]).load()
    elif file_type == "text":
        loaded = [Document(page_content=src["content"])]
    else:
        return []

    for d in loaded:
        d.metadata.update({
            "source":   src.get("name", "unknown"),
            "category": src.get("category", "general"),
            "date":     src.get("date", ""),
        })
    return loaded
