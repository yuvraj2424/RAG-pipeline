from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import VECTOR_STORE_DIR
from src.llm import embeddings


def embed_and_store(chunks: list[Document], persist_dir: str | None = None) -> tuple:
    """
    Build dense (Chroma) and sparse (BM25) indices.
    Returns (vectorstore, chunks) for the retrieval stage.
    """
    store_path = persist_dir or str(VECTOR_STORE_DIR)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=store_path,
    )
    print(f"[Indexing] Stored {len(chunks)} chunks in vector DB + BM25 ready")
    return vectorstore, chunks


def load_existing_store(persist_dir: str | None = None) -> tuple[Chroma, list[Document]] | None:
    """
    Load a persisted Chroma store and reconstruct Document objects for BM25.
    Returns (vectorstore, all_docs) or None if no store exists.
    """
    store_path = persist_dir or str(VECTOR_STORE_DIR)
    store_dir = VECTOR_STORE_DIR if persist_dir is None else Path(persist_dir)

    if not store_dir.exists() or not any(store_dir.iterdir()):
        return None

    vectorstore = Chroma(
        persist_directory=store_path,
        embedding_function=embeddings,
    )
    raw = vectorstore.get()
    all_docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return vectorstore, all_docs
