from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, SEMANTIC_BREAKPOINT_PERCENTILE
from src.llm import embeddings


def clean_documents(docs: list[Document]) -> list[Document]:
    """Strip noise: headers, footers, extra whitespace. Drop near-empty pages."""
    cleaned = []
    for d in docs:
        text = d.page_content
        text = "\n".join(
            line for line in text.splitlines()
            if not any(noise in line.lower() for noise in
                       ["page ", "confidential", "all rights reserved"])
        )
        text = " ".join(text.split())
        if len(text) > 50:
            d.page_content = text
            cleaned.append(d)

    print(f"[Indexing] Cleaned → {len(cleaned)} docs remaining")
    return cleaned


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Semantic chunking (meaning-aware splits).
    Falls back to RecursiveCharacterTextSplitter for short docs.
    """
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_PERCENTILE,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for doc in docs:
        try:
            splits = semantic_splitter.split_documents([doc])
        except Exception:
            splits = fallback_splitter.split_documents([doc])

        for i, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(splits)
        chunks.extend(splits)

    print(f"[Indexing] Chunked → {len(chunks)} chunks")
    return chunks
