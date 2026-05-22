from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

from src.config import RERANKER_TOP_N, FINAL_TOP_K


def build_reranker(base_retriever) -> ContextualCompressionRetriever:
    """Wrap a retriever with FlashrankRerank (local, free)."""
    return ContextualCompressionRetriever(
        base_compressor=FlashrankRerank(top_n=RERANKER_TOP_N),
        base_retriever=base_retriever,
    )


def deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks by content hash, keep top FINAL_TOP_K."""
    seen = set()
    unique = []
    for d in docs:
        key = hash(d.page_content[:100])
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique[:FINAL_TOP_K]
