from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import DENSE_K, DENSE_FETCH_K, SPARSE_K, HYBRID_WEIGHTS, MULTI_QUERY_COUNT
from src.llm import llm_with_retry as llm


def build_hybrid_retriever(vectorstore: Chroma, all_docs: list[Document]) -> EnsembleRetriever:
    """Dense MMR + sparse BM25 combined with configured weights."""
    dense = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": DENSE_K, "fetch_k": DENSE_FETCH_K},
    )
    sparse = BM25Retriever.from_documents(all_docs, k=SPARSE_K)

    return EnsembleRetriever(
        retrievers=[sparse, dense],
        weights=HYBRID_WEIGHTS,
    )


def transform_query(query: str) -> list[str]:
    """
    Multi-query expansion: LLM generates MULTI_QUERY_COUNT variants.
    Broadens recall without HyDE's hallucination risk.
    """
    prompt = f"""Generate {MULTI_QUERY_COUNT} different search queries to find information about:
"{query}"

Return ONLY the queries, one per line, no numbering."""
    resp = llm.invoke(prompt).content.strip()
    variants = [q.strip() for q in resp.splitlines() if q.strip()]
    return [query] + variants[:MULTI_QUERY_COUNT]
