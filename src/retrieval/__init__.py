from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.retrieval.retrievers import build_hybrid_retriever, transform_query
from src.retrieval.rerankers import build_reranker, deduplicate


class RetrievalPipeline:
    """Per-query pipeline: Multi-query expansion → Hybrid search → Rerank → Deduplicate."""

    def __init__(self, vectorstore: Chroma, all_docs: list[Document]):
        hybrid = build_hybrid_retriever(vectorstore, all_docs)
        self.retriever = build_reranker(hybrid)

    def retrieve(self, query: str) -> list[Document]:
        queries = transform_query(query)
        all_docs: list[Document] = []
        seen: set[int] = set()

        for q in queries:
            for d in self.retriever.invoke(q):
                key = hash(d.page_content[:100])
                if key not in seen:
                    seen.add(key)
                    all_docs.append(d)

        result = deduplicate(all_docs)
        print(f"[Retrieval] {len(queries)} queries → {len(result)} unique chunks")
        return result
