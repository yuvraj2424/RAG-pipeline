from langchain_core.documents import Document


def position_chunks(docs: list[Document]) -> list[Document]:
    """
    Lost-in-the-middle fix: place best chunks first and last,
    weakest in the middle (LLMs attend better to context boundaries).
    """
    if len(docs) <= 2:
        return docs
    mid = docs[1:-1]
    return [docs[0]] + mid[::-1] + [docs[-1]]


def build_context(docs: list[Document]) -> str:
    """Format chunks into a numbered citation-ready context block."""
    parts = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i + 1}] ({src}):\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
