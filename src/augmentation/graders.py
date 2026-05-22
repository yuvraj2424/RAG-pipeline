from langchain_core.documents import Document

from src.llm import llm_with_retry as llm


def grade_documents(query: str, docs: list[Document]) -> list[Document]:
    """
    CRAG: LLM grades each chunk as relevant or irrelevant.
    Drops irrelevant chunks before building context.
    """
    relevant = []
    for doc in docs:
        grade_prompt = f"""Question: {query}

Document excerpt:
{doc.page_content[:400]}

Is this document relevant to answer the question?
Reply with ONLY: "relevant" or "irrelevant" """
        grade = llm.invoke(grade_prompt).content.strip().lower()
        if "relevant" in grade and "irrelevant" not in grade:
            relevant.append(doc)

    print(f"[Augmentation] Graded: {len(relevant)}/{len(docs)} chunks relevant")
    return relevant
