from langchain_core.documents import Document

from src.augmentation.graders import grade_documents
from src.augmentation.context import position_chunks, build_context


class AugmentationPipeline:
    """Per-query pipeline: Grade docs → Position chunks → Build context."""

    def run(self, query: str, docs: list[Document]) -> tuple[list[Document], str]:
        relevant = grade_documents(query, docs)

        if not relevant:
            print("[Augmentation] No relevant docs — will trigger fallback")
            return [], ""

        positioned = position_chunks(relevant)
        context = build_context(positioned)
        return positioned, context
