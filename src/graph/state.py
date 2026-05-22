from typing import Optional, TypedDict

from langchain_core.documents import Document

from src.generation.schemas import RAGAnswer


class RAGState(TypedDict):
    query:        str
    chat_history: list          # [{"role": "user"/"assistant", "content": "..."}]
    documents:    list[Document]
    context:      str
    answer:       Optional[RAGAnswer]
    faithful:     bool
    faith_retries: int          # budget for faithfulness re-generation
