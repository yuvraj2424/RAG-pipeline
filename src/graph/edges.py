from langgraph.graph import END

from src.config import MAX_RETRIES
from src.graph.state import RAGState


def route_after_augment(state: RAGState) -> str:
    """No context + web-search budget remaining → web_search, otherwise → generate."""
    if not state["context"] and state["web_search_retries"] < MAX_RETRIES:
        return "web_search"
    return "generate"


def route_after_faithfulness(state: RAGState) -> str:
    """Faithful or faithfulness budget exhausted → END, otherwise → regenerate."""
    if state["faithful"] or state["faith_retries"] >= MAX_RETRIES:
        return END
    return "generate"
