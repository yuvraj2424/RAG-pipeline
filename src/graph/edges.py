from langgraph.graph import END

from src.config import MAX_RETRIES
from src.graph.state import RAGState


def route_after_faithfulness(state: RAGState) -> str:
    """Faithful or faithfulness budget exhausted → END, otherwise → regenerate."""
    if state["faithful"] or state["faith_retries"] >= MAX_RETRIES:
        return END
    return "generate"
