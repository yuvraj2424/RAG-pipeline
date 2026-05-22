from langchain_core.documents import Document

from src.augmentation import AugmentationPipeline
from src.generation import GenerationPipeline
from src.generation.schemas import RAGAnswer
from src.graph.state import RAGState
from src.retrieval import RetrievalPipeline
from src.tools.web_search import search_to_documents


def make_retrieve_node(retrieval: RetrievalPipeline):
    def retrieve_node(state: RAGState) -> dict:
        vector_docs = retrieval.retrieve(state["query"])
        # Merge with any docs pre-seeded in state (e.g. from MCP fetch in the router)
        combined = list(state.get("documents") or []) + vector_docs
        print(f"[Retrieval] {len(state.get('documents') or [])} MCP + {len(vector_docs)} vector = {len(combined)} total docs")
        return {"documents": combined}
    return retrieve_node


def make_augment_node(augment: AugmentationPipeline):
    def augment_node(state: RAGState) -> dict:
        docs, ctx = augment.run(state["query"], state["documents"])
        return {"documents": docs, "context": ctx}
    return augment_node


def make_generate_node(generation: GenerationPipeline):
    def generate_node(state: RAGState) -> dict:
        if not state["context"]:
            fallback = RAGAnswer(
                answer="I don't have enough information to answer this confidently.",
                confidence=0.0,
                citations=[],
                caveats="No relevant documents were found in the knowledge base.",
            )
            return {"answer": fallback, "faithful": True}
        answer = generation.generate(
            state["query"], state["context"], state.get("chat_history")
        )
        return {"answer": answer, "faithful": False}  # reset flag before faithfulness check
    return generate_node


def make_faithfulness_node(generation: GenerationPipeline):
    def faithfulness_node(state: RAGState) -> dict:
        faithful = generation.check_faithfulness(state["answer"], state["context"])
        return {"faithful": faithful, "faith_retries": state["faith_retries"] + 1}
    return faithfulness_node


def web_search_node(state: RAGState) -> dict:
    """Fallback: live web search via Tavily (preferred) or DuckDuckGo."""
    print(f"[CRAG] No relevant docs — falling back to web search for: {state['query']}")
    docs = search_to_documents(state["query"])
    return {"documents": docs, "web_search_retries": state["web_search_retries"] + 1}
