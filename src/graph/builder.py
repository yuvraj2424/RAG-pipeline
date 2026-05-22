from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.augmentation import AugmentationPipeline
from src.generation import GenerationPipeline
from src.graph.edges import route_after_augment, route_after_faithfulness
from src.graph.nodes import (
    make_augment_node,
    make_faithfulness_node,
    make_generate_node,
    make_retrieve_node,
    web_search_node,
)
from src.graph.state import RAGState
from src.retrieval import RetrievalPipeline


def build_rag_graph(vectorstore: Chroma, all_docs: list[Document]):
    """Assemble and compile the CRAG self-correcting LangGraph."""
    retrieval  = RetrievalPipeline(vectorstore, all_docs)
    augment    = AugmentationPipeline()
    generation = GenerationPipeline()

    graph = StateGraph(RAGState)

    graph.add_node("retrieve",    make_retrieve_node(retrieval))
    graph.add_node("augment",     make_augment_node(augment))
    graph.add_node("generate",    make_generate_node(generation))
    graph.add_node("faithfulness", make_faithfulness_node(generation))
    graph.add_node("web_search",  web_search_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "augment")
    graph.add_conditional_edges(
        "augment", route_after_augment,
        {"generate": "generate", "web_search": "web_search"},
    )
    graph.add_edge("web_search", "augment")
    graph.add_edge("generate", "faithfulness")
    graph.add_conditional_edges(
        "faithfulness", route_after_faithfulness,
        {END: END, "generate": "generate"},
    )

    return graph.compile()
