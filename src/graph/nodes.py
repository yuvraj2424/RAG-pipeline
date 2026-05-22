from src.augmentation import AugmentationPipeline
from src.generation import GenerationPipeline
from src.generation.schemas import RAGAnswer
from src.graph.state import RAGState
from src.retrieval import RetrievalPipeline


def make_retrieve_node(retrieval: RetrievalPipeline):
    def retrieve_node(state: RAGState) -> dict:
        vector_docs = retrieval.retrieve(state["query"])
        # Merge with any docs pre-seeded in state
        combined = list(state.get("documents") or []) + vector_docs
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
        return {"answer": answer, "faithful": False}
    return generate_node


def make_faithfulness_node(generation: GenerationPipeline):
    def faithfulness_node(state: RAGState) -> dict:
        faithful = generation.check_faithfulness(state["answer"], state["context"])
        return {"faithful": faithful, "faith_retries": state["faith_retries"] + 1}
    return faithfulness_node
