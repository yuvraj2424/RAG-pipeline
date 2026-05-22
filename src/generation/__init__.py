from src.generation.schemas import RAGAnswer
from src.generation.chains import generate, check_faithfulness


class GenerationPipeline:
    """Per-query pipeline: Generate structured answer → Faithfulness check."""

    def generate(self, query: str, context: str, chat_history: list | None = None) -> RAGAnswer:
        return generate(query, context, chat_history)

    def check_faithfulness(self, answer: RAGAnswer, context: str) -> bool:
        return check_faithfulness(answer, context)
