from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.generation.schemas import RAGAnswer, RAG_SYSTEM
from src.llm import llm, llm_with_retry, _RETRY_EXCEPTIONS

# Structured output with retry — chain: structured_output → retry wrapper
_structured_llm = llm.with_structured_output(RAGAnswer).with_retry(
    retry_if_exception_type=_RETRY_EXCEPTIONS,
    wait_exponential_jitter=True,
    stop_after_attempt=3,
)


def generate(query: str, context: str, chat_history: list | None = None) -> RAGAnswer:
    """
    Generate a structured, citation-grounded answer.
    Prepends chat_history so the LLM has multi-turn context.
    """
    messages = [SystemMessage(content=RAG_SYSTEM)]
    for msg in (chat_history or []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))

    answer: RAGAnswer = _structured_llm.invoke(messages)
    print(f"[Generation] Answer generated (confidence={answer.confidence:.2f})")
    return answer


def check_faithfulness(answer: RAGAnswer, context: str) -> bool:
    """
    Post-generation hallucination guard.
    Verifies every claim is grounded in the retrieved context.
    """
    faith_prompt = f"""Context provided to the model:
{context[:1500]}

Answer generated:
{answer.answer}

Is every factual claim in the answer directly supported by the context above?
Reply ONLY: "faithful" or "hallucinated" """
    result = llm_with_retry.invoke(faith_prompt).content.strip().lower()
    faithful = "faithful" in result and "hallucinated" not in result
    print(f"[Generation] Faithfulness check: {'PASS' if faithful else 'FAIL'}")
    return faithful
