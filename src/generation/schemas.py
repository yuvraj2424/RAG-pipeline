from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class RAGAnswer(BaseModel):
    answer:     str        = Field(description="Direct answer to the question, 2-4 sentences")
    confidence: float      = Field(description="Confidence 0.0–1.0 based on context quality")
    citations:  list[str]  = Field(description="Source references used, e.g. ['Source 1', 'Source 2']")
    caveats:    str | None = Field(description="Limitations or missing info, if any")


RAG_SYSTEM = """You are a precise assistant. Answer ONLY using the provided context.

RULES:
1. If context doesn't contain enough info, say exactly:
   "I don't have enough information to answer this confidently."
2. Never invent facts not in the context.
3. Always cite sources as [Source N].
4. Be concise — 2-4 sentences unless detail is requested.
5. Set confidence based on how well the context covers the question.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
