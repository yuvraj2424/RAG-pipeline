from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    chat_history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]
