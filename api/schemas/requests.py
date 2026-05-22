from typing import Literal
from pydantic import BaseModel, model_validator


class QueryRequest(BaseModel):
    query: str
    chat_history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]


class DocumentSource(BaseModel):
    """File-based source (pdf / docx / xlsx / web / text)."""
    type: Literal["text", "pdf", "docx", "xlsx", "web"]
    name: str
    category: str = "general"
    date: str = ""
    content: str | None = None  # required for type="text"
    path: str | None = None     # required for type="pdf"/"docx"/"xlsx"
    url: str | None = None      # required for type="web"

    @model_validator(mode="after")
    def check_required_fields(self):
        if self.type == "text" and not self.content:
            raise ValueError("'content' is required for type='text'")
        if self.type in ("pdf", "docx", "xlsx") and not self.path:
            raise ValueError(f"'path' is required for type='{self.type}'")
        if self.type == "web" and not self.url:
            raise ValueError("'url' is required for type='web'")
        return self


class SQLSource(BaseModel):
    """SQL Server (or any SQLAlchemy-compatible DB) source."""
    connection_string: str
    query: str | None = None        # Custom SELECT — takes priority over table
    table: str | None = None        # Load all rows from this table
    content_columns: list[str] | None = None
    metadata_columns: list[str] | None = None
    source_name: str = "sql_server"
    category: str = "database"

    @model_validator(mode="after")
    def check_query_or_table(self):
        if not self.query and not self.table:
            raise ValueError("Provide either 'query' or 'table'.")
        return self


class WebSource(BaseModel):
    """Crawl one or more web URLs."""
    urls: list[str]
    category: str = "web"
    source_name: str = "web"


class IndexRequest(BaseModel):
    # All fields optional — omitting all triggers source/ directory scan
    sources: list[DocumentSource] | None = None
    sql: SQLSource | None = None
    web: WebSource | None = None
