from sqlalchemy import create_engine, text

from langchain_core.documents import Document

from src.datasources.base import BaseDataSource


class SQLServerDataSource(BaseDataSource):
    """
    Load rows from a SQL Server (or any SQLAlchemy-compatible DB) as Documents.

    Connection string examples:
      SQL Server : mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server
      PostgreSQL : postgresql+psycopg2://user:pass@host/db
      SQLite     : sqlite:///path/to/file.db

    Usage:
        src = SQLServerDataSource(
            connection_string=os.getenv("SQL_CONNECTION_STRING"),
            query="SELECT title, body, category FROM articles",
            content_columns=["title", "body"],
            metadata_columns=["category"],
            source_name="articles_db",
        )
        docs = src.load()
    """

    def __init__(
        self,
        connection_string: str,
        query: str | None = None,
        table: str | None = None,
        content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
        source_name: str = "sql_server",
        category: str = "database",
    ):
        if not query and not table:
            raise ValueError("Provide either 'query' or 'table'.")
        self.connection_string = connection_string
        self.query = query or f"SELECT * FROM {table}"  # noqa: S608
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self._source_name = source_name
        self.category = category

    @property
    def source_name(self) -> str:
        return self._source_name

    def load(self) -> list[Document]:
        engine = create_engine(self.connection_string)
        docs = []

        with engine.connect() as conn:
            rows = conn.execute(text(self.query)).mappings().all()

        for row in rows:
            row_dict = dict(row)

            if self.content_columns:
                content = "\n".join(
                    f"{k}: {v}" for k, v in row_dict.items()
                    if k in self.content_columns
                )
            else:
                content = "\n".join(f"{k}: {v}" for k, v in row_dict.items())

            metadata = {
                k: str(v) for k, v in row_dict.items()
                if k in self.metadata_columns
            }
            metadata["source"] = self._source_name
            metadata["category"] = self.category

            docs.append(Document(page_content=content, metadata=metadata))

        print(f"[SQLServerDataSource] Loaded {len(docs)} rows from '{self._source_name}'")
        return docs
