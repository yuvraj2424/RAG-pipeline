from langchain_core.documents import Document

from src.datasources.base import BaseDataSource
from src.datasources.filesystem import FileSystemDataSource
from src.datasources.sql_server import SQLServerDataSource
from src.datasources.web_crawler import WebCrawlerDataSource

# Registry maps source_type string → class (extend here to add new sources)
REGISTRY: dict[str, type[BaseDataSource]] = {
    "filesystem": FileSystemDataSource,
    "sql_server":  SQLServerDataSource,
    "web":         WebCrawlerDataSource,
}


def load_from_source(source_type: str, **kwargs) -> list[Document]:
    """
    Convenience factory — load documents from a named source type.

    Examples:
        load_from_source("filesystem", directory="./source")
        load_from_source("sql_server", connection_string="...", table="products")
        load_from_source("web", urls=["https://..."])
    """
    cls = REGISTRY.get(source_type)
    if cls is None:
        raise ValueError(
            f"Unknown source type '{source_type}'. "
            f"Available: {list(REGISTRY)}"
        )
    return cls(**kwargs).load()


__all__ = [
    "BaseDataSource",
    "FileSystemDataSource",
    "SQLServerDataSource",
    "WebCrawlerDataSource",
    "REGISTRY",
    "load_from_source",
]
