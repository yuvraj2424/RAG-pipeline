from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from src.datasources.base import BaseDataSource


class WebCrawlerDataSource(BaseDataSource):
    """
    Load one or more web pages as Documents.

    Usage:
        src = WebCrawlerDataSource(
            urls=["https://example.com/faq", "https://example.com/pricing"],
            category="website",
        )
        docs = src.load()
    """

    def __init__(
        self,
        urls: list[str],
        category: str = "web",
        source_name: str = "web",
    ):
        self.urls = urls
        self.category = category
        self._source_name = source_name

    @property
    def source_name(self) -> str:
        return self._source_name

    def load(self) -> list[Document]:
        docs = []
        for url in self.urls:
            loaded = WebBaseLoader(url).load()
            for d in loaded:
                d.metadata.update({
                    "source":   url,
                    "category": self.category,
                })
            docs.extend(loaded)

        print(f"[WebCrawlerDataSource] Loaded {len(docs)} pages from {len(self.urls)} URL(s)")
        return docs
