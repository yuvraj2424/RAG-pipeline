from pathlib import Path

from langchain_core.documents import Document

from src.datasources.base import BaseDataSource
from src.indexing.loaders import load_from_directory, load_documents


class FileSystemDataSource(BaseDataSource):
    """
    Load documents from a directory (PDF, DOCX, XLSX) or an explicit
    list of source dicts. Backed by src/indexing/loaders.py.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        sources: list[dict] | None = None,
    ):
        if not directory and not sources:
            raise ValueError("Provide either 'directory' or 'sources'.")
        self.directory = Path(directory) if directory else None
        self.sources = sources

    def load(self) -> list[Document]:
        if self.directory:
            return load_from_directory(self.directory)
        return load_documents(self.sources)
