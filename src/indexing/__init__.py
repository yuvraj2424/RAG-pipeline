from langchain_core.documents import Document

from src.config import VECTOR_STORE_DIR
from src.indexing.loaders import load_documents, load_from_directory
from src.indexing.chunkers import clean_documents, chunk_documents
from src.indexing.embeddings import embed_and_store


class IndexingPipeline:
    """Offline pipeline: Load → Clean → Chunk → Embed → Store."""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or str(VECTOR_STORE_DIR)

    def run_from_directory(self, directory) -> tuple:
        """Load all supported files from a directory and index them."""
        docs = load_from_directory(directory)
        return self._process(docs)

    def run_from_docs(self, docs: list[Document]) -> tuple:
        """Index pre-loaded Document objects (from any datasource)."""
        return self._process(docs)

    def run(self, sources: list[dict]) -> tuple:
        """Load from an explicit list of source dicts and index them."""
        docs = load_documents(sources)
        return self._process(docs)

    def _process(self, docs: list[Document]) -> tuple:
        docs = clean_documents(docs)
        chunks = chunk_documents(docs)
        return embed_and_store(chunks, self.persist_dir)
