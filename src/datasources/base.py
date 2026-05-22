from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseDataSource(ABC):
    """
    Abstract base for all data source connectors.
    Each subclass loads raw data and returns LangChain Documents
    ready for the IndexingPipeline.
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """Load documents from the source."""
        ...

    @property
    def source_name(self) -> str:
        return self.__class__.__name__
