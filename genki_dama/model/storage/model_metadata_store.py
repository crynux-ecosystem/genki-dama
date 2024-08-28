import abc
from typing import Optional
from genki_dama.model.creative_model import CreativeModel


class ModelMetadataStore(abc.ABC):
    """An abstract base class for storing and retrieving model metadata."""

    @abc.abstractmethod
    async def store_model_metadata(self, hotkey: str, creative_model: CreativeModel):
        """Stores model metadata on this subnet for a specific miner."""
        pass

    @abc.abstractmethod
    async def retrieve_model_metadata(self, hotkey: str) -> Optional[CreativeModel]:
        """Retrieves model metadata + block information on this subnet for specific miner, if present"""
        pass
