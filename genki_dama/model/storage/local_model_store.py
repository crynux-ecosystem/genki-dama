import abc
from typing import Dict
from genki_dama.model.creative_model import CreativeModel
from constants import CompetitionParameters


class LocalModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model locally."""

    @abc.abstractmethod
    def store_model(self, hotkey: str, model: CreativeModel) -> CreativeModel:
        """Stores a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    def get_path(self, hotkey: str) -> str:
        """Returns the path to the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    def retrieve_model(self, hotkey: str, model_id: CreativeModel, parameters: CompetitionParameters) -> CreativeModel:
        """Retrieves a trained model from the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    def delete_unreferenced_models(self, valid_models_by_hotkey: Dict[str, CreativeModel], grace_period_seconds: int):
        """Check across all of local storage and delete unreferenced models out of grace period."""
        pass
