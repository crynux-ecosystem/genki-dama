import abc
from genki_dama.model.creative_model import CreativeModel
from constants import CompetitionParameters


class RemoteModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model."""

    @abc.abstractmethod
    async def upload_model(self, model: CreativeModel, parameters: CompetitionParameters) -> CreativeModel:
        """Uploads a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    async def download_model(self, model_id: CreativeModel, local_path: str, parameters: CompetitionParameters) -> CreativeModel:
        """Retrieves a trained model from the appropriate location and stores at the given path."""
        pass
