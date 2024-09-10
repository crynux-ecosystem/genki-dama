
from abc import ABC, abstractmethod

class GPTAPI(ABC):
    @abstractmethod
    def get_response(self, text: str) -> str:
        pass
