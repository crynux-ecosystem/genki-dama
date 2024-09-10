
from genki.validator.model_evaluator.api.api import GPTAPI


class ChatGPTAPI(GPTAPI):
    def get_response(self, text: str) -> str:
        return ""

