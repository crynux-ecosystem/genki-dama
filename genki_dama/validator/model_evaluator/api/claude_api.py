
from genki_dama.validator.model_evaluator.api.api import GPTAPI


class ClaudeAPI(GPTAPI):
    def get_response(self, text: str) -> str:
        return ""

