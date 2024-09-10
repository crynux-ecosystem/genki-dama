
import json
import os
import requests
from genki.validator.model_evaluator.api.api import GPTAPI

ENDPOINT = 'https://api.anthropic.com/v1/messages'


class ClaudeAPI(GPTAPI):

    def __init__(self) -> None:
        super().__init__()
        if not os.getenv("CLAUDE_API_ACCESS_TOKEN"):
            raise ValueError("No Claude API access token found.")
        self.api_key = os.getenv("CLAUDE_API_ACCESS_TOKEN")

        if not os.getenv("CLAUDE_ENDPOINT"):
            self.api_endpoint=ENDPOINT
        else:
            self.api_endpoint=os.getenv("CLAUDE_ENDPOINT")


    def get_response(self, text: str) -> str:

        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json',
        }

        data = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": text}
            ]
        }

        response = requests.post(self.api_endpoint, headers=headers, data=json.dumps(data))
        response_json = response.json()

        response_text = ""

        for item in response_json["content"]:
            if item["type"] == "text":
                if response_text != "":
                    response_text += "\n"
                response_text += item["text"]

        return response_text
