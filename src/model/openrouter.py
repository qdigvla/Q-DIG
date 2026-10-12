from .openai import OpenAIClient

from openai import OpenAI

class OpenRouterClient(OpenAIClient):
    def __init__(self, key, model_name):
        self.key = key
        self.model_name = model_name
        
        self.client = OpenAI(
            api_key=key,
            base_url = "https://openrouter.ai/api/v1/"
        )
