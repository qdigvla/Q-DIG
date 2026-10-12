from .huggingface import BaseLLM

from logging import warning

from typing import List

class OpenVLA(BaseLLM):
    def __init__(
        self,
        model_name
    ):
        # create model using gpt-4
        super().__init__(model_name)
        
        # TODO: load OpenVLA through here