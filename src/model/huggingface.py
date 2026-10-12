import math
from typing import List

import torch

# from accelerate import find_executable_batch_size
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM  # BitsAndBytesConfig,
from transformers import AutoTokenizer, pipeline

from src.utils.image_utils import pil_image_to_base64


class BaseLLM:
    """base class for any LLM models we want to use"""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        seed: int = 42,
        max_tokens=128,
        starting_batch_size=4,
    ) -> None:
        # credentials
        self.key = api_key
        self.model_name = model_name
        self.seed = seed

        # misc
        self.max_tokens = max_tokens
        self.queries = 0

        self.starting_batch_size = starting_batch_size

        # determine device for torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vision = False  # whether the model supports vision tasks, default false

    def query(self, task: str, temp: int):
        raise NotImplementedError("query method must be implemented in subclass")

    def batch_query(self, context_batch, prompt_batch, temp: int):
        """Batch query will return a list of responses (size len(context_batch) for a
        list of prompts"""
        raise NotImplementedError("batch_query method must be implemented in subclass")

    def batch_query_messages(self, all_messages: List):
        """Batch query will return a list of responses (size len(context_batch) for a
        list of prompts"""
        raise NotImplementedError("batch_query method must be implemented in subclass")

    def create_message_for_prompt(self, prompt: str, context: str = None, image=None):
        messages = []
        if context is not None:
            messages.append({"role": "system", "content": context})

        if image is None:
            messages.append({"role": "user", "content": prompt})
        else:
            base64_image = pil_image_to_base64(image)

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            )
        return messages


class HuggingFace(BaseLLM):
    """HuggingFace model base class for LLMs.

    Args:
        - model_name: Hugging face model name.
        - quantization_config: BnB quantization config
        - key: Hugging face token for restricted models.
    """

    # store loaded models in a dictionary, along with relevant quantization
    # ex. {model_name: (model, tokenizer, quantization)}
    loaded_models = {}

    def __init__(
        self,
        model_name: str,
        quantization_config: DictConfig = None,
        key: str = None,
        **kwargs,
    ):
        super().__init__(key, model_name, **kwargs)

        # Cache loaded (model, tokenizer, quantization) by model_name at the
        # class level so repeated instantiations (e.g. one HuggingFace wrapper
        # per worker/eval call) reuse the same in-memory weights instead of
        # reloading them from disk each time.
        if model_name in HuggingFace.loaded_models:
            self.model, self.tokenizer, self.bnb_config = HuggingFace.loaded_models[
                model_name
            ]
        else:
            # Quantization
            if quantization_config is not None:
                # self.bnb_config = BitsAndBytesConfig(**quantization_config)
                self.bnb_config = None
                extra_kwargs = {"quantization_config": self.bnb_config}
            else:
                self.bnb_config = None
                # This assumes all HF models are bfloat16, which might not be correct.
                # Setting it to "auto" might also work, but it is not tested.
                extra_kwargs = {"torch_dtype": torch.bfloat16}

            # create tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=key, padding_side="left"
            )
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto", token=key, **extra_kwargs
            )

            print(model.hf_device_map)
            print(model.get_memory_footprint())

            self.model = model
            self.tokenizer = tokenizer

            # store the model
            HuggingFace.loaded_models[model_name] = (model, tokenizer, self.bnb_config)

        self.generator = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def batch_query_messages(
        self,
        all_messages: List,
        temp: int = 0.4,
        top_p: float = 0.95,
        top_k: int = 50,
        use_chat_template=True,
    ):
        """Batch query a list of messages. Assumes messages are already in the format
        [{"context": str, "prompt": str}, ...]"""

        if use_chat_template:
            # chat template the messages
            templated_messages = []
            for m in all_messages:
                templated_messages.append(
                    self.tokenizer.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                )
            all_messages = templated_messages

        responses = []

        # feed forward parts of the batch according to dynamic batch size
        # @find_executable_batch_size(starting_batch_size=self.starting_batch_size)
        # def inner_training_loop(batch_size):
        responses.clear()

        batch_size = self.starting_batch_size

        for i in range(0, math.ceil(len(all_messages) / batch_size)):
            batch_messages = all_messages[i * batch_size : (i + 1) * batch_size]

            # tokenize the entire batch. batch encode plus also works, but
            # __call__ will check either way
            tokens = self.tokenizer(
                batch_messages, return_tensors="pt", padding=True
            ).to(self.device)
            # Padded input length -- used below to slice off the prompt tokens
            # from generate()'s output (which includes the input) so we only
            # decode the newly generated continuation.
            buffer = tokens.input_ids.shape[1]

            # Stop generation at either the tokenizer's own EOS or the
            # Llama-3-style "<|eot_id|>" turn-end token, since not all
            # tokenizers register the latter as eos_token_id.
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]

            outputs = self.model.generate(
                **tokens,
                max_new_tokens=self.max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
            )

            # decode outputs individually (batch decode also works)
            for i in range(len(outputs)):
                out = self.tokenizer.decode(
                    outputs[i][buffer:], skip_special_tokens=True
                )
                responses.append(out)

        # inner_training_loop()

        return responses

    def batch_query(
        self,
        context_batch,
        prompt_batch,
        temp: int = 0.4,
        top_p: float = 0.95,
        top_k: int = 50,
        chat=True,
    ):
        all_messages = []

        for context, prompt in zip(context_batch, prompt_batch):
            if chat:
                templated_prompt = self.create_message_for_prompt(context, prompt)
            else:
                templated_prompt = f"{context}\n{prompt}"

            all_messages.append(templated_prompt)

        return self.batch_query_messages(
            all_messages, temp=temp, top_p=top_p, top_k=top_k, use_chat_template=chat
        )
