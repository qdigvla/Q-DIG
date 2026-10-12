import functools
import json
import multiprocessing
import os
import random
from typing import Callable, Dict, List

import backoff
import numpy as np
import openai
import torch
import yaml
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from ...utils.image_utils import pil_image_to_base64


def backoff_hdlr(details):
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with args {args} and kwargs "
        "{kwargs}".format(**details)
    )


def pairwise_cosine_similarity(x1, x2):
    """
    Computes pairwise cosine similarity between two tensors.
    Args:
        x1 (torch.Tensor): First tensor of shape (N, D)
        x2 (torch.Tensor): Second tensor of shape (M, D)
    Returns:
        torch.Tensor: Pairwise cosine similarity matrix of shape (N, M)
    """
    x1 = x1 / x1.norm(dim=1, keepdim=True)
    x2 = x2 / x2.norm(dim=1, keepdim=True)
    return torch.matmul(x1, x2.t())


class CLIPEmbeddingModel:
    """Wraps a CLIP text encoder to embed candidate instructions, used by
    EmbodiedRedTeamModel's rejection sampling to pick the most diverse
    (lowest mean pairwise similarity) set of generated instructions."""

    def __init__(self, device="cuda"):
        self.device = device
        # Load by HF hub id (not a local cache path) so this works on any
        # machine; `from_pretrained` will use/populate the local HF cache
        # automatically.
        self.model = CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32", force_download=True, use_safetensors=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True
        )

    def __call__(self, text: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(
                self.device
            )
            outputs = self.model(**inputs)
            text_embeds = outputs.text_embeds
        return text_embeds


class InstructionSet(BaseModel):
    instructions: List[str]


class EmbodiedRedTeamModel:
    """
    The model takes the task description and the initial image as the feasible set as inputs and generates a set of instructions eliciting failure on the robot.
    """

    def __init__(
        self,
        embedding_model: Callable,
        model: str = "gpt-4o-2024-08-06",
        num_rejection_samples: int = 5,
    ):
        """
        num_rejection_samples: number of samples for diversity rejection sampling
        """
        self.client = OpenAI()
        self.model = model
        self.num_rejection_samples = num_rejection_samples
        self.embedding_model = embedding_model

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.BadRequestError),
        on_backoff=backoff_hdlr,
    )
    def __call__(
        self,
        task: str,
        image_url: str = None,
        examples: List[str] = [],
        num_instructions: int = 10,
        return_all_annotations=False,
    ):
        # Compose example sets
        if len(examples) > 0:
            examples_message = (
                " Here are the example instructions that met your goal:\n"
            )
            for i, example in enumerate(examples):
                examples_message += f"- {example}\n"
        else:
            examples_message = " "

        # open the image
        # from PIL import Image

        image = image_url  # Image.open(image_url)

        # Compose the prompt depending on providing image or not
        if image_url:
            image.load()
            
            base64_image = pil_image_to_base64(image)

            content = [
                {
                    "type": "text",
                    "text": f"The attached image is an example image of the initial state of a robot that will perform the task: {task}.{examples_message} Generate a diverse set of exactly {num_instructions} instructions.",
                },
                {
                    "type": "image_url",
                    # "image_url": {"url": image_url},
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        else:
            content = [
                {
                    "type": "text",
                    "text": f"The robot will perform the task: {task}.{examples_message} Generate a diverse set of exactly {num_instructions} instructions.",
                },
            ]

        # Query the LLM `num_rejection_samples` times (via `n=`) to get several
        # candidate instruction sets, then below we pick the most internally
        # diverse one by CLIP embedding similarity (rejection sampling for
        # diversity, not correctness).
        chat_completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a quality assurance engineer for a robot. Your goal is to come up with instructions that describe the given task correctly, is similar to what human users would possibly give, and yet challenge the robot's capability on accomplishing the task.",
                },
                {"role": "user", "content": content},
            ],
            response_format=InstructionSet,
            n=self.num_rejection_samples,
        )

        all_annotations: List[List[str]] = [
            choice.message.parsed.instructions for choice in chat_completion.choices
        ]
        
        # remove periods and lowercase all prompts
        for i in range(len(all_annotations)):
            all_annotations[i] = [
                instr[:-1].lower() if instr.endswith(".") else instr.lower()
                for instr in all_annotations[i]
            ]

        if self.num_rejection_samples > 1:
            # For each candidate instruction set, embed all its instructions and
            # average their pairwise similarity -- a proxy for how diverse that
            # set is internally (lower mean similarity = more diverse wording).
            all_sim: torch.Tensor = [
                self.embedding_model(annotations).mean().item()
                for annotations in all_annotations
            ]
        else:
            all_sim: torch.Tensor = torch.Tensor([1])  # Dummy

        # Select the instruction set with the lowest mean similarity, i.e. the
        # most diverse set of generated instructions.
        if return_all_annotations:
            return all_annotations[np.argmin(all_sim)], all_annotations

        return all_annotations[np.argmin(all_sim)]


def _vlm_worker(task_and_links, examples):
    """Multiprocessing worker: generate red-team instructions for one task
    using a randomly sampled initial-state image (VLM path). Each worker
    process builds its own CLIPEmbeddingModel/EmbodiedRedTeamModel instance
    since these hold CUDA state that can't be shared across processes."""
    task, links = task_and_links
    # task = list(task_and_links.keys())
    # links = list(task_and_links.values())

    if task in examples:
        spec_examples = examples[task]
    else:
        spec_examples = []

    embedding_model = CLIPEmbeddingModel("cuda")
    red_team = EmbodiedRedTeamModel(embedding_model=embedding_model)
    annotations = red_team(
        task, image_url=random.sample(links, k=1)[0], examples=spec_examples
    )
    return task, annotations


def _lm_worker(task_and_links, examples):
    """Multiprocessing worker: generate red-team instructions for one task
    using text only (no image), for the `use_image=False` path of main()."""
    task, links = task_and_links

    if task in examples:
        spec_examples = examples[task]
    else:
        spec_examples = []
    
    embedding_model = CLIPEmbeddingModel("cuda")
    red_team = EmbodiedRedTeamModel(embedding_model=embedding_model)
    annotations = red_team(
        task, examples=spec_examples
)
    return task, annotations


def main(
    output_path: str,
    examples_path: str = None,
    task_images: str = None,  # "vlm_initial_state_links.json",
    use_image: bool = True,
    max_num_workers: int = 8,
):
    """Generate a diverse red-team instruction set per task (optionally
    grounded on an initial-state image) and write them to a YAML file at
    `output_path` as {task: [instructions]}. Work is parallelized across
    tasks with a process pool since each worker loads its own CLIP model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    multiprocessing.set_start_method("spawn", force=True)
    # with open(task_images, "r") as f:
    #     vlm_task_to_links = json.load(f)
    vlm_task_to_links = task_images

    examples = {}
    if examples_path and os.path.exists(examples_path):
        with open(examples_path, "r") as f:
            examples: Dict[str, List[str]] = yaml.safe_load(f)

    with multiprocessing.Pool(max_num_workers) as pool:
        if use_image:
            results = list(
                tqdm(
                    pool.imap(
                        functools.partial(_vlm_worker, examples=examples),
                        vlm_task_to_links.items(),
                    ),
                    total=len(vlm_task_to_links),
                )
            )
        else:
            results = list(
                tqdm(
                    pool.imap(
                        functools.partial(_lm_worker, examples=examples),
                        vlm_task_to_links.items(),
                    ),
                    total=len(vlm_task_to_links),
                )
            )

    with open(output_path, "w") as f:
        yaml.dump({k: v for k, v in results}, f)
