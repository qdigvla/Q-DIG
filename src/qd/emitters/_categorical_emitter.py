import json
import logging
from typing import Dict, List

import numpy as np
from itertools import combinations
from ribs.archives import ArchiveBase

from src.model.huggingface import BaseLLM
from src.prompt.objective_calculate import similarity

from ._llm_emitter import LLMEmitterBase

logger = logging.getLogger(__name__)

from pydantic import BaseModel
class InstructionSet(BaseModel):
    instructions: List[str]

class CategoricalEmitter(LLMEmitterBase):
    """Categorical Emitter"""

    def __init__(
        self,
        llm: BaseLLM,
        archive: ArchiveBase,
        solution_dim: int,
        initial_solutions: List[Dict],
        measure_cfg: list,
        batch_size: int,
        bounds=None,
        seed=None,
        archive_sample_rate: float = 0.8,
        similarity_req: float = 0.5,
        targeted_exploration: bool = False,
    ):
        LLMEmitterBase.__init__(
            self,
            llm,
            archive,
            batch_size=batch_size,
            solution_dim=solution_dim,
            initial_solutions=initial_solutions,
            measure_cfg=measure_cfg,
            bounds=bounds,
            seed=seed,
        )

        self.prospective_solutions = None
        self.archive_sample_rate = archive_sample_rate
        self.similarity_req = similarity_req
        self.targeted_exploration = targeted_exploration

        self._max_tries = 25  # max requery tries

    def verify(self, prompts: List[str]):
        """Verify the integrity of all the prompts. If there are bad apples in the batch, throw out the entire batch

        TODO: Consider not throwing out the entire batch
        """
        # check if we have enough prompts
        if len(prompts) != self._batch_size * self._solution_dim:
            return False

        for prompt in prompts:
            if prompt is not None and len(prompt) > 400:
                return False
        return True

    def clean_prompts(
        self, prompts: List[str], original_prompts: List[str]
    ) -> List[str]:
        """Clean the prompts by removing any that are too long or empty"""
        cleaned_prompts = []
        for prompt, og in zip(prompts, original_prompts):
            if prompt is not None:
                if (
                    len(prompt) > 0
                    and len(prompt) <= 400
                    and "\n" not in prompt
                    and similarity(prompt, og) > self.similarity_req
                ):
                    cleaned_prompts.append(prompt)
        return cleaned_prompts

    def _sample_mutation_directions(
        self,
        num_prompts: int,
    ):
        """Sample mutation directions from the measure configuration"""
        if self.targeted_exploration:
            # sample from unexplored or low fitness cells in the archive
            data = self.archive.data()
            index = data["index"]  # get all indices in the archive

            total_indices = np.prod(
                [
                    len(indv_measure_cfg["categories"])
                    for indv_measure_cfg in self._measure_cfg
                ]
            )

            sorted_indices = sorted(
                list(range(total_indices)),
                key=lambda x: (
                    # (1) prioritize empty cells
                    1 if x in index else 0,
                    # (2) prioritize low fitness cells
                    (
                        data["objective"][np.where(index == x)[0][0]]
                        if x in index
                        else float("inf")
                    ),
                ),
            )

            selected = sorted_indices[:num_prompts]  # select the lowest fitness cells

            # convert the selected indices to categories
            all_category_directions = []
            for idx in selected:
                category_direction = []
                for i, indv_measure_cfg in enumerate(self._measure_cfg):
                    num_buckets_in_category = len(indv_measure_cfg["categories"])
                    category_index = idx % num_buckets_in_category
                    category_direction.append(
                        indv_measure_cfg["categories"][category_index]
                    )
                    idx //= num_buckets_in_category
                all_category_directions.append(category_direction)
            return all_category_directions
        else:
            # sample uniformly from all categories
            return [
                self._rng.choice(indv_measure_cfg["categories"])
                for indv_measure_cfg in self._measure_cfg
            ]

    def ask(
        self,
        **kwargs,
    ):
        # construct the number of prompts to mutate based on num to keep static
        num_prompts_to_mutate = self._batch_size * self._solution_dim        
        if num_prompts_to_mutate == 0:
            # return np.array([], dtype=self.archive.dtypes["solution"])
            return [[self._initial_solutions[0]["prompt"]]]
        
        list_of_final_outputs = []
        
        # do this 5 times
        for i in range(5):

            # keep querying until all outputs are valid or we exhaust our tries
            tries = 0
            final_outputs = []

            outputs = []
            while tries < self._max_tries and not self.verify(final_outputs):

                # select solutions to mutate
                prospective_solutions = self._random_sample(
                    num_prompts_to_mutate,
                    self.archive_sample_rate,
                    initial_solution_sample_key="prompt",
                )
                self.prospective_solutions = prospective_solutions

                directional_prompts = []
                self._directions = []

                all_category_directions = self._sample_mutation_directions(
                    num_prompts_to_mutate
                )

                for existing_prompt, categories_to_mutate_to in zip(
                    prospective_solutions, all_category_directions
                ):
                    print("To mutate", categories_to_mutate_to)
                    language_categories_to_mutate_to = " and ".join(categories_to_mutate_to)

                    mutation_direction_as_language = f"""A robotic manipulator is given the following task description:

\"{existing_prompt}\"

Your goal is to slightly rephrase this task description so that it reflects the following failure category:

{language_categories_to_mutate_to}

The rephrasing should stay close to the original meaning so the robot can still attempt to complete the task, but it should include the failure characteristic described.
Only output the modified task. Do not return any other text.
The modification must be clearly aligned with the given failure category.""" # Note that "coke can" refers to a Coca Cola can.

                    directional_prompts.append(mutation_direction_as_language)
                    print(mutation_direction_as_language)

                if len(directional_prompts) == self._batch_size:
                    break

                # if using VLM, add the initial image to the messages
                if self.llm.vision:
                    images = [
                        self._initial_solutions[0]["image"]
                        for _ in range(len(directional_prompts))
                    ]
                else:
                    images = [None] * len(directional_prompts)

                all_messages = [  # build out all the newly mutated prompts
                    self.llm.create_message_for_prompt(prompt, image=image)
                    for prompt, image in zip(directional_prompts, images)
                ]
                
                outputs = self.llm.batch_query_messages(all_messages, temp=1.1)    

                clean_outputs = self.clean_prompts(outputs, prospective_solutions)
                final_outputs.extend(clean_outputs)
                final_outputs = final_outputs[
                    : self._batch_size * self._solution_dim
                ]  # limit to the batch size

                # remove any duplicates
                outputs = list(set(outputs))

                tries += 1

                for i, otp in enumerate(final_outputs):
                    print(f"Prompt {i}: {otp}")

            if len(final_outputs) == 0:
                logger.warning(
                    "No valid prompts generated after max tries. Returning empty array."
                )
                return np.array([], dtype=self.archive.dtypes["solution"])

            # classify and categorize all the prompts, and keep the ones that are the highest diversity and lowest
            list_of_final_outputs.append(final_outputs)
            
        scores = []
        for final_outputs in list_of_final_outputs:
            score = sum([
                similarity(combs[0], combs[1])
                for combs in list(combinations(final_outputs, 2))
            ])
            scores.append(score)
        
        # best score is the one with the lowest similarity
        best_idx = np.argmin(np.array(scores))
        final_outputs = list_of_final_outputs[best_idx]
        
        print("TOTAL")
        print(scores)
        print(best_idx)

        self._prompts = np.array(final_outputs, dtype=self.archive.dtypes["solution"])
        return self._prompts.reshape(self._batch_size, self._solution_dim)

    def tell(self, **kwargs):
        # super().tell(**kwargs)
        self.prospective_solutions = None
