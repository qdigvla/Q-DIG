import json
import logging
import os
import subprocess as sp
import sys
import warnings
from collections import deque
from typing import Collection, Dict, List, Optional, Union

import numpy as np
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from itertools import product

from src.model.huggingface import BaseLLM
from src.prompt.objective_calculate import failure_variance

logger = logging.getLogger(__name__)

from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
# from simpler_env.utils.visualization import write_interval_video, write_video

def parse_range_tuple(t):
    """Expand a (min, max, count) config tuple into `count` evenly spaced
    values. Used to turn env_cfg's robot/object init-position ranges into
    concrete grids of candidate positions (see all_init_positions below)."""
    return np.linspace(t[0], t[1], int(t[2]))

import cv2
def save_video(images, episode_idx, fps=24):
    # Build output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, f"video{episode_idx}.mp4")

    # Get frame size
    height, width, _ = np.array(images[0]).shape

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_filename}")

    # Write frames
    for pil_img in images:
        np_img = np.array(pil_img)
        frame = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_filename}")

from pydantic import BaseModel
class InstructionSet(BaseModel):
    instructions: List[str]

class SimplerEnvBenchmark:
    """SimplerEnv Benchmark Evaluator

    Args:
        n_evals: Number of times to evaluate each solution during real evaluation.
            Evals are rolled out in the same batch and aggregated by the manager.
        module_cfg: Config passed to the SteakhouseModule. Can't use hydra's instantiate
            since we want to instantiate a new module for each worker.
        objective: Name of the objective function (it should be defined in
            objective_calculate.py).
        aggregation_type: Method of aggregating results. Currently ignored since QD +
            Steakhouse used a mix of aggregation for different metrics.
        ml_actions: Mapping of agent index to the action taken by the ML model.
        rng: Random generator. Can be set later. Uses `np.random.default_rng()` by
            default.
    """

    MIN_SCORE = 0

    def __init__(
        self,
        n_evals: int,
        measures: list,
        preprocess: bool = False,
        aggregation_type: str = "mean",
        inference_temperature: float = 1.1,
        rng: np.random.Generator = None,
        env_cfg: Optional[DictConfig] = None,
        task_id: int = 0,
        env_names: list = None,
        model_kwargs: dict = None,
        load_model: bool = True
    ):
        self.rng = np.random.default_rng(42)
        self.n_evals = n_evals
        self.measures = measures
        self.preprocess = preprocess

        self.aggregation_type = aggregation_type

        self.inference_temperature = inference_temperature

        self.env_cfg = env_cfg

        control_mode = get_robot_control_mode(env_cfg.robot_name, env_cfg.model_family)
        self.env_kwargs = dict(
            obs_mode="rgbd",
            robot=self.env_cfg.robot_name,
            sim_freq=self.env_cfg.sim_freq,
            control_mode=control_mode,
            control_freq=self.env_cfg.control_freq,
            max_episode_steps=self.env_cfg.max_episode_steps,
            scene_name=self.env_cfg.scene_name,
            camera_cfgs={"add_segmentation": True},
            rgb_overlay_path=getattr(self.env_cfg, "rgb_overlay_path", None),
        )

        if model_kwargs is None:
            model_kwargs = {}

        # Dispatch policy construction by model_family. load_model can be
        # False to skip this entirely (e.g. when the manager is only being
        # used to build/inspect environments, not run inference).
        if load_model:
            if env_cfg.model_family == "openvla":
                from simpler_env.policies.openvla.openvla_model import OpenVLAInference
                self.model = OpenVLAInference(
                    saved_model_path=env_cfg.pretrained_checkpoint,
                    policy_setup=env_cfg.policy_setup,
                    action_scale=env_cfg.action_scale,
                    **model_kwargs
                )
            elif env_cfg.model_family == "openvla-server":
                # OpenVLA-OFT spawned as a remote HTTP server (the official one from
                # the upstream OpenVLA-OFT repo, https://github.com/moojink/openvla-oft,
                # `vla-scripts/deploy.py`) instead of loaded in-process.
                from simpler_env.policies.openvla.openvla_server import OpenVLAServerInference
                self.model = OpenVLAServerInference(
                    saved_model_path=env_cfg.pretrained_checkpoint,
                    policy_setup=env_cfg.policy_setup,
                    action_scale=env_cfg.action_scale,
                    **model_kwargs
                )
            elif env_cfg.model_family == "openpi":
                from simpler_env.policies.openpi.pi0_or_fast import OpenPiFastInference
                self.model = OpenPiFastInference(
                    saved_model_path=env_cfg.pretrained_checkpoint,
                    policy_setup=env_cfg.policy_setup,
                    action_scale=env_cfg.action_scale,
                    **model_kwargs
                )
            elif env_cfg.model_family == "gr00t":
                # GR00T; pass host/port in model_kwargs to query a remote server
                # (Isaac-GR00T/gr00t/eval/run_gr00t_server.py) instead of loading in-process.
                from simpler_env.policies.gr00t.gr00t_model import Gr00tInference
                self.model = Gr00tInference(
                    saved_model_path=env_cfg.pretrained_checkpoint,
                    policy_setup=env_cfg.policy_setup,
                    action_scale=env_cfg.action_scale,
                    **model_kwargs
                )
            elif env_cfg.model_family == "diffusion":
                from simpler_env.policies.diffusion.model import DiffusionInference
                self.model = DiffusionInference(
                    saved_model_path=env_cfg.pretrained_checkpoint,
                    policy_setup=env_cfg.policy_setup,
                    action_scale=env_cfg.action_scale,
                    **model_kwargs
                )
        else:
            self.model = None

        self.task_id = task_id
        self.env_names = env_names
        
        self.all_init_positions = []

        # self env_cfg init
        self.robot_init_x = parse_range_tuple(self.env_cfg.robot_init.x)
        self.robot_init_y = parse_range_tuple(self.env_cfg.robot_init.y)


        if self.env_cfg.get("obj_init") is not None:
            self.obj_init_x = parse_range_tuple(self.env_cfg.obj_init.x_range)
            self.obj_init_y = parse_range_tuple(self.env_cfg.obj_init.y_range)


        # Generate all combinations of initial positions. Each entry is either
        # (robot_x, robot_y) or, when object position is also varied,
        # (robot_x, robot_y, obj_x, obj_y). evaluate() cycles through this list
        # (wrapping via modulo) to pick a deterministic-but-varied start state
        # per episode.
        self.all_init_positions = list(
            product(self.robot_init_x, self.robot_init_y)
        )

        if self.env_cfg.get("obj_init") is not None:
            obj_positions = list(product(self.obj_init_x, self.obj_init_y))
            self.all_init_positions = [
                (robot_x, robot_y, obj_x, obj_y)
                for (robot_x, robot_y) in self.all_init_positions
                for (obj_x, obj_y) in obj_positions
            ]

        # Randomize the order of all_init_positions
        self.rng.shuffle(self.all_init_positions)

        if self.measures is None:
            return

        # Build the classification prompt used in evaluate() to assign each
        # emitted prompt a measure/category, which QD uses to place solutions
        # in the archive.
        categories_and_options = ""
        for i, meas in enumerate(self.measures):
            cats = meas["categories"]
            categories_and_options += f"\nCategory {i}: " + ", ".join(cats)

        if len(self.measures) == 2:
            example_output = f"""["{self.measures[0]['categories'][3]}", "{self.measures[1]['categories'][2]}"]"""
        elif len(self.measures) == 1:
            example_output = f"""["{self.measures[0]['categories'][3]}"]"""

        # Note: this first classification_prompt is built but immediately
        # overwritten below by the version that classifies the *diff* between
        # the new and original prompt, rather than the prompt alone. Kept for
        # reference / possible reuse.
        classification_prompt = f"""
You are a classifier that assigns labels to a given prompt according to {len(measures)} distinct categories. Each category has its own set of options.
For the prompt provided, return your classification in JSON format as a **list** of exactly one selected option *per category*, in order.
Even if the prompt doesn't fit in any of the options for a category, you must select the best option from each category.
If the prompt fits in multiple options in one category, choose the option it fits *best* in.

### Prompt to Classify:
{{prompt}}

### Categories and Options:
{categories_and_options}

### Output Format (example):
{example_output}
"""

        classification_prompt = f"""You are a classifier that determines the difference between two given prompts according to 1 distinct category. Each category has its own set of options.

You will be given:
- An **original prompt**
- A **new prompt**

Your task: Identify the main difference in wording, style, or tone between the new prompt and the original prompt.

For the given pair of prompts, return your classification in JSON format as a **list** containing exactly one selected option from each category, in order.
Even if the difference does not perfectly match any option for a category, you must select the best-fitting option.
If the difference could match multiple options in one category, choose the option it fits *best*.

### Original Prompt:
{{old_prompt}}

### New Prompt:
{{prompt}}

### Categories and Options:
{categories_and_options}

### Output Format (example):
{example_output}"""

        preprocess_prompt = """A robotic manipulator is given the following task description:

\"{prompt}\"

Simplify the task description to ensure that it is clear and easily interpretable. Remove any unnecessary details or complexity, while retaining the core meaning of the task.
The robotic manipulator should still be able to understand the task and complete it as before. Only output the modified task, do not return any other text."""

        self.classification_prompt = classification_prompt
        self.preprocess_prompt = preprocess_prompt


    def evaluate(
        self,
        prompts: np.ndarray,
        llm: BaseLLM,
        debug: bool = False,
        render: bool = False,
        classify: bool = True,
        task_id: int = None,
    ) -> dict[str, str]:
        """Pipeline that takes solutions and evaluates it. Pipeline will
        evaluate all solutions continuously in batch forward pass.

        Args:
            sols: Emitted solution.
            agents_cfg: Hydra config of the agents.
            model: Model to query for prompts.
            debug: Whether to print debug logs.

        This function will complete the setup for the environment, which is then
        passed into _run_evaluation_batch to run the evaluation.

        Returns:
            Results of the evaluation.
        """
        results = []

        # start with the basic task_id
        task_id = task_id or self.task_id

        print("Current prompts:", prompts)
        original_prompts = prompts

        if self.preprocess:
            # Optionally rewrite each candidate prompt through the LLM (using
            # self.preprocess_prompt) before rolling it out, grounded on the
            # initial-state image.
            init_sol = self.get_initial_solution(task_id=task_id)[0]

            all_messages = [
                llm.create_message_for_prompt(
                    # self.preprocess_prompt.replace("prompt", p[0]),
                    self.preprocess_prompt.format(prompt=p[0]),
                    image=init_sol["image"],
                )
                for p in prompts
            ]

            prompt_results = llm.batch_query_messages(all_messages)
            prompts = [[p] for p in prompt_results]

            print("Updating prompts for task", task_id, "with results", prompt_results)

        if self.env_cfg.additional_env_build_kwargs != None:
            additional_env_build_kwargs = self.env_cfg.additional_env_build_kwargs
        else:
            additional_env_build_kwargs = {}

        if self.env_cfg.enable_raytracing:
            ray_tracing_dict = {"shader_dir": "rt"}
            ray_tracing_dict.update(additional_env_build_kwargs)
            # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
            additional_env_build_kwargs = ray_tracing_dict

        # One ManiSkill2 env is built per task_id and reused across all
        # episodes/prompts for this evaluate() call (only reset, not rebuilt,
        # between episodes).
        env = build_maniskill2_env(
            self.env_names[task_id],
            **additional_env_build_kwargs,
            **self.env_kwargs,
        )

        for prompt in prompts:
            task_description = str(prompt[0])
            original_description = env.get_language_instruction()

            logger.info(f"\nTask: {task_description}\n")
            result_info = {"failed": False}

            # Start episodes
            task_successes = 0
            for episode_idx in tqdm.tqdm(range(self.n_evals)):
                tqdm.tqdm.write(f"Episode {episode_idx}, Successes {task_successes}")
                
                # select a starting position (loop if not enough positions)
                init_position = self.all_init_positions[
                    episode_idx % len(self.all_init_positions)
                ]

                # initialize environment
                env_reset_options = {
                    "robot_init_options": {
                        "init_xy": np.array([init_position[0], init_position[1]]),
                        "init_rot_quat": self.env_cfg.robot_init.rot_quat_center,
                    }
                }

                if self.env_cfg.get("obj_init") is not None:
                    # Object position is being varied over a grid (see
                    # all_init_positions), so pull its x/y from this episode's
                    # sampled init_position tuple.
                    obj_variation_mode = "xy"

                    env_reset_options["obj_init_options"] = {
                        "init_xy":np.array([init_position[2], init_position[3]]),
                    }
                else:
                    # No object-position range configured: fall back to a
                    # fixed episode_id from env_cfg (env_cfg must define
                    # obj_episode_id in this case).
                    assert self.env_cfg.get("obj_episode_id") is not None
                    obj_variation_mode = "episode"
                    env_reset_options["obj_init_options"] = {
                        "episode_id": self.env_cfg.obj_episode_id,
                    }

                obs, _ = env.reset(options=env_reset_options)
                # obs, _ = env.reset()

                # for long-horizon environments, we check if the current subtask is the final subtask
                # Ignoring long-horizon environemnts for now
                is_final_subtask = True

                # Initialize logging
                image = get_image_from_maniskill2_obs_dict(
                    env, obs, camera_name=self.env_cfg.get("obs_camera_name")
                )
                images = [image]
                predicted_actions = []
                predicted_terminated, done, truncated = False, False, False

                # Initialize model
                self.model.reset(task_description)

                timestep = 0
                success = "failure"
                # action_ensemble = model.action_ensemble_temp  if hasattr(model, "action_ensemble") else "none"

                # Run episode
                success = False
                try:
                    # Step the environment
                    task_descriptions = []
                    while not (predicted_terminated or truncated):
                        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
                        
                        if self.env_cfg.model_family == "openvla":
                            raw_action, action = self.model.step(
                                image,
                                task_description,
                                obs=obs
                            )
                        elif self.env_cfg.model_family == "openpi":
                            raw_action, action = self.model.step(
                                image,
                                task_description,
                                eef_pos=obs["proprio"]
                            )
                        elif self.env_cfg.model_family == "diffusion":
                            # compress the image to 224x224
                            image_pil = Image.fromarray(image)
                            image_pil = image_pil.resize((224, 224))
                            image = np.array(image_pil)
                            
                            raw_action, action = self.model.step(
                                image,
                                task_description,
                                obs=obs
                            )

                        predicted_actions.append(raw_action)
                        predicted_terminated = bool(action["terminate_episode"][0] > 0)
                        if predicted_terminated:
                            if not is_final_subtask:
                                # The model predicted episode termination, but we're
                                # not on the final subtask of a long-horizon task --
                                # treat it as "finished this subtask" and advance
                                # instead of actually ending the episode.
                                predicted_terminated = False
                                env.advance_to_next_subtask()

                        # step the environment
                        obs, reward, done, truncated, info = env.step(
                            np.concatenate(
                                [
                                    action["world_vector"],
                                    action["rot_axangle"],
                                    action["gripper"],
                                ]
                            ),
                        )

                        if done:
                            success = True
                            break

                        # new_task_description = env.get_language_instruction()
                        # if new_task_description != task_description:
                        #     task_description = new_task_description
                        #     print(task_description)
                        is_final_subtask = env.is_final_subtask()

                        # print(timestep, info)

                        image = get_image_from_maniskill2_obs_dict(
                            env, obs, camera_name=self.env_cfg.get("obs_camera_name")
                        )
                        images.append(image)
                        task_descriptions.append(task_description)
                        timestep += 1

                    # save video
                    # save_video(images, episode_idx, fps=5)

                except Exception as e:
                    # Any failure mid-rollout aborts this episode and marks the
                    # whole task result as failed; add_failed_info() downstream
                    # checks result_info["failed"].
                    print(f"Caught exception: {e}")
                    # traceback
                    e.with_traceback()
                    logger.info(f"Caught exception: {e}\n")
                    result_info["failed"] = True
                    result_info["log_message"] = e
                    break

                if success:
                    task_successes += 1

            # calculate the objective as the variance of the failure
            objective = failure_variance(self.n_evals, task_successes)

            result_info.update(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "original_description": original_description,
                    "objective": objective,
                    "task_successes": task_successes,
                    "n_evals": self.n_evals,
                }
            )
            results.append(result_info)
            print(result_info)

        # if not classifying, return results directly
        if not classify:
            return results

        init_sol = self.get_initial_solution(task_id=task_id)[0]

        all_categories = []
        for i, prompt in enumerate(original_prompts):
            # Retry until the LLM returns a valid category combination (`good`
            # below) since classification is required to place this solution
            # in the QD archive; on error, back off and retry.
            success = False
            while not success:
                try:
                    # copy self.classification_prompt
                    base_prompt = self.classification_prompt.format(
                        prompt=prompt[0], old_prompt=init_sol["prompt"]
                    )

                    # Create classification message for this prompt
                    message = llm.create_message_for_prompt(base_prompt)
                    
                    chat_completion = llm.client.beta.chat.completions.parse(
                        model=llm.model_name,
                        messages=message,
                        response_format=InstructionSet,
                    )
                    
                    response: List[List[str]] = [
                        choice.message.parsed.instructions for choice in chat_completion.choices
                    ][0]

                    # Parse and validate
                    parsed = response # json.loads(response)

                    good = all(
                        cat in self.measures[j]["categories"]
                        for j, cat in enumerate(parsed)
                    )

                    if good:
                        print("Classification successful for prompt:", parsed)
                        all_categories.append(parsed)
                        results[i]["measures"] = parsed
                        success = True

                except Exception as e:
                    print(f"Error parsing prompt {i}: {e}")
                    print(prompt)
                    print(response)
                    import time
                    time.sleep(240)

        return results

    def actual_qd_score(self, objs: List):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= self.MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)

    @staticmethod
    def add_failed_info(sol, result) -> dict:
        """Returns a dict containing relevant information about failed levels.

        Args:
            sol: Emitted solution.
            result: Evaluation result.

        Returns:
            Dict with failed level information.
        """
        failed_level_info = {
            "solution": sol,
            "log_message": result["log_message"],
        }
        return failed_level_info

    def get_initial_solution(self, task_id: int = None) -> dict:
        """Get the initial solution from the environment
        Returns:
            Initial solution as a dict, with keys 'prompt' and 'image'
        """

        task_id = task_id or self.task_id

        additional_env_build_kwargs = {}

        # create the environment associated with the task id
        if self.env_cfg.enable_raytracing:
            ray_tracing_dict = {"shader_dir": "rt"}
            ray_tracing_dict.update(additional_env_build_kwargs)
            # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
            additional_env_build_kwargs = ray_tracing_dict
        env = build_maniskill2_env(
            self.env_names[task_id],
            **additional_env_build_kwargs,
            **self.env_kwargs,
        )
        obs, _ = env.reset()  # use default reset for initial solution

        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=self.env_cfg.get("obs_camera_name")
        )
        image = Image.fromarray(image)

        task_description = env.get_language_instruction()

        # return prompt and image, in-case of VLA
        return [{"prompt": task_description, "image": image}]

    def get_num_tasks(self):
        """Return the total number of tasks available."""
        return len(self.env_names)