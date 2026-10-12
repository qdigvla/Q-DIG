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
from libero.libero import benchmark
from omegaconf import DictConfig
from PIL import Image

from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from src.model.huggingface import BaseLLM
from src.prompt.objective_calculate import (
    failure_and_similarity,
    failure_rate,
    variance_and_similarity,
    failure_variance,
    similarity
)

from .LIBERO.libero_utils import (  # libero_utils; save_rollout_video,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video
)
from .LIBERO.openvla_utils import (
    get_action_from_server,
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
)
from .LIBERO.robot_utils import get_action  # DATE_TIME,
from .LIBERO.robot_utils import (
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
)
from .LIBERO.run_libero_eval import (
    check_unnorm_key,
    prepare_observation,
    process_action,
)

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

logger = logging.getLogger(__name__)

from pydantic import BaseModel
class InstructionSet(BaseModel):
    instructions: List[str]

class LIBEROBenchmark:
    """LIBERO Benchmark Evaluator

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
    ):
        self.rng = rng or np.random.default_rng()
        self.n_evals = n_evals
        self.measures = measures
        self.preprocess = preprocess

        self.aggregation_type = aggregation_type

        self.inference_temperature = inference_temperature

        self.env_cfg = env_cfg

        # [OpenVLA] Set action un-normalization key. The action head outputs
        # normalized actions, so we need a key into `model.norm_stats` matching
        # the dataset/task suite the policy was trained on in order to recover
        # real-world action scales.
        self.unnorm_key = env_cfg.task_suite_name

        # # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # # with the suffix "_no_noops" in the dataset name)
        # if (
        #     self.unnorm_key not in model.norm_stats
        #     and f"{self.unnorm_key}_no_noops" in model.norm_stats
        # ):
        #     self.unnorm_key = f"{self.unnorm_key}_no_noops"
        # assert (
        #     self.unnorm_key in model.norm_stats
        # ), f"Action un-norm key {self.unnorm_key} not found in VLA `norm_stats`!"
        
        
        # Dispatch model loading by model_family. Each branch sets up whatever
        # state `evaluate()` later needs to query that family's policy -- either
        # an in-process model (openvla, gr00t client handle, diffusion) or just
        # host/port info for a remote inference server (openvla-server, openpi).
        if env_cfg.model_family == "openvla":
            # Load model
            model = get_model(env_cfg)
            self.model = model
            
            # Load proprio projector if needed
            proprio_projector = None
            if env_cfg.use_proprio:
                proprio_projector = get_proprio_projector(
                    env_cfg,
                    model.llm_dim,
                    proprio_dim=8,  # 8-dimensional proprio for LIBERO
                )

            # Load action head if needed
            action_head = None
            if env_cfg.use_l1_regression or env_cfg.use_diffusion:
                action_head = get_action_head(env_cfg, model.llm_dim)

            # Load noisy action projector if using diffusion
            noisy_action_projector = None
            if env_cfg.use_diffusion:
                noisy_action_projector = get_noisy_action_projector(env_cfg, model.llm_dim)

            # Get OpenVLA processor if needed
            processor = None
            if env_cfg.model_family == "openvla":
                processor = get_processor(env_cfg)
                check_unnorm_key(env_cfg, model)

            # set variables locally
            self.action_head = action_head
            self.proprio_projector = proprio_projector
            self.noisy_action_projector = noisy_action_projector
            self.processor = processor
        elif env_cfg.model_family == "openvla-server":
            # OpenVLA-OFT spawned as a remote HTTP server (the official one from the
            # upstream OpenVLA-OFT repo, https://github.com/moojink/openvla-oft,
            # `vla-scripts/deploy.py`) instead of loaded in-process. No local
            # model/processor to load here -- we just remember where to send requests.
            self.host = env_cfg.host
            self.port = env_cfg.port

        elif env_cfg.model_family == "openpi":
            # load openpi client
            self.client = _websocket_client_policy.WebsocketClientPolicy(env_cfg.host, env_cfg.port)

        elif env_cfg.model_family == "gr00t":
            # Isaac-GR00T is spawned as its own remote server (see
            # Isaac-GR00T/gr00t/eval/run_gr00t_server.py) and queried over ZMQ here.
            from gr00t.policy.server_client import PolicyClient
            self.model = PolicyClient(host=env_cfg.host, port=env_cfg.port)
            
        elif env_cfg.model_family == "diffusion":
            from simpler_env.policies.diffusion.model import DiffusionInference
            self.model = DiffusionInference(
                saved_model_path=env_cfg.pretrained_checkpoint,
                policy_setup=env_cfg.policy_setup,
                action_scale=env_cfg.action_scale,
                device=self.device
            )

        # Initialize LIBERO task suite
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = self.benchmark_dict[self.env_cfg.task_suite_name]()

        self.task_id = task_id

        if self.measures is None:
            return

        # Build the classification prompt used in evaluate() to assign each
        # emitted prompt a measure/category (one selection per entry in
        # self.measures), which QD uses to place solutions in the archive.
        categories_and_options = ""
        for i, meas in enumerate(self.measures):
            cats = meas["categories"]
            categories_and_options += f"\nCategory {i}: " + ", ".join(cats)

        if len(self.measures) == 2:
            example_output = f"""["{self.measures[0]['categories'][3]}", "{self.measures[1]['categories'][2]}"]"""
        elif len(self.measures) == 1:
            example_output = f"""["{self.measures[0]['categories'][3]}"]"""

        # Note: this first classification_prompt is built but immediately
        # overwritten by the one below (which classifies the *diff* between the
        # new and original prompt rather than the prompt in isolation). Kept for
        # reference / reuse -- not dead code, just currently unused.
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
        self.debug = debug

        results = []

        # Get expected image dimensions. get_image_resize_size() currently
        # returns the OpenVLA resize size for any model_family; since all
        # families we support so far share the same 224x224 input size this is
        # fine, but revisit if a new family needs a different resolution.
        resize_size = get_image_resize_size(self.env_cfg)

        # start with the basic task_id
        task_id = task_id or self.task_id

        print("Current prompts:", prompts)
        original_prompts = prompts

        if self.preprocess:
            # Optionally rewrite each candidate prompt through the LLM (using
            # self.preprocess_prompt) before rolling it out, to strip
            # unnecessary complexity while keeping task semantics intact. Uses
            # the initial-state image for grounding context.
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
            

        for prompt in prompts:
            all_gradients = []
            all_gradients_original = []
            success_list = []

            description = str(prompt[0])

            logger.info(f"\nTask: {description}\n")

            # Get task
            task = self.task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = self.task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(
                task, self.env_cfg.model_family, resolution=256
            )
            original_description = task_description
            task_description = description

            result_info = {"failed": False}

            # Start episodes
            task_successes = 0
            for episode_idx in tqdm.tqdm(range(self.n_evals)):
                grad_info = []
                grad_info_original = []
                
                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                # Per-task-suite step budget. Each cap is set just above the
                # longest demo in that suite's training data (noted inline), so
                # a rollout that hasn't finished by max_steps is counted as a
                # failure rather than run indefinitely.
                if self.env_cfg.task_suite_name == "libero_spatial":
                    max_steps = 220  # longest training demo has 193 steps
                elif self.env_cfg.task_suite_name == "libero_object":
                    max_steps = 280  # longest training demo has 254 steps
                elif self.env_cfg.task_suite_name == "libero_goal":
                    max_steps = 100 # 300  # longest training demo has 270 steps
                elif self.env_cfg.task_suite_name == "libero_10":
                    max_steps = 520  # longest training demo has 505 steps
                elif self.env_cfg.task_suite_name == "libero_90":
                    max_steps = 400  # longest training demo has 373 steps
                    
                    
                # Action queue holds the current action chunk predicted by the
                # policy; it's refilled by querying the model whenever it runs
                # dry (see "If action queue is empty" below). openvla/-server use
                # a bounded deque sized to the open-loop replan horizon; the
                # other families replan every chunk so an unbounded deque is
                # fine (it's drained fully before the next query).
                if self.env_cfg.model_family == "openvla":
                    # Initialize action queue
                    if self.env_cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
                        print(
                            f"WARNING: cfg.num_open_loop_steps ({self.env_cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
                            f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
                            "both speed and success rate), we recommend executing the full action chunk."
                        )
                    action_queue = deque(maxlen=self.env_cfg.num_open_loop_steps)
                elif self.env_cfg.model_family == "openvla-server":
                    action_queue = deque(maxlen=self.env_cfg.num_open_loop_steps)
                elif self.env_cfg.model_family == "openpi":
                    action_queue = deque()
                    
                elif self.env_cfg.model_family == "gr00t":
                    action_queue = deque()
                    
                elif self.env_cfg.model_family == "diffusion":
                    action_queue = deque()


                # Run episode
                success = False
                try:
                    while t < max_steps + self.env_cfg.num_steps_wait:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < self.env_cfg.num_steps_wait:
                            obs, reward, done, info = env.step(
                                get_libero_dummy_action(self.env_cfg.model_family)
                            )
                            t += 1
                            continue
                        
                        # resolve image input based on model
                        if self.env_cfg.model_family == "openpi":
                            # Get preprocessed image
                            # IMPORTANT: rotate 180 degrees to match train preprocessing
                            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(img, resize_size, resize_size)
                            )
                            wrist_img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
                            )
                            
                        elif self.env_cfg.model_family in ("openvla", "openvla-server"):
                            # Prepare observation
                            observation, img = prepare_observation(obs, resize_size)

                        elif self.env_cfg.model_family == "gr00t":
                            # Prepare observation
                            observation, img = prepare_observation(obs, resize_size)

                        replay_images.append(img)

                        # If action queue is empty, requery model
                        if len(action_queue) == 0:
                            if self.env_cfg.model_family == "openpi":
                                # Finished executing previous action chunk -- compute new chunk
                                # Prepare observations dict
                                element = {
                                    "observation/image": img,
                                    "observation/wrist_image": wrist_img,
                                    "observation/state": np.concatenate(
                                        (
                                            obs["robot0_eef_pos"],
                                            quat2axisangle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"],
                                        )
                                    ),
                                    "prompt": str(task_description),
                                }

                                # Query model to get action
                                action_chunk = self.client.infer(element)["actions"]
                                assert (
                                    len(action_chunk) >= self.env_cfg.replan_steps
                                ), f"We want to replan every {self.env_cfg.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                                actions = action_chunk[: self.env_cfg.replan_steps]
                                
                            elif self.env_cfg.model_family == "openvla":
                                # actions2, gradients2 = get_action(
                                #     self.env_cfg,
                                #     self.model,
                                #     observation,
                                #     original_description,
                                #     processor=self.processor,
                                #     action_head=self.action_head,
                                #     proprio_projector=self.proprio_projector,
                                #     noisy_action_projector=self.noisy_action_projector,
                                #     use_film=self.env_cfg.use_film,
                                # )
                                # grad_info_original.append(gradients2)
                                
                                # Query model to get action
                                actions, gradients = get_action(
                                    self.env_cfg,
                                    self.model,
                                    observation,
                                    task_description,
                                    processor=self.processor,
                                    action_head=self.action_head,
                                    proprio_projector=self.proprio_projector,
                                    noisy_action_projector=self.noisy_action_projector,
                                    use_film=self.env_cfg.use_film,
                                    # original_prompt_vla_actions=gradients2,
                                )
                                # gradients2["gradient_actions"] = None
                                # gradients1["gradient_actions"] = None
                                
                                # if "gradient_actions" in gradients2:
                                #     del gradients2["gradient_actions"]
                                    
                                if "gradient_actions" in gradients:
                                    del gradients["gradient_actions"]
                                
                                grad_info.append(gradients)
                            elif self.env_cfg.model_family == "openvla-server":
                                # Query remote OpenVLA-OFT server instead of the local model
                                actions = get_action_from_server(
                                    observation,
                                    task_description,
                                    host=self.host,
                                    port=self.port,
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
                                actions = [action]  # wrap in list

                            elif self.env_cfg.model_family == "gr00t":
                                # get the actions
                                # Prepare observation in the format expected by GR00T policy
                                # GR00T expects video input with shape (B, T, H, W, C)
                                # We need to add batch dimension (1) and temporal dimension (1) to the image
                                video_input = observation["full_image"][None, None, ...]  # Add batch and time dims
                                wrist_input = observation["wrist_image"][None, None, ...]  # Add batch and time dims
                                
                                axang = quat2axisangle(obs["robot0_eef_quat"])

                                gripper = obs["robot0_gripper_qpos"][0]

                                # GR00T expects a flat dict of per-channel state/video
                                # keys (not a single "state" vector like openpi), each
                                # shaped (B, T, ...) with explicit batch and time dims.
                                actions, _ = self.model.get_action({
                                    "video.image": video_input,
                                    "video.wrist_image": wrist_input,
                                    "state.x": np.array([obs["robot0_eef_pos"][0]], dtype=np.float32)[None, None, ...],
                                    "state.y": np.array([obs["robot0_eef_pos"][1]], dtype=np.float32)[None, None, ...],
                                    "state.z": np.array([obs["robot0_eef_pos"][2]], dtype=np.float32)[None, None, ...],
                                    "state.roll": np.array([axang[0]], dtype=np.float32)[None, None, ...],
                                    "state.pitch": np.array([axang[1]], dtype=np.float32)[None, None, ...],
                                    "state.yaw": np.array([axang[2]], dtype=np.float32)[None, None, ...],
                                    "state.gripper": np.array(obs["robot0_gripper_qpos"], dtype=np.float32)[None, None, ...],
                                    "annotation.human.action.task_description": [task_description]
                                })
                                
                                for i in range(len(actions["action.gripper"])):
                                    # Remap GR00T's gripper convention (1 = open, 0 = close)
                                    # to LIBERO's (-1 = open, 1 = close) via 1 - 2*x.
                                    actions["action.gripper"][i] = 1 - 2 * actions["action.gripper"][i]

                                # Reassemble the per-key GR00T output dict into a single
                                # (T, 7) action array in the [x, y, z, roll, pitch, yaw,
                                # gripper] order LIBERO's env.step() expects, keeping
                                # only the first 5 timesteps of the predicted chunk.
                                keys = ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"]
                                arr = [
                                    actions[key] for key in keys
                                ]
                                arr = np.array(arr).T
                                actions = arr.reshape(-1, 7)[:5]#.tolist()

                            action_queue.extend(actions)

                        # Get action from queue
                        action = action_queue.popleft()

                        if self.env_cfg.model_family in ("openvla", "openvla-server"):
                            # Process action
                            action = process_action(action, "openvla")

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            success = True
                            break
                        t += 1

                except Exception as e:
                    # Any failure mid-rollout (sim error, server/model error, etc.)
                    # aborts this episode and marks the whole task result as failed
                    # rather than silently skipping -- downstream code checks
                    # result_info["failed"] via add_failed_info().
                    print(f"Caught exception: {e}")
                    # traceback
                    e.with_traceback()
                    logger.info(f"Caught exception: {e}\n")
                    result_info["failed"] = True
                    result_info["log_message"] = e
                    break

                if success:
                    task_successes += 1
                    success_list.append(1)
                else:
                    success_list.append(0)
                
                if render:
                    # save the video
                    save_rollout_video(
                        replay_images,
                        episode_idx,
                        success,
                        task_description
                    )
                    
                all_gradients.append(grad_info)
                all_gradients_original.append(grad_info_original)
                
                tqdm.tqdm.write(
                    f"Success: {task_successes}"
                )

            if original_description is not None:
                # When we have the original (un-rephrased) task description to
                # compare against, use failure_variance so the objective also
                # rewards/penalizes how much this prompt's behavior diverges
                # from the baseline, not just raw success rate.
                objective = failure_variance(
                    self.n_evals,
                    task_successes,
                )
            else:
                objective = failure_rate(self.n_evals, task_successes)

            result_info.update(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "objective": objective,
                    "task_successes": task_successes,
                    "n_evals": self.n_evals,
                    "gradient_info": all_gradients,
                    "success_list": success_list,
                    "gradient_info_original": all_gradients_original,
                }
            )
            # print("NEW RESULT INFO")
            # print(result_info)
            results.append(result_info)

        # if not classifying, return results directly
        if not classify:
            return results

        init_sol = self.get_initial_solution(task_id=task_id)[0]

        all_categories = []
        for i, prompt in enumerate(original_prompts):
            # Retry classification until the LLM returns a valid category
            # combination (checked by `good` below) or we hit a transient
            # parsing/API error, in which case we back off and retry rather
            # than giving up (classification is required to place this
            # solution in the QD archive).
            success = False
            while not success:
                try:
                    # copy self.classification_prompt
                    base_prompt = self.classification_prompt.format(
                        prompt=prompt[0], old_prompt=init_sol["prompt"]
                    )
                    # base_prompt = base_prompt.replace(
                    #     "{prompt}", prompt[0]
                    # )  # replace {prompt} with the actual prompt

                    # Create classification message for this prompt
                    message = [llm.create_message_for_prompt(base_prompt)]

                    # # Query LLM for this single message
                    # response = llm.batch_query_messages(message)[0]
                    
                    # # remove anything before `[` or after `]`
                    # response = 
                    
                    chat_completion = llm.client.beta.chat.completions.parse(
                        model=llm.model_name,
                        messages=message[0],
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
                    # e.with_traceback()

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

        task = self.task_suite.get_task(task_id)
        env, task_description = get_libero_env(
            task, self.env_cfg.model_family, resolution=256
        )

        # Get default LIBERO initial states
        initial_states = self.task_suite.get_task_init_states(task_id)

        # Set initial states
        obs = env.set_init_state(initial_states[0])

        # get the image
        # resize_size = get_image_resize_size(self.env_cfg)
        # img = get_libero_image(obs) # , resize_size)

        # step the environment to get the initial image
        for _ in range(self.env_cfg.num_steps_wait):
            obs, reward, done, info = env.step(
                get_libero_dummy_action(self.env_cfg.model_family)
            )

        image = Image.fromarray(obs["agentview_image"][::-1])

        # return prompt and image, in-case of VLA
        return [{"prompt": task_description, "image": image}]

    def get_num_tasks(self):
        """Return the total number of tasks available."""
        return self.task_suite.get_num_tasks()