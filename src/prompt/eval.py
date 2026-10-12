
import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import enable_full_determinism, set_seed

from src.utils.hydra_utils import define_resolvers

from hydra.utils import instantiate
import yaml
import os
import json
import pickle


logger = logging.getLogger(__name__)

def evaluate(cfg):
    """Standalone Hydra-driven evaluator: replays a saved set of candidate
    prompts (from a previous QDIG/ERT run, pointed to by `+eval_dir=`) through
    the benchmark's rollout loop and dumps success/gradient info, instead of
    running the QD ask/tell loop. Intended to be invoked with Hydra `+`
    overrides on the CLI (e.g. `+eval_dir=... +original=true +cv_fold=0`)
    rather than via normal config fields, since these are eval-specific knobs
    not part of the regular manager config schema.
    """
    rng = np.random.default_rng(seed=cfg["seed"])
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    logdir = hydra_cfg.runtime.output_dir

    # Parse ad-hoc `+key=value` Hydra overrides (rather than schema'd config
    # fields) since these only apply to this standalone eval script.
    overrides = hydra_cfg.overrides.task
    eval_dir = None
    original = False
    cv_fold = None

    for ov in overrides:
        if ov.startswith("+eval_dir="):
            eval_dir = ov.split("=", 1)[1]
        elif ov.startswith("+original="):
            original = ov.split("=", 1)[1].lower() == "true"
        elif ov.startswith("+cv_fold="):
            cv_fold = int(ov.split("=", 1)[1])

    # check to make sure variables are all set
    assert eval_dir is not None, "Must specify file for evaluation!"
    
    cfg = cfg["manager"]

    benchmark = instantiate(
        cfg["benchmark"],
        measures=None,
        rng=rng,
    )

    # open eval file
    with open(eval_dir, "r") as f:
        prompts_to_eval = yaml.safe_load(f)

    # arrange the prompts accordingly
    prompts = []
    for task_id in range(
        benchmark.get_num_tasks() # get the total number of tasks available for this benchmark
    ):
        init_sol = benchmark.get_initial_solution(task_id=task_id)[0]
        if original:
            # `+original=true`: evaluate only the benchmark's own default task
            # description for each task, ignoring the saved eval_dir prompts.
            prompts.append([init_sol["prompt"]])
        else:
            prompt_dict = prompts_to_eval[init_sol["prompt"]]
            prompts_list = [p.lower() for p in prompt_dict.keys()]
            # prompts.append(prompts_list) # eval them all

            if cv_fold is not None:
                # Cross-validation-style fold selection over the saved
                # candidate prompts: splits prompts_list into fixed-size-2
                # chunks and takes only the chunk for this fold, so each
                # fold evaluates on a disjoint slice of prompts "unseen" by
                # the other folds (held out), rather than re-evaluating all
                # prompts every time.
                # prompts_list = prompts_list[: 2*cv_fold] + prompts_list[2*(cv_fold+1) :]
                prompts_list = prompts_list[2*cv_fold:2*(cv_fold+1)] # select unseen prompts for evaluation

            prompts.append(prompts_list)
            
    print("Evaluating:")
    print(prompts)

    # log all results to this file
    all_res = []
    all_res_dict = {}

    # compile prompts to evaluate
    for task_id, prompt_batch in enumerate(prompts):
        print(f"Evaluating task {task_id + 1}/{len(prompts)}: {prompt_batch}")
        init_sol = benchmark.get_initial_solution(task_id=task_id)[0]
        
        prompt_batch = [[p] for p in prompt_batch]  # wrap each prompt in a list to match expected input format
        
        prompt_batch = prompt_batch
        
        results = benchmark.evaluate(
            prompt_batch,
            None, # no llm needed
            classify = False,
            task_id=task_id,
        )
        # print(results)
        all_res.append(results)
        
        all_res_dict[init_sol["prompt"]] = {
            k[0]: {
                "success": v["task_successes"],
                "total": v["n_evals"],
                "gradient_info": v["gradient_info"],
                "all_gradients_original": v["gradient_info_original"],
                "success_list": v["success_list"],
            
            } for k, v in zip(prompt_batch, results)
        }
    
    # store eval
    path = "eval.yaml"
    path = os.path.join(
        logdir,
        path
    )
    
    # dump
    with open(os.path.join(logdir, "eval_dict.pkl"), "wb") as f:
        pickle.dump(all_res_dict, f)
    
    print(f"Saving results to {path}")
    with open(path, "w") as f:
        json.dump(all_res, f, indent=4)

    with open(os.path.join(logdir, "eval_dict.yaml"), "w") as f:
        json.dump(all_res_dict, f)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    """Parses command line flags and sets up and runs experiment.

    Args:
        cfg: Hydra config. See config/config.yaml and the sub-directories for default
            options and docs.
    """

    # Try-except to log any tracebacks to file.
    try:
        # Define OmegaConf resolvers.
        define_resolvers()

        # Set seeds.
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        # Can specify deterministic=True, but there are some issues with it.
        set_seed(cfg["seed"])
        
        # Run the evaluations directly in this script
        evaluate(cfg)
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
