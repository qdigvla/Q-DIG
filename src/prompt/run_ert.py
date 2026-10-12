"""
run_ert.py

Standalone (non-Hydra) driver for the ERT-style baselines: ERT itself, and the
"Rephrase" ablation that skips the image-conditioned step. This is the script
invoked by `scripts/paper/pi05_ERT.sh` and `scripts/paper/GR00T_Rephrase.sh`.

Unlike `src.prompt.main` (the QDIG entry point, which is driven entirely by
Hydra configs under `config/`), this script manually loads one benchmark
config file (selected via `--model`) and drives its own generate -> evaluate
-> select-top-k loop across `--k` iterations and `--n` repeats. See the
"Generating Instructions" section of the top-level README for how this fits
into the three example pipelines.
"""

import os
import subprocess
import time

import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from .ERT.generate_instructions import main as generate_instructions
from .ERT.select_topk import main as select_topk

# Repo root, resolved relative to this file (src/prompt/run_ert.py) so the
# benchmark config lookups below work regardless of where the repo is cloned,
# instead of being hardcoded to one machine's path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate instructions for red teaming"
    )
    parser.add_argument("--n", required=True, type=int, help="Seed")
    parser.add_argument("--k", required=True, type=int, help="Index of iterations")
    parser.add_argument(
        "--benchmark",
        required=True,
        type=str,
        help="benchmark to use",
        default="LIBERO",
    )
    parser.add_argument(
        "--variation",
        type=str,
        help="variation of ERT",
        default="ERT", # or rephrase
    )
    parser.add_argument(
        "--reload",
        type=str,
        help="reload dir",
        default="", # or rephrase
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to use",
        default="openvla",
    )
    args = parser.parse_args()

    if args.reload == "":
        formatted_tome = time.strftime("%Y-%m-%d/%H-%M-%S")
    else:
        formatted_tome = args.reload

    output_path = f"outputs/libero_{formatted_tome}/div_vlm/{{i}}/{{k}}.yaml"
    example_path_dir = f"outputs/libero_{formatted_tome}/div_vlm/{{i}}/{{k}}/"
    example_path = f"outputs/libero_{formatted_tome}/div_vlm/{{i}}/{{k}}/top3.yaml"
    results_path = f"outputs/libero_{formatted_tome}/div_vlm/{{i}}/{{k}}/eval.yaml"

    # NOTE: these are fixed for all ERT/Rephrase runs rather than exposed as
    # CLI flags -- task_id starts at 0 (the loop below sweeps every task via
    # `num_tasks`), measures are unused by this (non-QD) baseline, and 10
    # evals/task-description matches the paper's eval budget.
    measures = None  # ["TEST", "TEST"]
    task_id = 0
    n_evals = 10

    if args.benchmark == "LIBERO":
        from src.prompt.envs.libero_manager import LIBEROBenchmark

        # Load the single consolidated LIBERO benchmark config and override
        # model_family (+ that model's remote-server host/port, for the
        # remote families) in code instead of loading a separate YAML per
        # model. See README "Running an experiment" for how to start each
        # server: openpi's serve_policy.py, OpenVLA-OFT's (upstream)
        # vla-scripts/deploy.py, or GR00T's run_gr00t_server.py.
        path = os.path.join(_REPO_ROOT, "config/manager/benchmark/LIBERO.yaml")
        env_cfg = DictConfig(yaml.safe_load(open(path, "r"))).env_cfg

        if args.model == "openpi":
            env_cfg.model_family = "openpi"
            env_cfg.pretrained_checkpoint = "openpi/"
            env_cfg.port = 8000

        elif args.model == "openvla":
            env_cfg.model_family = "openvla"

        elif args.model == "groot":
            # Port 8788 (not GR00T's usual default of 8787) so an ERT/Rephrase
            # run can be pointed at a second GR00T server instance running
            # alongside the one used for QDIG.
            env_cfg.model_family = "gr00t"
            env_cfg.pretrained_checkpoint = "gr00t/"
            env_cfg.port = 8788

        else:
            raise ValueError(f"Unsupported --model for LIBERO benchmark: {args.model}")

        # create the benchmark
        benchmark = LIBEROBenchmark(
            n_evals,
            measures,
            env_cfg=env_cfg,
            task_id=task_id,
        )
        num_tasks = 10
    elif args.benchmark == "SimplerEnv":
        from src.prompt.envs.simpler_env_manager import SimplerEnvBenchmark

        simpler_path = os.path.join(_REPO_ROOT, "config/manager/benchmark/SimplerEnv.yaml")
        cfg = DictConfig(yaml.safe_load(open(simpler_path, "r")))

        n_evals = 10

        benchmark = SimplerEnvBenchmark(
            n_evals,
            measures,
            env_cfg=cfg.env_cfg,
            task_id=task_id,
            env_names=cfg.env_names,
        )
        num_tasks = len(cfg.env_names)

    else:
        raise ValueError("Unsupported benchmark")

    # Gather the benchmark's initial task images/prompts up front, keyed by
    # the original task description. These seed the very first round of
    # instruction generation (k=0) and let us map a generated instruction
    # back to the task_id it was generated for.
    task_images = {}
    original_prompts = []
    for t in range(num_tasks):
        res = benchmark.get_initial_solution(task_id=t)[0]
        task_images[res["prompt"]] = [res["image"]]
        original_prompts.append(res["prompt"])

    # ERT conditions instruction generation on the task's scene image;
    # Rephrase does not (it only rewords the existing instruction), so
    # `use_image` is False for that variation.
    use_image = args.variation == "ERT"

    # Outer loop: `k` ERT self-improvement rounds. Each round's instructions
    # are generated using the previous round's top-3 results as few-shot
    # examples (`examples_path`), so round k depends on round k-1 having
    # finished. Rephrase is normally run with --k=0 (a single, non-iterative
    # pass) since there is nothing to self-improve from.
    for k in range(0, args.k + 1):

        # Inner loop: `n` independent repeats of this round (different RNG
        # seeds / instruction samples), used to average results across runs.
        for i in tqdm(range(args.n)):
            # Step 1: generate a batch of candidate instructions for every
            # task, conditioned on last round's top-3 (if k > 0). Skipped if
            # this round's output file already exists, so an interrupted
            # sweep can be resumed without regenerating finished rounds.
            if not os.path.exists(f"{output_path.format(i=i, k=k)}"):
                generate_instructions(
                    output_path=output_path.format(i=i, k=k),
                    examples_path=(example_path.format(i=i, k=k - 1) if k > 0 else None),
                    task_images=task_images,
                    use_image=use_image,
                    max_num_workers=8,
                )

            else:
                print(f"File exists: {output_path.format(i=i, k=k)}")
                # continue

            # Step 2: evaluate every generated instruction by rolling it out
            # in the benchmark (no LLM-based classifier needed here -- we
            # only care about rollout success/failure counts). Also skipped
            # if already computed, for the same resume-ability reason.
            if not os.path.exists(results_path.format(i=i, k=k)):
                # read output path
                results = yaml.safe_load(open(output_path.format(i=i, k=k), "r"))

                results_dict = {}
                for original_task_description, prompts in results.items():
                    task_id = original_prompts.index(original_task_description)

                    # evaluate the prompts
                    evaluated_prompts = benchmark.evaluate(
                        [[p] for p in prompts],
                        None,
                        classify=False,  # no llm needed
                        task_id=task_id,
                    )

                    # store the results in a dict
                    results_dict[original_task_description] = {}
                    for res_info in evaluated_prompts:
                        prompt = res_info["task_description"]
                        results_dict[original_task_description][prompt] = {
                            "success": res_info["task_successes"],
                            "total": res_info["n_evals"],
                        }

                # make example dir
                os.makedirs(example_path_dir.format(i=i, k=k), exist_ok=True)

                # dump results
                import json

                json.dump(results_dict, open(results_path.format(i=i, k=k), "w"))

            # Step 3: pick the top-3 instructions per task by success rate.
            # These become next round's (`k+1`) few-shot examples.
            select_topk(
                results_path.format(i=i, k=k),  # result_path,
                example_path.format(i=i, k=k),  # output_path
                3,
            )
