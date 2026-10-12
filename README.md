# Q-DIG: Quality Diversity for Diverse Instruction Generation

[Paper (arXiv)](https://arxiv.org/abs/2603.12510) | [Project website](http://qdigvla.github.io/)

## Installation

Follow the setup instructions below to get started (adapted from the
[OpenVLA](https://github.com/openvla/openvla?tab=readme-ov-file) instructions)

```bash
# Create and activate conda environment
conda create -n qd-safety python=3.10 -y
conda activate qd-safety

# Clone and install this repo. Recurse to download all submodules
git clone https://github.com/qdigvla/Q-DIG.git --recurse-submodules
cd Q-DIG
pip install -r requirements.txt

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

Also install [pre-commit](https://pre-commit.com), which will format all your
files every time you commit:

```bash
pip install pre-commit
pre-commit install
```

Note that we use [black](https://github.com/psf/black) and
[isort](https://pycqa.github.io/isort/) for formatting Python, and
[prettier](https://prettier.io) for formatting Markdown and YAML.

## Instructions

### Prompt QD

Run the prompt experiments with

```bash
python -m src.prompt.main
```

The config is in `config/config.yaml`. Refer to the manager in
`config/manager/default.yaml` for details on modifying the experiment.

### Running an experiment

Experiments are launched through `src.prompt.main`, which uses
[Hydra](https://hydra.cc) to compose a config from `config/config.yaml` and the
`config/manager/` subtrees. Rather than editing YAML files, you normally
override the pieces you want on the command line, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python -m src.prompt.main \
    manager=qd_vlm \
    manager/qd=rainbow_teaming \
    manager/benchmark=LIBERO \
    manager.max_evals=110 \
    manager.benchmark.task_id=5 \
    hydra.run.dir=./outputs/my_run
```

- `manager=<name>` selects a top-level manager config from `config/manager/`
  (e.g. `qd_vlm.yaml`).
- `manager/qd=<name>` selects the QD algorithm config from
  `config/manager/qd/` (currently just `rainbow_teaming`).
- `manager/benchmark=<name>` selects **which model/environment to evaluate
  against**, from `config/manager/benchmark/`. This is the key switch for
  changing models -- see the next section.
- Any other dotted key (`manager.max_evals`, `manager.benchmark.task_id`, ...)
  overrides a single field in the composed config without needing a new YAML
  file.
- `hydra.run.dir=...` controls where logs/outputs for the run are written.

See `scripts/OpenVLA_QDIG.sh` for a complete, runnable example that wraps
this command in a loop over task IDs and repeats:

```bash
./scripts/OpenVLA_QDIG.sh <k> <n>
```

`k` sets `manager.max_evals` (scaled internally to `(k + 1) * 10`) and `n` is
the number of repeated iterations per task. It calls `python -m
src.prompt.main` with `manager/benchmark=LIBERO` (OpenVLA-OFT loaded
in-process, the config's default) and writes each run to
`./outputs/qdig_openvla_<date>/task<id>/k<k>_n<n>_i<i>`. Copy this script as a
template for other models/benchmarks by swapping the override flags (see
below) and the output directory name. Note the script pins
`CUDA_VISIBLE_DEVICES=0`; adjust this to a GPU that is actually
available/allowed on your machine before running it. There is no built-in parallelization so Q-DIG's GPU allocation is quite minimal.

### Switching models via `manager/benchmark`

There are exactly two benchmark configs: `config/manager/benchmark/LIBERO.yaml`
and `.../SimplerEnv.yaml`, selected with `manager/benchmark=LIBERO` or
`manager/benchmark=SimplerEnv`. Each is a single file covering every
supported model for that benchmark -- there is no separate file per model.
Each defaults to OpenVLA(-OFT) loaded in-process (no server needed); to use a
different model, override `env_cfg.model_family` (and that model's specific
fields) from the command line, rather than adding another YAML file. See each
YAML's top comment for the full list of overridable fields.

`env_cfg.model_family` picks the model implementation: `openvla` and
`diffusion` load in-process; `openvla-server`, `openpi`, and `gr00t` (3 models tested in our paper) are
always remote, querying an already-running model server over
`env_cfg.host`/`env_cfg.port` (for SimplerEnv, these go under `model_kwargs`
rather than `env_cfg`).

Before running with a remote `model_family`, start the corresponding model
server:

```bash
# OpenVLA-OFT -- run from inside the openvla-oft submodule, since deploy.py
# imports from its own experiments/robot package (not installed as a module)
cd openvla-oft
python vla-scripts/deploy.py --pretrained_checkpoint <ckpt> \
    --unnorm_key <task_suite> --host 0.0.0.0 --port 8777
cd ..

# pi0 / pi05 (openpi)
python -m openpi.scripts.serve_policy --port 8000 ...

# Isaac-GR00T
python -m gr00t.eval.run_gr00t_server --host 0.0.0.0 --port 8787 ...
```

See `scripts/` for examples on running evals with different models, simulators, and baselines!


**Using VLAs with SimplerEnv:** the `SimplerEnv-OpenVLA` submodule (downloaded
via `--recurse-submodules`, see Installation) already includes an in-process
OpenVLA policy (`model_family=openvla`, SimplerEnv.yaml's default), so no
extra setup is needed to run OpenVLA against SimplerEnv. To use a different
VLA with SimplerEnv instead of adding another in-process policy
implementation, host that model behind its own inference server (as above)
and set `model_family=openvla-server` (or `openpi`/`gr00t`) with the matching
`model_kwargs` as shown above.

### Generating Instructions

Adversarial/diverse task instructions are generated by one of two pipelines,
each with a runnable example script under `scripts/`. See the main paper for details on how these algorithms vary.

```bash
# Q-DIG against OpenVLA-OFT on LIBERO-Goal
./scripts/OpenVLA_QDIG.sh 5 2

# ERT against pi0/pi05 on LIBERO-Goal (requires an openpi server running)
./scripts/pi05_ERT.sh 2 1

# Rephrase ablation against Isaac-GR00T on LIBERO-Goal (requires a GR00T server running)
./scripts/GR00T_Rephrase.sh 4
```

### Data Augmentation and Finetuning

We do not include code for data augmentation / fine-tuning in this repo: for
any given base model (OpenVLA-OFT, pi0/pi05, Isaac-GR00T, ...), this is a
minor addition to that model's own existing supervised fine-tuning (SFT)
code, not a separate pipeline.

The change is localized to the instruction-encoding step of SFT. Where the
baseline fine-tuning script encodes a single, static instruction per task
(e.g. `"pick up the red block"`), instead provide the list of adversarial/
diverse instructions generated by Q-DIG (or another adversarial instruction
generation method, e.g. ERT) for that task, and sample one instruction from
the list each time the example is encoded (e.g. once per epoch, or once per
batch draw) rather than always encoding the same fixed string. Everything
else in the base model's SFT pipeline -- tokenization, action/proprio
encoding, loss, optimizer -- is unchanged.

  
## Troubleshooting

- Sometimes `flash-attn` takes a while to build. Ensure you have ninja
  downloaded as shown in the instructions. You can also clear the pip cache for
  this library by running `pip cache remove flash_attn`, or increase the number
  of workers:
  `MAX_JOBS=64 python -m pip -v install flash-attn==2.5.5 --no-build-isolation`

## Setup Kinova with SimplerEnv

Obtain Kinova Description from: https://github.com/Kinovarobotics/kinova-ros/tree/noetic-devel/kinova_description

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{srikanth2026redteamingvisionlanguageactionmodelsquality,
      title={Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies},
      author={Siddharth Srikanth and Freddie Liang and Ya-Chuan Hsu and Varun Bhatt and Shihan Zhao and Henry Chen and Bryon Tjanaka and Minjune Hwang and Akanksha Saran and Daniel Seita and Aaquib Tabrez and Stefanos Nikolaidis},
      year={2026},
      eprint={2603.12510},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.12510},
}
```