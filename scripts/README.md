# Scripts

Exactly three example scripts, one per paper-table pipeline, each a thin
wrapper around the Python entry points documented in the top-level README's
"Running an experiment" / "Generating Instructions" sections:

- `OpenVLA_QDIG.sh` -- Q-DIG (quality-diversity adversarial instruction
  search) against OpenVLA-OFT, via the Hydra entry point `src.prompt.main`.
- `pi05_ERT.sh` -- the ERT baseline against pi0/pi05 (served remotely via
  openpi), via the standalone driver `src.prompt.run_ert`.
- `GR00T_Rephrase.sh` -- the "Rephrase" ERT ablation against Isaac-GR00T
  (served remotely), also via `src.prompt.run_ert`.

Each script documents its own prerequisites (e.g. starting a remote model
server first) and CLI arguments in a header comment. Use them as templates:
copy one and swap the `manager/benchmark=...` config (QDIG script) or
`--model=...`/`--variation=...` flags (ERT-style scripts) to point at a
different model or ablation.

To run OpenVLA-OFT remotely instead of in-process, start the official server
from the upstream OpenVLA-OFT repo (the `openvla-oft` submodule,
`vla-scripts/deploy.py`) -- see the README.
