k="$1"
n="$2"

# default k to 2 rounds
if [ -z "$k" ]; then
    k=2
fi

# default n to 1 repeat
if [ -z "$n" ]; then
    n=1
fi

# NOTE: CUDA_VISIBLE_DEVICES must name a GPU that is actually available/
# allowed on your machine -- adjust before running. (This is only used for
# any local components of the pipeline; pi0/pi05 inference itself runs on
# whatever GPU the openpi server above was started with.)
CUDA_VISIBLE_DEVICES=0 python -m src.prompt.run_ert \
    --benchmark="LIBERO" \
    --model="openpi" \
    --reload="ert_openpi" \
    --k="$k" \
    --n="$n"
