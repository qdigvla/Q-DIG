n="$1"

# default n to 4 repeats
if [ -z "$n" ]; then
    n=4
fi

# NOTE: CUDA_VISIBLE_DEVICES must name a GPU that is actually available/
# allowed on your machine -- adjust before running. (This is only used for
# any local components of the pipeline; GR00T inference itself runs on
# whatever GPU the run_gr00t_server process above was started with.)
CUDA_VISIBLE_DEVICES=0 python -m src.prompt.run_ert \
    --benchmark="LIBERO" \
    --model="groot" \
    --variation="Rephrase" \
    --reload="rephrase_groot" \
    --k=0 \
    --n="$n"
