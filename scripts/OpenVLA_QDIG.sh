k="$1"
n="$2"

# default k to 5 rounds
if [ -z "$k" ]; then
    k=5
fi

# default n to 2 repeats
if [ -z "$n" ]; then
    n=2
fi

# Convert round count k into the manager's `max_evals` budget.
max_evals=$(((k + 1) * 10))

# Tag output directories with today's date so repeated sweeps don't clobber
# each other.
DATE=$(date +%Y-%m-%d)

# LIBERO-Goal has a single task_id (0) in this configuration; sweep over it
# explicitly so this script's structure matches the multi-task GR00T/pi05
# scripts and is easy to extend to more task_ids later.
for task_id in $(seq 0 0); do

    for i in $(seq 1 "$n"); do
        echo "Running iteration $i with max_evals=$max_evals"

        # NOTE: CUDA_VISIBLE_DEVICES must name a GPU that is actually
        # available/allowed on your machine -- adjust before running.
        CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python -m src.prompt.main \
            manager=qd_vlm \
            manager/qd=rainbow_teaming \
            manager/benchmark=LIBERO \
            manager.max_evals=$max_evals \
            manager.benchmark.task_id=$task_id \
            hydra.run.dir="./outputs/qdig_openvla_${DATE}/task${task_id}/k${k}_n${n}_i${i}"

        if [ $? -ne 0 ]; then
            echo "Error occurred during iteration $i. Exiting."
            exit 1
        fi
    done

done
