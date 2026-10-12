import json
from typing import List

import yaml
from tqdm import tqdm


def main(
    result_path: str,
    output_path: str,
    k: int,
):
    """
    Assume the `result_path` contains a JSON like:
    {
        task: {
            annotation: {
                "success": ...,
                "total": ...,
            }
        }
    }

    Output the YAML file like:
    task1:
    - annotation1
    - annotation2
    task2:
    ...
    ...
    """

    output = {}
    with open(result_path, "r") as f:
        results = json.load(f)
        for task, task_info in tqdm(results.items()):
            # [(annotation, {"success": ..., "total": ....}), ...]
            # Sort ascending by success count, so taking the first k (below)
            # keeps the k annotations/instructions the policy succeeded at
            # *least* often -- i.e. the most effective red-team instructions
            # for eliciting failure on this task.
            sorted_task_infos = sorted(
                list(task_info.items()), key=lambda item: item[1]["success"]
            )
            # [annotation_1, annotation_2, ..., annotation_k]
            topK = list(map(lambda x: x[0], sorted_task_infos[:k]))
            output[task] = topK

    with open(output_path, "w") as f:
        yaml.dump(output, f)
