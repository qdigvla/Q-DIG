import math

import numpy as np
from sentence_transformers import SentenceTransformer

# Shared sentence-embedding model used by similarity() to compare prompts.
# Loaded once at import time since loading it per-call would be expensive.
sbert_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
)


def failure_rate(n_evals: int, successes: int, **kwargs):
    """Fraction of rollouts that failed. Used as an objective to maximize:
    higher failure rate means the prompt is more effective at breaking the
    policy."""
    if n_evals == 0:
        return 1

    return 1 - successes / n_evals


def failure_variance(n_evals: int, successes: int, **kwargs):
    """4x the empirical Bernoulli variance of the success indicator
    (p(1-p) with p = successes/n_evals), scaled by the factor of 4 so the
    result is normalized to [0, 1] (variance peaks at 0.25 when p=0.5).
    This rewards prompts whose outcome is inconsistent across rollouts
    (neither always succeeds nor always fails), as opposed to failure_rate
    which just rewards always failing."""
    if n_evals == 0:
        return 1

    return 4 * (successes * (n_evals - successes)) / (n_evals**2) if n_evals > 1 else 0


def similarity(str1: str, str2: str):
    """Cosine similarity between SBERT embeddings of two strings, in
    [-1, 1] (NaN, e.g. from a zero-norm embedding, is mapped to 0)."""
    v1 = sbert_model.encode(str1, show_progress_bar=False)
    v2 = sbert_model.encode(str2, show_progress_bar=False)

    cossine_similarity = np.matmul(v1, v2.T) / abs(
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

    if math.isnan(cossine_similarity):
        cossine_similarity = 0.0
    return cossine_similarity


def failure_and_similarity(
    n_evals: int, successes: int, new_prompt: str, base_prompt: str
):
    """Objective that rewards prompts that both cause failure AND stay
    semantically close to the original/base prompt (so the rephrase is still
    a faithful version of the task, not an unrelated instruction)."""
    cossine_similarity = similarity(new_prompt, base_prompt)

    return failure_rate(n_evals, successes) * cossine_similarity


def variance_and_similarity(
    n_evals: int, successes: int, new_prompt: str, base_prompt: str
):
    """Objective combining outcome variance (inconsistent success/failure)
    with semantic drift from the base prompt: `1 - cosine_similarity` is
    added so that *less* similar prompts also increase the objective,
    i.e. this rewards both inconsistent behavior and larger rephrasing."""
    cossine_similarity = similarity(new_prompt, base_prompt)

    return failure_variance(n_evals, successes) + 1 - cossine_similarity
