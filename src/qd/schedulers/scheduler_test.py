"""Tests extra features of the scheduler."""

import pytest
from ribs.emitters import GaussianEmitter

from ...qd.archives import GridArchive

# from .scheduler import Scheduler
from ribs.schedulers import Scheduler
from .llm_scheduler import LLMScheduler

# pylint: disable = redefined-outer-name

from src.qd.emitters import CategoricalEmitter


@pytest.fixture(params=["batch", "single"])
def scheduler(request):
    add_mode = request.param

    solution_dim = 2
    num_solutions = 2
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        record_history=False,
    )
    result_archive = GridArchive(
        solution_dim=solution_dim,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        record_history=False,
    )
    emitters = [
        GaussianEmitter(archive, sigma=0.1, x0=[0.0, 0.0], batch_size=num_solutions),
        GaussianEmitter(archive, sigma=0.1, x0=[0.0, 0.0], batch_size=num_solutions),
    ]
    return Scheduler(
        archive,
        emitters,
        add_mode=add_mode,
        result_archive=result_archive,
    )


def test_no_success_mask(scheduler):
    _ = scheduler.ask()
    scheduler.tell(
        [1, 1, 1, 1],
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],
    )
    assert len(scheduler.archive) == 4
    assert len(scheduler.result_archive) == 4


def test_tell_success_mask(scheduler):
    _ = scheduler.ask()
    scheduler.tell(
        [1, 0.0, 0.0, 1],
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],
        success_mask=[True, False, False, True],
    )
    assert len(scheduler.archive) == 2
    assert len(scheduler.result_archive) == 2


from ribs.archives import CategoricalArchive

def test_categorical_archive():
    batch_size = 4
    archive = CategoricalArchive(
        solution_dim=1,
        categories=[
            ["A", "B", "C"],
            ["One", "Two", "Three", "Four"],
        ],
        dtype={
            "solution": np.object_,
            "objective": np.float32,
            "measures": np.object_,
        },
    )
    measure_cfg = [
        {"categories": ["A", "B", "C"]},
        {"categories": ["One", "Two", "Three", "Four"]}
    ]

    emitters = [
        CategoricalEmitter(archive, solution_dim=1, initial_solutions=["hey", "hey"], measure_cfg=measure_cfg, batch_size=batch_size)
    ]
    scheduler = Scheduler(archive=archive, emitters=emitters, add_mode="batch")

    measures_batch = [["A", "Four"], ["B", "Three"], ["C", "One"], ["C", "Two"]]
    asked = scheduler.ask()  # Ignore the actual values of the solutions.
    scheduler.tell(np.ones(batch_size), measures_batch)

import numpy as np
if __name__ == "__main__":
    test_categorical_archive()