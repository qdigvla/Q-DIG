import dataclasses
import itertools
import logging
import os
import pickle
import subprocess as sp
import uuid

import cloudpickle
import hydra
import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.utils.metric_logger import MetricLogger

logger = logging.getLogger(__name__)


def get_gpu_memory():
    # credits to https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


class Manager:
    """Orchestrates a QD experiment: builds/reloads the archive, scheduler,
    LLM, and benchmark env, then runs the ask/evaluate/tell loop until
    max_evals is reached, checkpointing along the way.

    Note: the constructor uses a try/except FileNotFoundError as its control
    flow for reload-vs-fresh-start -- if `reload.pkl` exists in the hydra run
    dir, it is loaded and execution resumes from the saved iteration;
    otherwise we fall into the except block and start a brand-new experiment.
    """

    def __init__(self, seed: int, cfg: DictConfig) -> None:
        self._logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        self._cfg = cfg
        self.max_evals = self._cfg["max_evals"]
        self.debug = self._cfg["debug"]
        self.render = self._cfg["render"]

        try:
            logger.info(f"Attempting to load from {self._logdir}/reload.pkl")
            if self.debug:  # for debugging, we track the memory footprint
                logger.info(f"Memory footprint before file open: {get_gpu_memory()}")

            with open(f"{self._logdir}/reload.pkl", "rb") as f:
                if self.debug:
                    logger.info(
                        f"Memory footprint before pickle load: {get_gpu_memory()}"
                    )

                logger.info(f"Reloading experiment from {self._logdir}")
                data = pickle.load(f)
                self.rng = data["rng"]
                self.itrs_completed = data["itrs_completed"]
                self.evals_used = data["evals_used"]
                self.overall_min_obj = data["overall_min_obj"]
                self.metadata_id = data["metadata_id"]
                self.cur_best_id = data["cur_best_id"]
                self.failed_levels = data["failed_levels"]

                self.measures = data["measures"]

                if self._cfg["wandb"]["enable"]:
                    if "wandb_id" in data:
                        self.wandb_id = data["wandb_id"]
                    else:
                        logger.warning(
                            "No wandb_id stored in the checkpoint. This will lead to "
                            "creating a new wandb run instead of resuming the old one."
                        )
                        self.wandb_id = str(uuid.uuid4())

                # load the model for the scheduler
                self._setup_llm(data["llm_cfg"])

                if self.debug:
                    # print memory footprint before
                    logger.info(
                        f"Memory footprint after pickle load, before model load: {get_gpu_memory()}"
                    )

                # create the model for the scheduler
                if data.get("scheduler"):
                    self.scheduler = data["scheduler"]
                    for emitter in self.scheduler._emitters:
                        # if getattr(emitter, "llm"):
                        setattr(emitter, "llm", self.llm)
                else:
                    self._create_scheduler(seed, cfg)

                self.archive = self.scheduler.archive
                if self.scheduler.result_archive is not self.archive:
                    self.result_archive = self.scheduler.result_archive
                else:
                    self.result_archive = None

                self.metrics = data["metrics"]

                if self.debug:
                    # print memory footprint after
                    logger.info(
                        f"Memory footprint after model load: {get_gpu_memory()}"
                    )

            # Instantiate the env manager
            self.benchmark = instantiate(
                cfg["benchmark"],
                measures=self.measures,
                rng=self.rng,
            )

            logger.info(f"Itrs already completed: {self.itrs_completed}")
            logger.info(
                f"Execution continues from itr {self.itrs_completed + 1} (1-based)"
            )

        except FileNotFoundError:
            logger.info(f"Creating a new experiment at {self._logdir}")
            self.rng = np.random.default_rng(seed=seed)
            self.itrs_completed = 0
            self.evals_used = 0
            self.overall_min_obj = np.inf
            self.metadata_id = 0
            self.cur_best_id = None  # ID of most recent best solution.
            self.failed_levels = []

            # establish measures
            self.measures = OmegaConf.to_container(cfg.qd.measures, resolve=True)

            # Instantiate the env manager
            self.benchmark = instantiate(
                cfg["benchmark"],
                measures=self.measures,
                rng=self.rng,
            )

            # Setup LLM
            self._setup_llm(cfg["llm"])

            # Setup QD components
            self._create_scheduler(seed, cfg)

            # Setup metrics
            self._setup_metrics()

            if self._cfg["wandb"]["enable"]:
                self.wandb_id = str(uuid.uuid4())

        # Setup wandb if required
        self.wandb_logger = None
        if self._cfg["wandb"]["enable"]:
            # runtime_choices = hydra.core.hydra_config.HydraConfig.get().runtime.choices
            logger.warning(f"wandb setup not enabled yet")

    def _create_scheduler(self, seed, cfg):
        """Instantiate archive and emitters and then the schedulers."""
        logger.info("Setting up scheduler for QD")

        # Get initial solutions
        # TODO: get the initial solutions from the config somehow
        # x0 = ["pick up the black bowl between the plate and the ramekin and place it on the plate"]
        # img = ""
        x0 = self.benchmark.get_initial_solution()

        sol_size = 1  # single prompt solution
        logger.info(f"Solution size: {sol_size}")

        # construct the ranges and
        dims = []
        ranges = []

        categories = [m["categories"] for m in self.measures]

        # create the archive
        self.archive = instantiate(
            cfg["qd"]["archive"],
            categories=categories,
            seed=seed,
            extra_fields={
                "metadata": ((), object),
            },
        )

        # create result archive (if specified)
        self.result_archive = None
        if "result_archive" in cfg["qd"] and cfg["qd"]["result_archive"] is not None:
            self.result_archive = instantiate(
                cfg["qd"]["result_archive"],
                dims=dims,
                ranges=ranges,
                seed=seed,
                extra_fields={
                    "metadata": ((), object),
                },
            )
        else:
            self.result_archive = None
        logger.info(f"Archive: {self.archive}")
        logger.info(f"Result archive: {self.result_archive}")

        # Instantiate emitters
        # NOTE: initial solutions are defined in the config file. Specification allows for multiple initial solutions
        # Each emitter has specific config that determines how the initial solutions are used
        emitters = []
        for e in cfg["qd"]["emitters"]:
            emitter_seeds = self.rng.integers(
                np.iinfo(np.int32).max / 2,
                size=e["num"],
                endpoint=True,
            )

            e_kwargs = {
                "llm": self.llm,
                "archive": self.archive,
                "solution_dim": sol_size,
                "measure_cfg": list(self.measures),
                "initial_solutions": x0,
                "batch_size": e["batch_size"],
            }

            emitters.extend(
                [
                    instantiate(
                        e["type"],
                        **e_kwargs,
                        seed=s,
                    )
                    for s in emitter_seeds
                ]
            )
        logger.info(f"Emitters: {emitters}")

        # Create custom scheduler
        self.scheduler = instantiate(
            cfg["qd"]["scheduler"],
            archive=self.archive,
            emitters=emitters,
            result_archive=self.result_archive,
        )
        logger.info(f"Scheduler: {self.scheduler}")

    def _setup_metrics(self):
        """Setup metric list for logging."""
        metric_list = [
            ("Total Evals", True),
            ("Mean Evaluation", False),
            ("Actual QD Score", True),
            ("Archive Size", True),
            ("Archive Coverage", True),
            ("Best Objective", False),
            ("Worst Objective", False),
            ("Mean Objective", False),
            ("Overall Min Objective", False),
        ]

        # Extra logging for result archive
        if self.result_archive is not None:
            res_metric_list = [(f"(res) {k}", v) for k, v in metric_list]
            metric_list += res_metric_list

        self.metrics = MetricLogger(metric_list)

    def _setup_llm(self, model_cfg):
        """Setup the LLM for the experiment."""

        # we create the LLM model here centrally
        self.llm = instantiate(model_cfg)

    def _finished(self) -> bool:
        """Whether execution is done."""
        return self.evals_used >= self._cfg["max_evals"]

    def _msg_all(self, msg: str):
        """Logs msg on master, on all workers, and in dashboard_status.txt."""
        logger.info(msg)
        with open(f"{self._logdir}/dashboard_status.txt", "w") as f:
            f.write(msg)

    def _extract_metadata(self, meta) -> dict:
        """Constructs metadata object from results of an evaluation."""
        # meta = dataclasses.asdict(r)

        # Remove unwanted keys.
        none_keys = [key for key in meta if meta[key] is None]
        for key in itertools.chain(none_keys, []):
            try:
                meta.pop(key)
            except KeyError:
                pass

        meta["metadata_id"] = self.metadata_id
        self.metadata_id += 1

        return meta

    def _evaluate_solutions(self, sols):
        """Evaluates a batch of solutions and adds them to the archive."""
        logger.info("Evaluating solutions")

        # If this batch would exceed the experiment's total eval budget, only
        # evaluate as many as fit; the rest are recorded below as skipped
        # (NaN objective/measures) so array lengths still match `sols`.
        skipped_sols = 0
        if self.evals_used + len(sols) > self._cfg["max_evals"]:
            remaining_evals = self._cfg["max_evals"] - self.evals_used
            remaining_sols = remaining_evals
            skipped_sols = len(sols) - remaining_sols
            sols = sols[:remaining_sols]
            logger.info(
                f"Unable to evaluate all solutions; will evaluate "
                f"{remaining_sols} instead"
            )

        self.evals_used += len(sols)

        if self.debug:
            logger.info(f"evals_used (old): {self.evals_used - len(sols)}")
            logger.info(f"evals_used (new): {self.evals_used}")

        logger.info("Distributing evaluations")
        results = self.benchmark.evaluate(
            sols,
            self.llm,
            self.debug,
            self.render,
        )

        logger.info("Adding solutions to the scheduler")
        objs, measures, metadata = [], [], []
        success_mask = []

        for sol, r in zip(sols, results):
            if not r["failed"]:
                obj = r["objective"]
                meas = r["measures"]
                meta = self._extract_metadata(r)

                objs.append(obj)
                measures.append(meas)
                metadata.append(meta)
                success_mask.append(True)
            else:
                failed_level_info = self.benchmark.add_failed_info(sol, r)
                self.failed_levels.append(failed_level_info)
                objs.append(np.nan)
                measures.append(np.full(self.archive.measure_dim, np.nan))
                metadata.append(None)
                success_mask.append(False)

        # logger.info(f"Skipp {skipped_sols} solutions, {len(sols)} evaluated")
        logger.info(f"Filling in null values for skipped sols: {skipped_sols}")
        for _ in range(skipped_sols):
            objs.append(np.nan)
            measures.append(np.full(self.archive.measure_dim, np.nan))
            metadata.append(None)
            success_mask.append(False)

        # Tell results to scheduler.
        self.scheduler.tell(
            objs,
            measures,
            success_mask=success_mask,
            metadata=metadata,
        )

        self.metrics.add("Mean Evaluation", np.nanmean(objs), logger)
        self.overall_min_obj = min(self.overall_min_obj, np.nanmin(objs))

    def _add_performance_metrics(self, archive, prefix="", commit=False):
        """Calculates various performance metrics at the end of each iter."""
        objs = archive.data("objective")
        stats = archive.stats

        metrics_dict = {
            "Total Evals": self.evals_used,
            "Actual QD Score": self.benchmark.actual_qd_score(objs),
            "Archive Size": stats.num_elites,
            "Archive Coverage": stats.coverage,
            "Best Objective": np.max(objs),
            "Worst Objective": np.min(objs),
            "Mean Objective": np.mean(objs),
            "Overall Min Objective": self.overall_min_obj,
        }

        for k, v in metrics_dict.items():
            self.metrics.add(f"{prefix}{k}", v, logger)

        if self.wandb_logger is not None:
            # Plot archive heatmap
            if len(self.measures) != 2:
                logger.warning(
                    "Number of measures is not 2. Skipping archive heatmap from wandb logs."
                )
            else:
                import matplotlib.pyplot as plt
                from ribs.visualize import grid_archive_heatmap

                fig, ax = plt.subplots(1, 1)
                grid_archive_heatmap(
                    archive,
                    ax=ax,
                    vmin=self.benchmark.module.MIN_SCORE,
                    vmax=self.benchmark.module.MAX_SCORE,
                    cmap="viridis",
                    rasterized=True,
                )
                fig.suptitle("Archive heatmap")
                ax.set_xlabel(self.measures[0]["name"])
                ax.set_ylabel(self.measures[1]["name"])
                im = wandb.Image(fig)
                metrics_dict["Archive heatmap"] = im
                plt.close(fig)

            # Log to wandb
            wandb_prefix = "Train Archive" if prefix == "" else "Result Archive"
            wandb.log(
                {f"{wandb_prefix}/{k}": v for k, v in metrics_dict.items()},
                commit=commit,
                step=self.itrs_completed,
            )

    def _plot_metrics(self):
        """Plots metrics every plot_metrics_freq itrs or on final itr."""
        if (
            self.itrs_completed % self._cfg["plot_metrics_freq"] == 0
            or self._finished()
        ):
            logger.info(f"Metrics:\n{self.metrics.get_plot_text()}")

    def _save_archive_history(self):
        """Saves the archive's history.

        We are okay with a pickle file here because there are only numpy arrays
        and Python objects, both of which are stable.
        """
        if self.result_archive is not None:
            with open(f"{self._logdir}/result_archive_history.pkl", "wb") as f:
                pickle.dump(self.result_archive.history(), f)
        with open(f"{self._logdir}/archive_history.pkl", "wb") as f:
            pickle.dump(self.archive.history(), f)

    def _save_archive(self):
        """Saves dataframes of the archive.

        The archive, including solutions and metadata, is saved to
        logdir/archive/archive_{itr}.pkl

        Note that the archive is saved as an dict storing common
        Python objects, so it should be stable (at least, given fixed software
        versions).
        """
        os.makedirs(f"{self._logdir}/archive", exist_ok=True)
        itr = self.itrs_completed
        if self.result_archive is not None:
            data = self.result_archive.data()
            with open(f"{self._logdir}/archive/result_archive_{itr}.pkl", "wb") as f:
                pickle.dump(data, f)

        data = self.archive.data()
        with open(f"{self._logdir}/archive/archive_{itr}.pkl", "wb") as f:
            pickle.dump(data, f)

        # save it as a final archive final as well (for simplicity)
        with open(f"{self._logdir}/archive/archive.pkl", "wb") as f:
            pickle.dump(data, f)

    def _save_reload_data(self):
        """Saves data necessary for a reload.

        Current reload files:
        - reload.pkl

        Since saving may fail due to memory issues, data is first placed in
        reload-tmp.pkl. reload-tmp.pkl then overwrites reload.pkl.
        """

        # for QD-LLM, the scheduler contains contents which cannot be pickled
        # from copy import deepcopy
        # scheduler = deepcopy(self.scheduler)
        # for i, emitter in enumerate(scheduler._emitters):
        #     clone = deepcopy(emitter)
        #     if getattr(clone, "llm"):
        #         clone.llm = None
        #     scheduler._emitters[i] = clone

        logger.info("Saving reload data")
        logger.info("Saving reload-tmp.pkl")
        with open(f"{self._logdir}/reload-tmp.pkl", "wb") as file:
            reload_data = {
                "rng": self.rng,
                "itrs_completed": self.itrs_completed,
                "evals_used": self.evals_used,
                "overall_min_obj": self.overall_min_obj,
                "metadata_id": self.metadata_id,
                "cur_best_id": self.cur_best_id,
                "failed_levels": self.failed_levels,
                # "scheduler": scheduler,
                "metrics": self.metrics,
                "llm_cfg": self._cfg["llm"],
                "measures": self.measures,
            }

            if self._cfg["wandb"]["enable"]:
                reload_data["wandb_id"] = self.wandb_id

            cloudpickle.dump(reload_data, file)

        logger.info("Renaming tmp reload files")
        os.rename(f"{self._logdir}/reload-tmp.pkl", f"{self._logdir}/reload.pkl")

        logger.info("Finished saving reload data")

    def _save_data(self):
        """Saves archive, reload data, history, and metrics if necessary.

        This method must be called at the _end_ of each itr. Otherwise, some things
        might not be complete. For instance, the metrics may be in the middle of an
        iteration, so when we reload, we get an error because we did not end the
        iteration.
        """
        if self._cfg["archive_save_freq"] is None:
            save_full_archive = False
        elif self._cfg["archive_save_freq"] == -1 and self._finished():
            save_full_archive = True
        elif (
            self._cfg["archive_save_freq"] > 0
            and self.itrs_completed % self._cfg["archive_save_freq"] == 0
        ):
            save_full_archive = True
        else:
            save_full_archive = False

        logger.info("Saving metrics")
        self.metrics.to_json(f"{self._logdir}/metrics.json")

        logger.info("Saving archive history")
        self._save_archive_history()

        if save_full_archive:
            logger.info("Saving full archive")
            self._save_archive()
        if (
            self.itrs_completed % self._cfg["reload_save_freq"] == 0
        ) or self._finished():
            self._save_reload_data()
        if self._finished():
            logger.info("Saving failed levels")
            with open(f"{self._logdir}/failed_levels.pkl", "wb") as f:
                pickle.dump(self.failed_levels, f)

    def run_experiment(self):
        """Runs an experiment."""
        while not self._finished():
            self._msg_all(
                f"----- Itr {self.itrs_completed + 1} "
                f"({self.evals_used} evals) -----"
            )
            self.metrics.start_itr()
            self.archive.new_history_gen()
            if self.result_archive is not None:
                self.result_archive.new_history_gen()

            logger.info("Running QD ask/tell")
            sols = self.scheduler.ask()

            # Normalize each solution's task description before evaluation:
            # strip periods and lowercase, so minor punctuation/casing
            # differences in emitted prompts don't count as distinct
            # solutions or affect downstream string-based comparisons
            # (e.g. similarity()/classification against the original prompt).
            for i in range(len(sols)):
                description = sols[i][0]

                # quickly preprocess the task description
                description = description.replace(".", "")
                description = description.lower()

                sols[i][0] = description
            
            self._evaluate_solutions(sols)

            logger.info("Itr complete - now logging and saving data")
            self.itrs_completed += 1

            # Metrics + optional wandb logging
            if self.result_archive is not None:
                self._add_performance_metrics(self.result_archive, "(res) ")
            self._add_performance_metrics(self.archive, commit=True)
            self.metrics.end_itr()
            self._plot_metrics()
            self._save_data()  # Keep at end of loop (see method docstring).

        self._msg_all(
            f"----- Done! {self.itrs_completed} itrs, " f"{self.evals_used} evals -----"
        )

        if self.wandb_logger is not None:
            self.wandb_logger.finish()
