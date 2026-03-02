import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import agentdojo.agent_pipeline as pipeline
import agentdojo.attacks
import agentdojo.base_tasks
import agentdojo.benchmark
import agentdojo.logging
import agentdojo.task_suite
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatMessage
from pydantic import BaseModel, ConfigDict, model_validator

from lamb import agent, logging


# This is only used for serializing/deserializing runs
class TaskResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    suite_name: str
    user_task_id: str | None
    injection_task_id: str | None
    attack_type: str | None
    injections: dict[str, str] | None
    messages: list[ChatMessage]
    error: str | None
    utility: bool
    security: bool
    duration: float

    @model_validator(mode="after")
    def check_messages(self) -> Self:
        if len(self.messages) < 2:
            raise ValueError("TaskResults must have at least two messages.")
        return self


@dataclass
class TaskScores:
    utility: bool
    security: bool


class SuiteResults:
    user_runs: dict[str, TaskScores]
    injection_runs: dict[str, TaskScores]
    attack_runs: dict[tuple[str, str], TaskScores]
    utility_user_tasks: float = 0.0
    utility_injection_tasks: float = 0.0
    utility_with_attack: float = 0.0
    security: float = 0.0
    count: int = 0

    def __init__(
        self,
        user_runs: dict[str, TaskScores],
        injection_runs: dict[str, TaskScores],
        attack_runs: dict[tuple[str, str], TaskScores],
    ) -> None:
        self.user_runs = user_runs
        self.injection_runs = injection_runs
        self.attack_runs = attack_runs

        self.utility_user_tasks = self.aggregate_results(
            self.get_all_utility(lambda suite_results: suite_results.user_runs)
        )
        self.utility_injection_tasks = self.aggregate_results(
            self.get_all_utility(lambda suite_results: suite_results.injection_runs)
        )
        self.utility_with_attack = self.aggregate_results(
            self.get_all_utility(lambda suite_results: suite_results.attack_runs)
        )
        self.security = self.aggregate_results(self.get_all_security())
        self.count = (
            len(self.user_runs) + len(self.injection_runs) + len(self.attack_runs)
        )

    def get_all_utility(
        self, runs: Callable[["SuiteResults"], dict[Any, TaskScores]]
    ) -> dict[Any, bool]:
        return {name: task_results.utility for name, task_results in runs(self).items()}

    def get_all_security(self) -> dict[Any, bool]:
        return {
            name: task_results.security
            for name, task_results in self.attack_runs.items()
        }

    @staticmethod
    def aggregate_results(results: dict[Any, bool]) -> float:
        all_results = list(results.values())
        return sum(all_results) / len(all_results) if all_results else 0.0


@dataclass
class BenchmarkResults:
    suite_results: dict[str, SuiteResults]


def run_task(
    suite: agentdojo.task_suite.TaskSuite,
    agent_pipeline: pipeline.BasePipelineElement,
    user_task: agentdojo.base_tasks.BaseUserTask | None,
    injection_task: agentdojo.base_tasks.BaseInjectionTask | None,
    attack: agentdojo.attacks.BaseAttack | None,
    logdir: Path | None,
    force_rerun: bool,
    benchmark_version: str
    | None = None,  # TODO: remove this through using own TraceLogger
) -> TaskScores:
    """Run a task. If both a user task and an injection task are specified, as
    well as an attack, the user task is carried out with the injection task
    injected in the environment through the attack.
    If only the user or injection task is specified, that single task is run.
    This is useful to benchmark the utility of a pipeline.

    Args:
        suite: The suite to use to run the benchmark.
        agent_pipeline: The agent pipeline to benchmark.
        user_task: The user task to run. This can be None, in this case the
            injection_task is expected to be specified.
        injection_task: The injection task to run. This can be run separately
            or with a user_task and attack selected.
        attack: The attack to run.
        logdir: The directory to save logs to.
        force_rerun: Whether to force rerun the task in case in the logdir
            there already exists a log for the task.
        benchmark_version: The benchmark version that gets logged.

    Returns:
        A tuple with the utility and security of the task. In case of running a
        single task without an attack, the security is always True.

    """

    task = user_task
    if task is None:
        task = injection_task

    assert task is not None, "Either user_task or injection_task should be set"

    with_attack = False
    injection_task_id = "none"
    injections = {}
    attack_name = "none"
    injection_task_to_run = None

    if user_task and injection_task:
        assert attack is not None, (
            "Attack must be set for injection task to be injected in the environment"
        )
        with_attack = True
        attack_name = attack.name
        injection_task_id = injection_task.ID
        injection_task_to_run = suite.get_injection_task_by_id(injection_task_id)
        injections = attack.attack(user_task, injection_task_to_run)

    if logdir is not None and agent_pipeline.name is not None:
        try:
            task_result = load_task_results(
                suite_name=suite.name,
                user_task=task.ID,
                attack_name=attack_name,
                injection_task=injection_task_id,
                logdir=logdir,
            )
        except (agentdojo.benchmark.ValidationError, FileNotFoundError):
            task_result = None
    else:
        task_result = None

    if task_result is not None and not force_rerun:
        if with_attack:
            logging.info(
                "Skipping task"
                " '{user_task.ID}' with '{injection_task.ID}'"  # type: ignore
                " because it was already run."
            )
        else:
            logging.info(
                f"Skipping task '{task.ID}' with no injections because it was"
                " already run."
            )
        return TaskScores(task_result.utility, task_result.security)

    with agentdojo.logging.TraceLogger(
        delegate=agentdojo.logging.Logger.get(),
        suite_name=suite.name,
        user_task_id=task.ID,
        injection_task_id=injection_task_id,
        injections=injections,
        attack_type=attack_name,
        pipeline_name=agent_pipeline.name,
        benchmark_version=benchmark_version,
    ) as logger:
        try:
            utility, security = suite.run_task_with_pipeline(
                agent_pipeline, task, injection_task=None, injections=injections
            )
        except agentdojo.benchmark.BadRequestError as e:
            if (
                e.code == "context_length_exceeded"
                or e.param == "max_tokens"
                or "reduce the length of the messages." in repr(e)
            ):
                logger.log_error(
                    f"Skipping task {task.ID} due to context_length_exceeded: {e}"
                )
                utility = False
                security = True
            else:
                raise
        except agentdojo.benchmark.UnprocessableEntityError as e:
            if "max_new_tokens" in str(e):
                logger.log_error(
                    f"Skipping task {task.ID} due to context_length_exceeded: {e}"
                )
                utility = False
                security = True
            else:
                raise
        except agentdojo.benchmark.ApiError as e:
            if "internal server error" in str(e):
                logger.log_error(
                    f"Skipping task {task.ID} because of internal server error: {e}"
                )
                utility = False
                security = True
            else:
                raise
        except agentdojo.benchmark.ServerError as e:
            logger.log_error(
                f"Skipping task {task.ID} because of internal server error: {e}"
            )
            utility = False
            security = True
        logger.set_contextarg("utility", utility)
        logger.set_contextarg("security", security)

    return TaskScores(utility, security)


def load_task_results(
    suite_name: str,
    user_task: str,
    attack_name: str,
    injection_task: str,
    logdir: Path,
) -> TaskResults:
    path = logdir / suite_name / user_task / attack_name / f"{injection_task}.json"
    with path.open() as f:
        res_dict = json.load(f)

    return TaskResults(**res_dict)


def load_suite_results(
    suite_name: str,
    attack_name: str,
    logdir: Path = Path("../.log"),
    benchmark_version: str = "v1.2",
) -> tuple[dict[tuple[str, str], TaskResults], dict[tuple[str, str], TaskResults]]:
    suite = get_suite(benchmark_version, suite_name)
    results_without_injections = {}
    results_with_injections = {}

    for user_task_id in suite.user_tasks:
        results_without_injections[(user_task_id, None)] = load_task_results(
            suite_name, user_task_id, "none", "none", logdir
        )
        if "dos" in attack_name:
            injection_task_id = next(iter(suite.injection_tasks.keys()))
            results_with_injections[(user_task_id, injection_task_id)] = (
                load_task_results(
                    suite_name,
                    user_task_id,
                    attack_name,
                    injection_task_id,
                    logdir,
                )
            )
        elif attack_name != "none":
            for injection_task_id in suite.injection_tasks:
                results_with_injections[(user_task_id, injection_task_id)] = (
                    load_task_results(
                        suite_name,
                        user_task_id,
                        attack_name,
                        injection_task_id,
                        logdir,
                    )
                )
    return results_without_injections, results_with_injections


def _get_git_revision() -> str | None:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])  # noqa: S607
            .decode("ascii")
            .strip()
        )
        diff = (
            subprocess.check_output(["git", "diff", "--stat"]).decode("ascii").strip()  # noqa: S607
        )
        if diff:
            print("Warning: Uncommitted changes detected in the repository.")
    except subprocess.CalledProcessError:
        return None
    else:
        return commit


# TODO: benchmark level JSON file should only be created after
def init_benchmark_dir(log_dir: Path) -> Path:
    timestamp_folder = datetime.now(tz=UTC).strftime("%d%H%M%S")
    timestamp_log = datetime.now().astimezone().isoformat()
    bench_dir = log_dir / f"bench-{timestamp_folder}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    meta = {"commit": _get_git_revision(), "timestamp": timestamp_log}

    with Path.open(bench_dir / "bench.json", "w") as f:
        json.dump(meta, f, indent=4)

    return bench_dir


def benchmark_suite(
    agent_pipeline: pipeline.BasePipelineElement,
    suite: agentdojo.task_suite.TaskSuite,
    logdir: Path | None,
    attack: agentdojo.attacks.BaseAttack | None = None,
    user_tasks: Sequence[str] | None = None,
    injection_tasks: Sequence[str] | None = None,
    benchmark_version: str | None = None,
    n_repeats: int = 1,
    force_rerun: bool = False,
) -> SuiteResults:
    """Benchmark a suite.

    Args:
        agent_pipeline: The agent pipeline to use.
        suite: The suite to benchmark.
        logdir: The directory to save logs to.
        attack: The attack to use in the injection tests.
        user_tasks: A subset of user tasks to run. If None, all user tasks are
            run.
        injection_tasks: A subset of injection tasks to run. If None, all
            injection tasks are run.
        benchmark_version: The benchmarking version for the given suite.
        n_repeats: Sets how many times a given user task is run.
        force_rerun: Sets if earlier runs should be recovered from the log
            files or ignored and run from the start.

    """
    user_runs = {}
    injection_runs = {}
    attack_runs = {}

    if user_tasks is not None:
        user_tasks_to_run = {
            user_task_id: suite.get_user_task_by_id(user_task_id)
            for user_task_id in user_tasks
        }
    else:
        user_tasks_to_run = suite.user_tasks

    if injection_tasks is not None:
        injection_tasks_to_run = {
            injection_task_id: suite.get_injection_task_by_id(injection_task_id)
            for injection_task_id in injection_tasks
        }
    else:
        injection_tasks_to_run = suite.injection_tasks

    for user_task_id, user_task in user_tasks_to_run.items():
        task_scores = run_task(
            suite=suite,
            agent_pipeline=agent_pipeline,
            user_task=user_task,
            injection_task=None,
            attack=None,
            logdir=logdir,
            force_rerun=force_rerun,
            benchmark_version=benchmark_version,
        )
        user_runs[user_task_id] = task_scores

    for injection_task_id, injection_task in injection_tasks_to_run.items():
        task_scores = run_task(
            suite=suite,
            agent_pipeline=agent_pipeline,
            user_task=None,
            injection_task=injection_task,
            attack=None,
            logdir=logdir,
            force_rerun=force_rerun,
            benchmark_version=benchmark_version,
        )
        injection_runs[injection_task_id] = task_scores

    for user_task_id, user_task in user_tasks_to_run.items():
        for injection_task_id, injection_task in injection_tasks_to_run.items():
            task_scores = run_task(
                suite=suite,
                agent_pipeline=agent_pipeline,
                user_task=user_task,
                injection_task=injection_task,
                attack=attack,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
            attack_runs[(user_task_id, injection_task_id)] = task_scores

    return SuiteResults(
        user_runs=user_runs, injection_runs=injection_runs, attack_runs=attack_runs
    )


def benchmark(
    agent_loop: agent.ADAgentLoop,
    log_dir: Path = Path(".log"),
    attack: str | None = None,
    user_tasks: list[str] | None = None,
    injection_tasks: list[str] | None = None,
    suite_version: str = "v1.2",
    n_repeats: int = 1,
    force_rerun: bool = False,
) -> BenchmarkResults:
    bench_dir = init_benchmark_dir(log_dir)
    suites = agentdojo.task_suite.get_suites(suite_version)
    suite_results = {}
    for suite_name, suite in suites.items():
        # if suite_name != "slack":
        #     continue
        logging.debug(f"Executing suite {suite_name}")
        agent_pipeline = pipeline.AgentPipeline([agent_loop])
        # This is an inconsistency in AgentDojo.
        # The benchmark requires the attribute `name` to be set,
        # but the `AgentPipeline` class doesn't have it.
        agent_pipeline.__setattr__("name", "local_lamb")
        base_attack = None
        if attack is not None:
            base_attack = agentdojo.attacks.load_attack(attack, suite, agent_pipeline)
        results: SuiteResults = benchmark_suite(
            agent_pipeline=agent_pipeline,
            suite=suite,
            logdir=bench_dir,
            attack=base_attack,
            user_tasks=user_tasks,
            injection_tasks=injection_tasks,
            benchmark_version=suite_version,
            n_repeats=n_repeats,
            force_rerun=force_rerun,
        )
        if base_attack is not None:
            print(f"Suite: {suite_name}\tUtility score: {results.utility_with_attack}")
            print(f"\t\t\tSecurity score: {results.security}")
        else:
            print(f"Suite: {suite_name}\tUtility score: {results.utility_user_tasks}")
        suite_results[suite_name] = results
    return BenchmarkResults(suite_results=suite_results)
