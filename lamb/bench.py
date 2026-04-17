import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from json import dump, load
from pathlib import Path
from typing import Self

import agentdojo.agent_pipeline as pipeline
import agentdojo.attacks
import agentdojo.base_tasks
import agentdojo.benchmark
import agentdojo.task_suite
from pydantic import BaseModel, ConfigDict, model_validator

from lamb import agent, logging
from lamb.types import ChatMessage

# TODO: timestamp individual test runs


class TaskResults(BaseModel):
    model_config = ConfigDict(extra="allow")
    user_task_id: str | None
    injection_task_id: str | None
    attack_type: str | None
    injections: dict[str, str] | None
    messages: list[ChatMessage]
    utility: bool
    security: bool
    duration: float
    error: str | None = None

    @model_validator(mode="after")
    def check_messages(self) -> Self:
        if self.error is None and len(self.messages) < 2:
            raise ValueError("TaskResults must have at least two messages.")
        return self

    def save(self, log_dir: Path, filename: str) -> None:
        log_dir.mkdir(exist_ok=True)
        file_path = log_dir / filename
        with file_path.open("w") as f:
            dump(self.model_dump(mode="json"), f, indent=2)


def aggregate(results: list[bool]) -> float:
    return sum(results) / len(results) if results else 0.0


def aggregate_utilty(results: Iterable[TaskResults]) -> float:
    return aggregate([result.utility for result in results])


def aggregate_security(results: Iterable[TaskResults]) -> float:
    return aggregate([result.security for result in results])


class SuiteResults:
    suite: str
    user_runs: dict[str, TaskResults]
    injection_runs: dict[str, TaskResults]
    attack_runs: dict[tuple[str, str], TaskResults]
    utility_user_tasks: float = 0.0
    utility_injection_tasks: float = 0.0
    utility_with_attack: float = 0.0
    security: float = 0.0
    count: int = 0

    def __init__(
        self,
        suite: str,
        user_runs: dict[str, TaskResults],
        injection_runs: dict[str, TaskResults],
        attack_runs: dict[tuple[str, str], TaskResults],
    ) -> None:
        self.suite = suite
        self.user_runs = user_runs
        self.injection_runs = injection_runs
        self.attack_runs = attack_runs

        self.utility_user_tasks = aggregate_utilty(self.user_runs.values())
        self.utility_injection_tasks = aggregate_utilty(self.injection_runs.values())
        self.utility_with_attack = aggregate_utilty(self.attack_runs.values())
        self.security = aggregate_security(self.attack_runs.values())
        self.count = (
            len(self.user_runs) + len(self.injection_runs) + len(self.attack_runs)
        )

    def save(self, log_dir: Path) -> None:
        suite_folder = log_dir / self.suite
        suite_folder.mkdir(exist_ok=True)

        results_data = {
            "suite": self.suite,
            "utility_user_tasks": self.utility_user_tasks,
            "utility_injection_tasks": self.utility_injection_tasks,
            "utility_with_attack": self.utility_with_attack,
            "security": self.security,
            "count": self.count,
        }
        with (suite_folder / "results.json").open("w") as f:
            dump(results_data, f, indent=2)


@dataclass
class BenchmarkResults:
    suite_results: dict[str, SuiteResults]
    model: str
    agent: str
    timestamp: str
    commit: str
    suites: list[str]
    user_tasks: list[str] | None = None
    injection_tasks: list[str] | None = None

    def test_count(self) -> int:
        return sum([sr.count for sr in self.suite_results.values()])

    def _serialize_task_list(self, tasks: list[str] | None) -> list[str] | str:
        """Convert None to 'ALL_TESTS' sentinel for JSON serialization."""
        return "ALL_TESTS" if tasks is None else tasks

    def save(self, log_dir: Path) -> None:
        results_data = asdict(self)
        results_data.pop("suite_results")
        results_data["test_count"] = self.test_count()
        results_data["user_tasks"] = self._serialize_task_list(self.user_tasks)
        results_data["injection_tasks"] = self._serialize_task_list(
            self.injection_tasks
        )
        file_path = log_dir / "bench.json"
        with file_path.open(mode="w") as file:
            dump(results_data, file, indent=2)


def run_task(
    suite: agentdojo.task_suite.TaskSuite,
    agent_pipeline: pipeline.BasePipelineElement,
    user_task: agentdojo.base_tasks.BaseUserTask | None,
    injection_task: agentdojo.base_tasks.BaseInjectionTask | None,
    attack: agentdojo.attacks.BaseAttack | None,
    logdir: Path | None,
    force_rerun: bool,  # noqa: FBT001
) -> TaskResults:
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

    task = user_task or injection_task

    assert task is not None, "Either user_task or injection_task should be set"

    with_attack = False
    injection_task_id = injection_task.ID if injection_task else "none"
    injections: dict[str, str] = {}
    attack_name = "none"

    if user_task and injection_task:
        assert attack is not None, (
            "Attack must be set for injection task to be injected in the environment"
        )
        with_attack = True
        attack_name = attack.name
        injections = attack.attack(user_task, injection_task)

    if logdir is not None:
        try:
            task_result = load_task_results(
                agent_pipeline=agent_pipeline.name or "none",
                suite_name=suite.name,
                user_task=user_task.ID if user_task else "none",
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
                f" '{task_result.user_task_id}' with '{task_result.injection_task_id}'"
                " because it was already run."
            )
        else:
            logging.info(
                f"Skipping task '{task_result.user_task_id}' with no injections because it was"
                " already run."
            )
        return task_result

    start = time.time()
    with logging.CaptureLogger() as cap:
        try:
            if with_attack:
                utility, security = suite.run_task_with_pipeline(
                    agent_pipeline=agent_pipeline,
                    user_task=task,
                    injection_task=injection_task,
                    injections=injections,
                )
            else:
                utility, security = suite.run_task_with_pipeline(
                    agent_pipeline=agent_pipeline,
                    user_task=task,
                    injection_task=None,
                    injections={},
                )
        except agentdojo.benchmark.BadRequestError as e:
            if (
                e.code == "context_length_exceeded"
                or e.param == "max_tokens"
                or "reduce the length of the messages." in repr(e)
            ):
                logging.log_error(
                    f"Skipping task {task.ID} due to context_length_exceeded: {e}"
                )
                utility = False
                security = True
            else:
                raise
        except agentdojo.benchmark.UnprocessableEntityError as e:
            if "max_new_tokens" in str(e):
                logging.log_error(
                    f"Skipping task {task.ID} due to context_length_exceeded: {e}"
                )
                utility = False
                security = True
            else:
                raise
        except agentdojo.benchmark.ApiError as e:
            if "internal server error" in str(e):
                logging.log_error(
                    f"Skipping task {task.ID} because of internal server error: {e}"
                )
                utility = False
                security = True
            else:
                raise
        except agentdojo.benchmark.ServerError as e:
            logging.log_error(
                f"Skipping task {task.ID} because of internal server error: {e}"
            )
            utility = False
            security = True
        messages = cap.messages

    duration = time.time() - start
    return TaskResults(
        user_task_id=user_task.ID if user_task else "none",
        injection_task_id=injection_task_id,
        attack_type=attack_name,
        injections=injections,
        messages=messages,
        utility=utility,
        security=security,
        duration=duration,
    )


def load_task_results(
    agent_pipeline: str,
    suite_name: str,
    user_task: str,
    injection_task: str,
    logdir: Path,
) -> TaskResults | None:
    file_path = Path(suite_name)
    res_dict = None

    if user_task != "none" and injection_task == "none":
        # Only user task was specified
        file_path = file_path / f"user-{user_task}.json"

    elif user_task == "none" and injection_task != "none":
        # Only injection task was specified
        file_path = file_path / f"injection-{injection_task}.json"

    elif user_task != "none" and injection_task != "none":
        # User task was run with an attack
        file_path = file_path / f"attack-{user_task}-{injection_task}.json"

    base_path = logdir.parent

    dirs = [d.name for d in base_path.iterdir() if Path.is_dir(d)]
    dirs.sort(reverse=True)

    for folder in dirs:
        bench_file = base_path / folder / "bench.json"
        if not Path.exists(bench_file):
            continue
        with bench_file.open() as bench_json:
            bench = load(bench_json)
            bench_pipeline = f"{bench['agent']}_{bench['model']}".removeprefix("local_")
            agent_pipeline = agent_pipeline.removeprefix("local_")
            if bench_pipeline != agent_pipeline:
                continue
            file = base_path / folder / file_path
            if not Path.exists(file):
                continue
            with file.open() as test_file:
                res_dict = load(test_file)
                if not TaskResults(**res_dict).error:
                    return TaskResults(**res_dict)

    return None


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


def init_benchmark_dir(log_dir: Path, agent: str) -> tuple[Path, datetime]:
    run_start = datetime.now(tz=UTC)
    timestamp_folder = run_start.astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    bench_dir = log_dir / f"bench-{agent}-{timestamp_folder}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    return bench_dir, run_start


def benchmark_suite(
    agent_pipeline: pipeline.BasePipelineElement,
    suite: agentdojo.task_suite.TaskSuite,
    log_dir: Path | None,
    attack: agentdojo.attacks.BaseAttack | None = None,
    user_tasks: Sequence[str] | None = None,
    injection_tasks: Sequence[str] | None = None,
    n_repeats: int = 1,  # TODO: use n_repeats
    force_rerun: bool = False,
) -> tuple[SuiteResults, bool]:
    """Benchmark a suite.

    Args:
        agent_pipeline: The agent pipeline to use.
        suite: The suite to benchmark.
        log_dir: The directory to save logs to.
        attack: The attack to use in the injection tests.
        user_tasks: A subset of user tasks to run. If None, all user tasks are
            run.
        injection_tasks: A subset of injection tasks to run. If None, all
            injection tasks are run.
        benchmark_version: The benchmarking version for the given suite.
        n_repeats: Sets how many times a given user task is run.
        force_rerun: Sets if earlier runs should be recovered from the log
            files or ignored and run from the start.

    Returns:
        (suite_results, completed): A tuple of the SuiteResults instance
        containing the task results and the `completed` flag which signals
        if benchmarking the suite has run to completion. In case of a
        KeyboardInterrupt this value is set to False.

    """
    user_runs: dict[str, TaskResults] = {}
    injection_runs: dict[str, TaskResults] = {}
    attack_runs: dict[tuple[str, str], TaskResults] = {}

    if user_tasks is not None:
        user_tasks_to_run = {
            user_task_id: suite.get_user_task_by_id(user_task_id)
            for user_task_id in user_tasks
            if user_task_id in suite.user_tasks
        }
    else:
        user_tasks_to_run = suite.user_tasks

    if injection_tasks is not None:
        injection_tasks_to_run = {
            injection_task_id: suite.get_injection_task_by_id(injection_task_id)
            for injection_task_id in injection_tasks
            if injection_task_id in suite.injection_tasks
        }
    else:
        injection_tasks_to_run = suite.injection_tasks

    def run_and_save_task(
        user_task: agentdojo.task_suite.BaseUserTask | None,
        injection_task: agentdojo.task_suite.BaseInjectionTask | None,
        attack: agentdojo.attacks.BaseAttack | None,
        file_name: str,
    ) -> TaskResults:
        task_results = None
        try:
            task_results = run_task(
                suite=suite,
                agent_pipeline=agent_pipeline,
                user_task=user_task,
                injection_task=injection_task,
                attack=attack,
                logdir=log_dir,
                force_rerun=force_rerun,
            )
            if log_dir:
                task_results.save(log_dir=log_dir / suite.name, filename=file_name)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            logging.exception(str(e))
            task_results = TaskResults(
                user_task_id=user_task.ID if user_task else None,
                injection_task_id=injection_task.ID if injection_task else None,
                attack_type=attack.name if attack else None,
                injections=None,
                messages=[],
                utility=False,
                security=False,
                duration=0.0,
                error=str(e),
            )
            if log_dir:
                task_results.save(log_dir / suite.name, file_name)
        return task_results

    try:
        for user_task_id, user_task in user_tasks_to_run.items():
            task_results = run_and_save_task(
                user_task=user_task,
                injection_task=None,
                attack=None,
                file_name=f"user-{user_task_id}.json",
            )
            user_runs[user_task_id] = task_results

        for injection_task_id, injection_task in injection_tasks_to_run.items():
            task_results = run_and_save_task(
                user_task=None,
                injection_task=injection_task,
                attack=None,
                file_name=f"injection-{injection_task_id}.json",
            )
            injection_runs[injection_task_id] = task_results

        for user_task_id, user_task in user_tasks_to_run.items():
            for injection_task_id, injection_task in injection_tasks_to_run.items():
                task_results = run_and_save_task(
                    user_task=user_task,
                    injection_task=injection_task,
                    attack=attack,
                    file_name=f"attack-{user_task_id}-{injection_task_id}.json",
                )
                attack_runs[(user_task_id, injection_task_id)] = task_results
    except KeyboardInterrupt:
        return SuiteResults(
            suite=suite.name,
            user_runs=user_runs,
            injection_runs=injection_runs,
            attack_runs=attack_runs,
        ), False
    return SuiteResults(
        suite=suite.name,
        user_runs=user_runs,
        injection_runs=injection_runs,
        attack_runs=attack_runs,
    ), True


def benchmark(
    agent_loop: agent.ADAgentLoop,
    agent: str,
    model: str,
    log_dir: Path = Path(".log"),
    attack: str | None = None,
    suites: list[str] | None = None,
    user_tasks: list[str] | None = None,
    injection_tasks: list[str] | None = None,
    suite_version: str = "v1.2",  # TODO: use different version?
    n_repeats: int = 1,
    force_rerun: bool = False,
) -> BenchmarkResults:
    bench_dir, timestamp = init_benchmark_dir(log_dir, agent)
    suites_to_run: dict[str, agentdojo.task_suite.TaskSuite] = (
        agentdojo.task_suite.get_suites(suite_version)
    )
    if suites:
        suites_to_run = {
            suite: agentdojo.task_suite.get_suite(suite_version, suite)
            for suite in suites
        }
    suite_results = {}
    for suite_name, suite in suites_to_run.items():
        logging.debug(f"Executing suite {suite_name}")
        agent_pipeline = pipeline.AgentPipeline([agent_loop])
        agent_pipeline.name = f"local_{agent}_{model}"
        """`local` needs to be part of the agent pipeline name attribute
        in order for some AD attacks to run that need to fetch the model name
        for certain injection vectors."""

        base_attack = None
        if attack is not None:
            base_attack = agentdojo.attacks.load_attack(attack, suite, agent_pipeline)
        results: SuiteResults
        results, completed = benchmark_suite(
            agent_pipeline=agent_pipeline,
            suite=suite,
            log_dir=bench_dir,
            attack=base_attack,
            user_tasks=user_tasks,
            injection_tasks=injection_tasks,
            n_repeats=n_repeats,
            force_rerun=force_rerun,
        )
        results.save(log_dir=bench_dir)
        if base_attack is not None:
            print(
                f"Suite: {suite_name}\tUtility score with attack: "
                f"{results.utility_with_attack}"
            )
            print(f"\t\t\tUtility (user tasks): {results.utility_user_tasks}")
            print(f"\t\t\tUtility (injection tasks): {results.utility_injection_tasks}")
            print(f"\t\t\tSecurity score: {results.security}")
        else:
            print(
                f"Suite: {suite_name}\tUtility score (user tasks): \t\t"
                f"{results.utility_user_tasks}"
            )
            print(
                "\t\tUtility score (injection tasks):\t"
                f"{results.utility_injection_tasks}"
            )
        suite_results[suite_name] = results
        if not completed:
            break
    benchmark_results = BenchmarkResults(
        suite_results=suite_results,
        agent=agent,
        model=model,
        timestamp=timestamp.astimezone().isoformat(),
        commit=_get_git_revision() or "none",
        suites=list(suites_to_run.keys()),
        user_tasks=user_tasks,
        injection_tasks=injection_tasks,
    )
    benchmark_results.save(bench_dir)
    return benchmark_results
