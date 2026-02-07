import logging
import os
import sys
import typing
from pathlib import Path

from agentdojo import agent_pipeline as pipeline
from agentdojo import benchmark as bench
from agentdojo import task_suite
from agentdojo.logging import OutputLogger
from rich.logging import RichHandler

from lamb import (
    controller,
    llm,
    prompts,
    tool_categories,
    tool_exec,
    tool_llm,
    tool_result,
    tools,
    types,
)


def main() -> None:
    suites = task_suite.get_suites("v1.2")
    with OutputLogger(None, None):
        for suite_name, suite in suites.items():
            # retain_tasks(suite, ["user_task_0"])
            lamb_pipeline = create_pipeline(suite)
            results = bench.benchmark_suite_without_injections(
                suite=suite,
                agent_pipeline=lamb_pipeline,
                logdir=Path("./logs"),
                force_rerun=False,
            )
            utility = bench.aggregate_results([results["utility_results"]])
            print(f"Utility of {suite_name}: {utility}")


def create_pipeline(suite: task_suite.TaskSuite) -> pipeline.BasePipelineElement:
    suite_tools = {tool.run for tool in suite.tools}
    untrusted_fns = {tools.query_llm} | (tool_categories.UNTRUSTED & suite_tools)
    read_only_fns = tool_categories.READ_ONLY & suite_tools

    def is_untrusted(fn: typing.Callable) -> bool:
        return fn in untrusted_fns

    try:
        gemini_key = os.environ["LAMB_GEMINI_API_KEY"]
    except KeyError:
        logging.exception("LAMB_GEMINI_API_KEY env var not set")
        sys.exit(1)
    try:
        cerebras_key = os.environ["LAMB_CEREBRAS_API_KEY"]
    except KeyError:
        logging.exception("LAMB_CEREBRAS_API_KEY env var not set")
        sys.exit(1)
    # model = llm.gemma(gemini_key)
    # model=llm.cerebras(llm.CerebrasModel.GPT_OSS, cerebras_key)
    model = llm.local(llm.OllamaModel.GPT_OSS_120B, thinking="medium")
    config = types.Config(
        llm=model,
        system_prompt=prompts.P_LLM_SYSTEM_PROMPT,
        tool_executor=tool_exec.default,
        tool_result_formatter=tool_result.VariableFormatter(hide_result=is_untrusted),
        tool_llm=tool_llm.bounded_llm,
        read_only_tools=read_only_fns,
    )
    agent_loop = types.ADAgentLoop(init=controller.init(config), loop=controller.loop)
    lamb_pipeline = pipeline.AgentPipeline(
        [
            pipeline.InitQuery(),
            agent_loop,
        ]
    )
    # This is an inconsistency in AgentDojo.
    # The benchmark requires the attribute `name` to be set,
    # but the `AgentPipeline` class doesn't have it.
    lamb_pipeline.__setattr__("name", "lamb")
    return lamb_pipeline


def retain_tasks(suite: task_suite.TaskSuite, task_names: list[str]) -> None:
    # ruff: disable[SLF001]
    suite._user_tasks = {
        task_name: task
        for task_name, task in suite._user_tasks.items()
        if task_name in task_names
    }
    suite._injection_tasks = {
        task_name: task
        for task_name, task in suite._injection_tasks.items()
        if task_name in task_names
    }
    # ruff: enable[SLF001]


def configure_logging() -> None:
    fmt = "%(message)s"
    logging.basicConfig(
        format=fmt,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )


if __name__ == "__main__":
    configure_logging()
    main()
