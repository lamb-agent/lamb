import logging
import os
import sys
from pathlib import Path

import agentdojo.agent_pipeline as pipeline
import agentdojo.benchmark as bench
import agentdojo.task_suite as task_suite
from agentdojo.logging import OutputLogger
from rich.logging import RichHandler

import lamb.llm_wrapper as llm
import lamb.tool_result as tool_result


def main():
    try:
        gemini_key = os.environ["LAMB_GEMINI_API_KEY"]
    except KeyError:
        logging.error("LAMB_GEMINI_API_KEY env var not set")
        sys.exit(1)
    try:
        cerebras_key = os.environ["LAMB_CEREBRAS_API_KEY"]
    except KeyError:
        logging.error("LAMB_CEREBRAS_API_KEY env var not set")
        sys.exit(1)
    model = llm.cerebras(
        llm.CerebrasModel.GPT_OSS.value,
        cerebras_key,
        {"tool_result_formatter": tool_result.VariableFormatter()},
    )
    tools_pipeline = pipeline.AgentPipeline(
        [
            pipeline.SystemMessage(
                "You are a helpful AI assistant with tool-calling capabilities."
            ),
            pipeline.InitQuery(),
            model,
        ]
    )
    # This is an inconsistency in AgentDojo.
    # The benchmark requires the attribute `name` to be set,
    # but the `AgentPipeline` class doesn't have it.
    tools_pipeline.__setattr__("name", "lamb")

    suites = task_suite.get_suites("v1.2")
    with OutputLogger(None, None):
        for suite_name, suite in suites.items():
            if suite_name == "travel":
                continue
            results = bench.benchmark_suite_without_injections(
                suite=suite,
                agent_pipeline=tools_pipeline,
                logdir=Path("./logs"),
                force_rerun=False,
            )
            utility = bench.aggregate_results([results["utility_results"]])
            print(f"Utility of {suite_name}: {utility}")


def configure_logging():
    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )


if __name__ == "__main__":
    configure_logging()
    main()
