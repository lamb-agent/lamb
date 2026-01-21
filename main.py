import logging
import os
import sys
from enum import Enum
from pathlib import Path

import agentdojo.agent_pipeline as pipeline
import agentdojo.benchmark as bench
import agentdojo.task_suite as task_suite
from agentdojo.logging import OutputLogger
from rich.logging import RichHandler

import lamb.llm_wrapper
import lamb.tool_execution as tools


class OllamaModel(Enum):
    DEEPSEEK = "deepseek-r1:1.5b"
    FUNCTIONGEMMA = "functiongemma:latest"
    GEMMA = "gemma3:4b"
    LLAMA3 = "llama3.2:3b"
    MINISTRAL = "ministral-3:3b"
    MISTRAL = "mistral:7b"
    PHI = "phi3:3.8b"


MODEL = OllamaModel.LLAMA3


def main():
    try:
        gemini_key = os.environ["LAMB_GEMINI_API_KEY"]
    except KeyError:
        logging.error("LAMB_GENINI_API_KEY env var not set")
        sys.exit(1)
    llm = lamb.llm_wrapper.Gemma(api_key=gemini_key)
    tools_loop = tools.ToolsExecutionLoop(
        [
            tools.ToolsExecutor(),
            llm,
        ]
    )
    tools_pipeline = pipeline.AgentPipeline(
        [
            pipeline.SystemMessage(
                "You are a helpful AI assistant with tool-calling capabilities."
            ),
            pipeline.InitQuery(),
            llm,
            tools_loop,
        ]
    )
    # This is an inconsistency in AgentDojo.
    # The benchmark requires the attribute `name` to be set,
    # but the `AgentPipeline` class doesn't have it.
    tools_pipeline.__setattr__("name", "lamb")

    slack_suite = task_suite.get_suite("v1.2", "slack")
    with OutputLogger(None, None):
        results = bench.benchmark_suite_without_injections(
            suite=slack_suite,
            agent_pipeline=tools_pipeline,
            logdir=Path("./logs"),
            force_rerun=False,
        )
        utility = bench.aggregate_results([results["utility_results"]])
        print(f"Utility: {utility}")


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
