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
        logging.error("LAMB_GEMINI_API_KEY env var not set")
        sys.exit(1)
    llm = lamb.llm_wrapper.gemma(gemini_key)
    # llm = lamb.llm_wrapper.local(MODEL.value)
    tools_pipeline = pipeline.AgentPipeline(
        [
            pipeline.SystemMessage(
                "You are a helpful AI assistant with tool-calling capabilities."
            ),
            pipeline.InitQuery(),
            llm,
        ]
    )
    # This is an inconsistency in AgentDojo.
    # The benchmark requires the attribute `name` to be set,
    # but the `AgentPipeline` class doesn't have it.
    tools_pipeline.__setattr__("name", "lamb")

    suites = task_suite.get_suites("v1.2")
    with OutputLogger(None, None):
        for suite_name, suite in suites.items():
            if suite_name != "slack":
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
