import logging
import sys
from enum import Enum
from pathlib import Path

import agentdojo.agent_pipeline as pipeline
import agentdojo.benchmark as bench
import agentdojo.task_suite as task_suite
import openai
from agentdojo.logging import OutputLogger
from rich.logging import RichHandler

import lamb.tool_execution as tools
from lamb.prompting_llm import PromptingLLM


class OllamaModel(Enum):
    DEEPSEEK = "deepseek-r1:1.5b"
    FUNCTIONGEMMA = "functiongemma:latest"
    GEMMA = "gemma3:4b"
    LLAMA3 = "llama3.2:3b"
    MINISTRAL = "ministral-3:3b"
    MISTRAL = "mistral:7b"
    PHI = "phi3:3.8b"


MODEL = OllamaModel.LLAMA3


def local_llm(model: OllamaModel) -> PromptingLLM:
    try:
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )
        return PromptingLLM(client, model.value)
    except openai.APIConnectionError:
        print("Ollama must be running", file=sys.stderr)
        sys.exit(1)


def main():
    llm = local_llm(MODEL)
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
