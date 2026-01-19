import ollama
from typing import Callable
from rich.logging import RichHandler
import logging
from enum import Enum
import sys
from sys import exit
from agentdojo.functions_runtime import FunctionsRuntime, make_function
from openai import OpenAI
from agentdojo.agent_pipeline import (
    PromptingLLM,
    ToolsExecutionLoop,
    ToolsExecutor,
    AgentPipeline,
    SystemMessage,
    InitQuery,
)
from agentdojo.logging import OutputLogger
import openai


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
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )
        return PromptingLLM(client, model.value)
    except openai.APIConnectionError:
        print("Ollama must be running", file=sys.stderr)
        exit(1)


def main():
    llm = local_llm(MODEL)
    tools_loop = ToolsExecutionLoop(
        [
            ToolsExecutor(),
            llm,
        ]
    )
    tools_pipeline = AgentPipeline(
        [
            SystemMessage(
                "You are a helpful AI assistant with tool-calling capabilities."
            ),
            InitQuery(),
            llm,
            tools_loop,
        ]
    )

    runtime = FunctionsRuntime([])
    for f in functions():
        runtime.register_function(f)

    with OutputLogger(None, None):
        tools_pipeline.query(
            input("> "), runtime
        )


def configure_logging():
    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )


def functions() -> list[Callable]:
    def sum(a: int, b: int):
        """Sums two numbers.

        :param a: the first number
        :param b: the second number
        """
        return a + b

    def web_search(query: str):
        """Search the web. You MUST use this function for real-time information.

        :param query: the query for the search bar
        """
        return ollama.generate(
            model=MODEL.value,
            system="Give a straight-forward answer to the provided questions",
            prompt=query,
        ).response

    return [sum, web_search]


if __name__ == "__main__":
    configure_logging()
    main()
