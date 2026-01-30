import logging
import sys
import time
from enum import Enum

import agentdojo.agent_pipeline as pipeline
import openai

from lamb.prompting_llm import PromptingLLM
from lamb.types import State, Transition


class OllamaModel(Enum):
    DEEPSEEK = "deepseek-r1:1.5b"
    FUNCTIONGEMMA = "functiongemma:latest"
    GEMMA = "gemma3:4b"
    LLAMA3 = "llama3.2:3b"
    MINISTRAL = "ministral-3:3b"
    MISTRAL = "mistral:7b"
    PHI = "phi3:3.8b"


class CerebrasModel(Enum):
    GPT_OSS = "gpt-oss-120b"
    LLAMA3 = "llama-3.3-70b"


def local(model: str) -> Transition:
    return _new_prompting_llm_openai(
        model=model,
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )


def gemma(api_key: str) -> Transition:
    return _new_prompting_llm_openai(
        model="gemma-3-27b-it",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )


def cerebras(model: str, api_key: str) -> Transition:
    return _new_prompting_llm_openai(
        model=model,
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )


def _new_prompting_llm_openai(
    model: str,
    base_url: str,
    api_key: str,
) -> Transition:
    """Instantiate prompting llm with openai client"""

    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    llm = PromptingLLM(pipeline.OpenAILLM(client=client, model=model))
    return _make_retry(llm.next)


def _make_retry(llm: Transition):
    """Make the llm retry on rate limit exhaustion."""

    def retry(state: State, timeout: int = 30) -> State:
        """Retries calling the llm if rate limit is exceeded.

        Fails gracefully, if connection couldn't be established.
        """

        try:
            return llm(state)
        except openai.RateLimitError:
            logging.error("Google API exhausted, waiting...")
            time.sleep(timeout)
            return retry(state)
        except openai.APIConnectionError:
            logging.error("Failed to connect")
            sys.exit(1)

    return retry
