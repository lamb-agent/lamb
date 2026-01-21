import logging
import sys
import time

import agentdojo.agent_pipeline as pipeline
import openai

import lamb.controller as controller
from lamb.prompting_llm import PromptingLLM
from lamb.types import Llm, PipeElementWrapper, State


def local(model: str) -> pipeline.BasePipelineElement:
    return _new_prompting_llm_openai(
        model=model,
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )


def gemma(api_key: str) -> pipeline.BasePipelineElement:
    return _new_prompting_llm_openai(
        model="gemma-3-27b-it",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )


def _new_prompting_llm_openai(
    model: str,
    base_url: str,
    api_key: str,
) -> pipeline.BasePipelineElement:
    """Instantiate prompting llm with openai client"""

    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    llm = PromptingLLM(pipeline.OpenAILLM(client=client, model=model))
    return PipeElementWrapper(_make_run(llm.next))


def _make_run(llm: Llm):
    """Create run function that executes the agent loop."""

    def retry_llm(state: State, timeout: int = 30):
        """Retries calling the llm if rate limit is exceeded.

        Fails gracefully, if connection couldn't be established.
        """

        try:
            return llm(state)
        except openai.RateLimitError:
            logging.error("Google API exhausted, waiting...")
            time.sleep(timeout)
            return retry_llm(state)
        except openai.APIConnectionError:
            logging.error("Failed to connect")
            sys.exit(1)

    def run(state: State):
        return controller.controller(
            state=state,
            tool_executor=controller.tool_executor,
            llm=retry_llm,
        )

    return run
