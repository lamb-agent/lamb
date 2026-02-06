import logging
import sys
import time
from dataclasses import replace
from enum import Enum

import openai
import openai.types.chat as openai_types

from lamb import types
from lamb.prompting_llm import PromptingLLM
from lamb.openai_llm import OpenAILLM


class OllamaModel(Enum):
    GPT_OSS_20B = "gpt-oss:20b"
    GPT_OSS_120B = "gpt-oss:120b"
    LLAMA3_3_70B = "llama3.3:70b"
    LLAMA4_16X17B = "llama4:16x17b"
    MINISTRAL = "ministral-3:3b"
    MISTRAL = "mistral:7b"
    MISTRAL_LARGE_123B = "mistral-large:123b"
    GRANITE4 = "granite4:3b"


class CerebrasModel(Enum):
    GPT_OSS = "gpt-oss-120b"
    LLAMA3 = "llama-3.3-70b"


def local_prompting(model: OllamaModel) -> types.Transition:
    return _new_prompting_llm_openai(
        model=model.value,
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )


def local(
    model: OllamaModel,
    thinking: openai_types.ChatCompletionReasoningEffort = None,
) -> types.Transition:
    return _new_openai(
        model=model.value,
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
        thinking=thinking,
    )


def gemma(api_key: str) -> types.Transition:
    return _new_prompting_llm_openai(
        model="gemma-3-27b-it",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )


def cerebras(model: CerebrasModel, api_key: str) -> types.Transition:
    return _new_prompting_llm_openai(
        model=model.value,
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )


def _new_prompting_llm_openai(
    model: str,
    base_url: str,
    api_key: str,
) -> types.Transition:
    """Instantiate prompting llm with openai client."""

    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    llm = PromptingLLM(OpenAILLM(client=client, model=model))
    return _make_retry(llm.next)


def _new_openai(
    model: str,
    base_url: str,
    api_key: str,
    thinking: openai_types.ChatCompletionReasoningEffort,
) -> types.Transition:
    """Instantiate tool-calling llm with openai client."""

    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    llm = OpenAILLM(
        client=client,
        model=model,
        reasoning_effort=thinking,
    )

    def llm_next(state: types.State) -> types.State:
        _, runtime, env, messages, _ = llm.query(
            "",
            state.runtime,
            state.env,
            state.messages,
            {"response_format": state.response_format},
        )
        return replace(state, runtime=runtime, env=env, messages=messages)

    return _make_retry(llm_next)


def _make_retry(llm: types.Transition) -> types.Transition:
    """Make the llm retry on rate limit exhaustion."""

    def retry(state: types.State, timeout: int = 30) -> types.State:
        """Retries calling the llm if rate limit is exceeded.

        Fails gracefully, if connection couldn't be established.
        """

        try:
            return llm(state)
        except openai.RateLimitError:
            logging.exception("Google API exhausted, waiting...")
            time.sleep(timeout)
            return retry(state)
        except openai.APIConnectionError:
            logging.exception("Failed to connect")
            sys.exit(1)

    return retry
