import logging
import os
import time
import typing
from dataclasses import dataclass
from enum import Enum

import openai
import openai.types.chat as openai_types
from openai.types.chat.completion_create_params import ResponseFormat

import lamb.llm_wrapper
from lamb import types
from lamb.runtime import Runtime


class OllamaModel(Enum):
    GEMMA4 = "gemma4:31b"
    GPT_OSS_20B = "gpt-oss:20b"
    GPT_OSS_120B = "gpt-oss:120b"
    LLAMA3_3_70B = "llama3.3:70b"
    LLAMA4_16X17B = "llama4:16x17b"
    MINISTRAL = "ministral-3:3b"
    MISTRAL = "mistral:7b"
    MISTRAL_LARGE_123B = "mistral-large:123b"
    GRANITE4 = "granite4:3b"

    def nickname(self) -> str:
        match self:
            case OllamaModel.GEMMA4:
                return "gemma4"
            case OllamaModel.GPT_OSS_120B:
                return "gpt-oss"
            case _:
                return self.name

class OpenAIModel(Enum):
    GPT5_MINI = "gpt-5-mini-2025-08-07"


class Llm(typing.Protocol):
    def prompt(
        self,
        runtime: Runtime,
        messages: list[types.ChatMessage],
        response_format: ResponseFormat,
    ) -> types.ChatMessage: ...


@dataclass
class LiveLlm(Llm):
    """A stateless LLM."""

    model: str
    api_key: str
    reasoning: openai_types.ChatCompletionReasoningEffort
    base_url: str | None = None

    def prompt(
        self,
        runtime: Runtime,
        messages: list[types.ChatMessage],
        response_format: ResponseFormat,
    ) -> types.ChatMessage:
        try:
            return lamb.llm_wrapper.prompt(
                model=self.model,
                runtime=runtime.functions_runtime,
                messages=messages,
                response_format=response_format,
                base_url=self.base_url,
                api_key=self.api_key,
                reasoning_effort=self.reasoning,
            )
        except openai.RateLimitError:
            logging.exception("Google API exhausted, waiting...")
            time.sleep(30)
            return self.prompt(runtime, messages, response_format)

    @staticmethod
    def ollama_openai(
        model: OllamaModel,
        reasoning: openai_types.ChatCompletionReasoningEffort = "high",
        port: str = "11434",
    ) -> "LiveLlm":
        return LiveLlm(
            model.value,
            base_url=f"http://localhost:{port}/v1",
            api_key="ollama",
            reasoning=reasoning,
        )

    @staticmethod
    def ollama_chat(
        model: OllamaModel,
        reasoning: openai_types.ChatCompletionReasoningEffort = "high",
    ) -> "LiveLlm":
        return LiveLlm(
            model.value,
            base_url="http://localhost:11434",
            api_key="ollama",
            reasoning=reasoning,
        )

    @staticmethod
    def openai(
        model: OpenAIModel,
        reasoning: openai_types.ChatCompletionReasoningEffort = "high",
    ) -> "LiveLlm":
        env_var = "LAMB_OPENAI_API_KEY"
        api_key = os.environ.get(env_var)
        if api_key is None:
            print(f"{env_var} must be set!", file=sys.stderr)
            sys.exit(1)
        return LiveLlm(
            model.value,
            api_key=api_key,
            reasoning=reasoning,
        )


class MockLlm(Llm):
    history: typing.Iterator[types.ChatMessage]

    def __init__(self, history: typing.Iterable[types.ChatMessage]) -> None:
        self.history = iter(history)

    def prompt(
        self,
        runtime: Runtime,  # noqa: ARG002
        messages: list[types.ChatMessage],  # noqa: ARG002
        response_format: ResponseFormat,  # noqa: ARG002
    ) -> types.ChatMessage:
        return self.history.__next__()


@dataclass
class ToolLlm:
    """An LLM with tools available in the runtime."""

    llm: Llm
    runtime: Runtime

    def prompt(
        self,
        messages: list[types.ChatMessage],
        response_format: ResponseFormat,
    ) -> types.ChatMessage:
        return self.llm.prompt(self.runtime, messages, response_format)
