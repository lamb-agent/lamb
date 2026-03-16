import logging
import time
import typing
from dataclasses import dataclass
from enum import Enum

import openai
import openai.types.chat as openai_types
from agentdojo.types import ChatMessage
from openai.types.chat.completion_create_params import ResponseFormat

import lamb.llm_wrapper
from lamb.runtime import Runtime


class OllamaModel(Enum):
    GPT_OSS_20B = "gpt-oss:20b"
    GPT_OSS_120B = "gpt-oss:120b"
    LLAMA3_3_70B = "llama3.3:70b"
    LLAMA4_16X17B = "llama4:16x17b"
    MINISTRAL = "ministral-3:3b"
    MISTRAL = "mistral:7b"
    MISTRAL_LARGE_123B = "mistral-large:123b"
    GRANITE4 = "granite4:3b"

class Llm(typing.Protocol):
    def prompt(
        self,
        runtime: Runtime,
        messages: list[ChatMessage],
        response_format: ResponseFormat,
    ) -> ChatMessage: ...


@dataclass
class LiveLlm(Llm):
    """A stateless LLM."""

    model: str
    base_url: str
    api_key: str
    reasoning: openai_types.ChatCompletionReasoningEffort

    def prompt(
        self,
        runtime: Runtime,
        messages: list[ChatMessage],
        response_format: ResponseFormat,
    ) -> ChatMessage:
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
        reasoning: openai_types.ChatCompletionReasoningEffort = None,
    ) -> Llm:
        return LiveLlm(
            model.value,
            "http://localhost:11434/v1",
            api_key="ollama",
            reasoning=reasoning,
        )

    @staticmethod
    def ollama_chat(
        model: OllamaModel,
        reasoning: openai_types.ChatCompletionReasoningEffort = None,
    ) -> Llm:
        return LiveLlm(
            model.value,
            "http://localhost:11434",
            api_key="ollama",
            reasoning=reasoning,
        )


@dataclass
class ToolLlm:
    """An LLM with tools available in the runtime."""

    llm: Llm
    runtime: Runtime

    def prompt(
        self,
        messages: list[ChatMessage],
        response_format: ResponseFormat,
    ) -> ChatMessage:
        return self.llm.prompt(self.runtime, messages, response_format)
