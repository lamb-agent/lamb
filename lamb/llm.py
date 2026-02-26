import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum

import openai
import openai.types.chat as openai_types
from agentdojo.types import ChatMessage
from openai.types.chat.completion_create_params import ResponseFormat

import lamb.llm_wrapper
from lamb.runtime import Runtime


class OllamaModel(Enum):
    GPT_OSS_20B = "ollama_chat/gpt-oss:20b"
    GPT_OSS_120B = "openai/gpt-oss:120b"
    LLAMA3_3_70B = "ollama_chat/llama3.3:70b"
    LLAMA4_16X17B = "ollama_chat/llama4:16x17b"
    MINISTRAL = "ollama_chat/ministral-3:3b"
    MISTRAL = "ollama_chat/mistral:7b"
    MISTRAL_LARGE_123B = "ollama_chat/mistral-large:123b"
    GRANITE4 = "ollama_chat/granite4:3b"


@dataclass
class Llm:
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
        except openai.APIConnectionError:
            logging.exception("Failed to connect")
            sys.exit(1)

    @staticmethod
    def ollama_openai(
        model: OllamaModel,
        reasoning: openai_types.ChatCompletionReasoningEffort = None,
    ) -> "Llm":
        return Llm(
            model.value,
            "http://localhost:11434/v1",
            api_key="ollama",
            reasoning=reasoning,
        )

    @staticmethod
    def ollama_chat(
        model: OllamaModel,
        reasoning: openai_types.ChatCompletionReasoningEffort = None,
    ) -> "Llm":
        return Llm(
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
