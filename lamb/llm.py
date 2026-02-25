import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum

import openai
import openai.types.chat as openai_types
from agentdojo.types import ChatMessage
from openai.types.chat.completion_create_params import ResponseFormat

from lamb.openai_llm import OpenAILLM
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


class CerebrasModel(Enum):
    GPT_OSS = "gpt-oss-120b"
    LLAMA3 = "llama-3.3-70b"


class Llm:
    """A stateless LLM."""

    llm: OpenAILLM

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        reasoning: openai_types.ChatCompletionReasoningEffort,
    ) -> None:
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        llm = OpenAILLM(
            client=client,
            model=model,
            reasoning_effort=reasoning,
        )
        self.llm = llm

    def prompt(
        self,
        runtime: Runtime,
        messages: list[ChatMessage],
        response_format: ResponseFormat,
    ) -> ChatMessage:
        try:
            _, _, _, msgs, _ = self.llm.query(
                "",
                runtime.functions_runtime,
                runtime.env,
                messages,
                {"response_format": response_format},
            )
            assert len(msgs) == len(messages) + 1
            return msgs[-1]
        except openai.RateLimitError:
            logging.exception("Google API exhausted, waiting...")
            time.sleep(30)
            return self.prompt(runtime, messages, response_format)
        except openai.APIConnectionError:
            logging.exception("Failed to connect")
            sys.exit(1)

    @staticmethod
    def local(
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
