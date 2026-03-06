from collections.abc import Callable, Mapping
from enum import Enum
from typing import Protocol, runtime_checkable

import agentdojo.functions_runtime as rt
import agentdojo.types as ad
from openai.types.shared_params import ResponseFormatText

type CallLLM = Callable[[list[ad.ChatMessage]], ad.ChatMessage]
type CallTools = Callable[[list[rt.FunctionCall]], list[ad.ChatMessage]]
type Args = Mapping[str, rt.FunctionCallArgTypes]

TEXT_FORMAT: ResponseFormatText = {"type": "text"}


@runtime_checkable
class Formatter(Protocol):
    def format(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
    ) -> str: ...
    def expand(
        self,
        tool: rt.Function,
        args: Args,
    ) -> Args: ...


class Identity(Enum):
    SINGLE = "single"
    PRIVILEGED = "priviledged"
    QUARANTINED = "quarantined"
    BOUNDED = "bounded"


class Role(Enum):
    USER = "user"
    TOOL = "tool"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def make_user_prompt(prompt: str) -> ad.ChatMessage:
    return {
        "role": "user",
        "content": [ad.text_content_block_from_string(prompt)],
    }


def make_assistant_prompt(prompt: str) -> ad.ChatAssistantMessage:
    return {
        "role": "assistant",
        "content": [ad.text_content_block_from_string(prompt)],
        "tool_calls": [],
    }


def get_response(history: list[ad.ChatMessage]) -> str:
    assert len(history) >= 3, "LLM must have added a final message"
    last_message = history[-1]
    assert last_message["role"] == "assistant", (
        "Last message must be an assistant message"
    )
    assert last_message["content"] is not None and len(last_message["content"]) >= 1, (
        "Assistant message must have content"
    )
    return last_message["content"][0]["content"]
