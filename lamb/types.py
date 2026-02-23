from collections.abc import Callable, Mapping
from enum import Enum
from typing import Protocol, runtime_checkable

import agentdojo.functions_runtime as rt
import agentdojo.types as ad_types
from openai.types.shared_params import ResponseFormatText

type CallLLM = Callable[[list[ad_types.ChatMessage]], ad_types.ChatMessage]
type CallTools = Callable[[list[rt.FunctionCall]], list[ad_types.ChatMessage]]
type CheckIFC = Callable[[rt.Function, Mapping[str, rt.FunctionCallArgTypes]], None]

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
        args: Mapping[str, rt.FunctionCallArgTypes],
    ) -> Mapping[str, rt.FunctionCallArgTypes]: ...


class Identity(Enum):
    SINGLE = "single"
    PRIVILEDGED = "priviledged"
    QUARANTINED = "quarantined"
    BOUNDED = "bounded"


class Role(Enum):
    USER = "user"
    TOOL = "tool"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def make_user_prompt(prompt: str) -> ad_types.ChatMessage:
    return {
        "role": "user",
        "content": [ad_types.text_content_block_from_string(prompt)],
    }
