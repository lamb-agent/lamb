import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import agentdojo.functions_runtime as rt
import agentdojo.types as ad
from openai.types.shared_params import ResponseFormatText

type CallLLM = Callable[[list[ChatMessage]], ChatMessage]
type CallTools = Callable[[list[FunctionCall]], list[ChatMessage]]
type Args = Mapping[str, FunctionCallArgTypes]

TEXT_FORMAT: ResponseFormatText = {"type": "text"}


@runtime_checkable
class Formatter(Protocol):
    def format(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
    ) -> tuple[str, str, str]: ... # TODO: update label type
    def expand(
        self,
        tool: rt.Function,
        args: Args,
    ) -> tuple[Args, str, str]: ...


class Identity(Enum):
    SINGLE = "single"
    PRIVILEGED = "priviledged"
    QUARANTINED = "quarantined"
    BOUNDED = "bounded"


type Role = typing.Literal["user", "assistant", "tool"]

type IFCLabelStr = typing.Literal["TL", "UL", "TH", "UH", ""]


@dataclass
class FunctionCall:
    """An object containing information about a function call requested by an agent."""

    function: str
    """The name of the function to call."""
    args: typing.MutableMapping[str, "FunctionCallArgTypes"]
    """The arguments to pass to the function."""
    id: str
    """An optional ID for the function call. E.g., used by OpenAI and Anthropic."""
    placeholder_args: Mapping[str, "FunctionCallArgTypes"] | None = None
    """An optional dictionary of placeholder arguments to use in by ground truth agent in injection tasks."""  # noqa: E501


type FunctionCallArgTypes = (
    str
    | int
    | float
    | bool
    | None
    | dict[FunctionCallArgTypes, FunctionCallArgTypes]
    | list[FunctionCallArgTypes]
    | FunctionCall
)


def is_arg(obj: object) -> typing.TypeGuard[FunctionCallArgTypes]:
    match obj:
        case str() | int() | float() | bool() | None | FunctionCall():
            return True
        case list() if is_arg_list(obj):
            return True
        case dict() if is_arg_dict(obj):
            return True
        case _:
            return False


def is_arg_dict(
    obj: dict,
) -> typing.TypeGuard[dict[FunctionCallArgTypes, FunctionCallArgTypes]]:
    return all(is_arg(key) and is_arg(val) for key, val in obj.items())


def is_arg_list(
    obj: list,
) -> typing.TypeGuard[dict[FunctionCallArgTypes, FunctionCallArgTypes]]:
    return all(is_arg(elem) for elem in obj)


def arg_to_ad(arg: FunctionCallArgTypes) -> rt.FunctionCallArgTypes:
    match arg:
        case FunctionCall():
            return function_call_to_ad(arg)
        case _:
            return arg


def args_to_ad(args: Args) -> typing.MutableMapping[str, rt.FunctionCallArgTypes]:
    return {key: arg_to_ad(val) for key, val in args.items()}


def function_call_to_ad(call: FunctionCall) -> rt.FunctionCall:
    return ad.FunctionCall(
        function=call.function,
        args=args_to_ad(call.args),
        id=call.id,
        placeholder_args=(
            args_to_ad(call.placeholder_args) if call.placeholder_args else None
        ),
    )


def content_to_ad(content: str) -> list[ad.MessageContentBlock]:
    return [ad.TextContentBlock(type="text", content=content)]


@dataclass
class UserMessage:
    content: str
    label: str
    role: typing.Literal["user"] = "user"


@dataclass
class AssistantMessage:
    content: str
    tool_calls: list[FunctionCall]
    label: str # TODO: update those to ifc label type
    role: typing.Literal["assistant"] = "assistant"


@dataclass
class ToolMessage:
    content: str
    error: str | None
    tool_call_id: str
    tool_call: FunctionCall
    call_source: str
    call_sink: str
    result_source: str
    result_sink: str
    history: list["ChatMessage"] | None = None
    role: typing.Literal["tool"] = "tool"


type ChatMessage = UserMessage | AssistantMessage | ToolMessage


def message_to_ad(message: ChatMessage) -> ad.ChatMessage:
    match message:
        case UserMessage(content, role):
            return ad.ChatUserMessage(
                role=role,
                content=content_to_ad(content),
            )
        case AssistantMessage(content, tool_calls, label, role):
            msg = ad.ChatAssistantMessage(
                role=role,
                tool_calls=[function_call_to_ad(call) for call in tool_calls],
                content=content_to_ad(content),
            )
            msg["label"] = label  # type: ignore
            return msg
        case ToolMessage(
            content,
            error,
            tool_call_id,
            tool_call,
            call_source,
            call_sink,
            result_source,
            result_sink,
            history,
            role,
        ):
            tool_message = ad.ChatToolResultMessage(
                tool_call=function_call_to_ad(tool_call),
                content=content_to_ad(content),
                role=role,
                tool_call_id=tool_call_id,
                error=error,
            )
            tool_message["call_source"] = call_source # type: ignore
            tool_message["call_sink"] = call_sink # type: ignore
            tool_message["result_source"] = result_source # type: ignore
            tool_message["result_sink"] = result_sink # type: ignore
            if history:
                # it's just a dict, we can append the history for logging
                tool_message["history"] = history  # type: ignore
            return tool_message
        case _:
            typing.assert_never(message)


def make_system_prompt(prompt: str) -> ChatMessage:
    return UserMessage(prompt, "TL")


def make_user_prompt(prompt: str, label: str) -> ChatMessage:
    return UserMessage(prompt, label)


def make_assistant_prompt(prompt: str) -> ChatMessage:
    return AssistantMessage(prompt, [], "") # TODO: proper value


def get_response(history: list[ChatMessage]) -> str:
    assert len(history) >= 3, "LLM must have added a final message"
    last_message = history[-1]
    assert last_message.role == "assistant", "Last message must be an assistant message"
    return last_message.content
