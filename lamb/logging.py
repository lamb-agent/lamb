import logging
import typing
from collections.abc import Sequence

import rich.logging
from agentdojo.logging import Logger

from lamb import types


class CaptureLogger(Logger):
    """A simple logger that captures the `log(messages, ...)` call
    into an in-memory list so that it can be serialized after
     running a test.
    """

    def __init__(self) -> None:
        self.messages = []
        self.error = None

    def log_error(self, error: str) -> None:
        self.error = error

    def log(self, messages: list[types.ChatMessage]) -> None:
        self.messages = messages


logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent double logging from root logger

handler = rich.logging.RichHandler(
    show_path=False, markup=True, log_time_format="%H:%M:%S"
)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def info(message: str) -> None:
    logger.info(message)


def debug(message: str) -> None:
    logger.debug(message)


def warning(message: str) -> None:
    logger.warning(message)


def exception(message: str) -> None:
    logger.exception(message)


def log_error(message: str) -> None:
    logger.error(message)


def log_message(identity: types.Identity, message: types.ChatMessage) -> None:
    match logger.level:
        case logging.DEBUG:
            # log entire message instance
            logger.debug(format_chat_message(identity, message.role, f"{message}"))
        case _:
            log: str
            match message:
                case types.AssistantMessage():
                    content = message.content
                    tool_calls = [format_tool_call(call) for call in message.tool_calls]
                    if len(tool_calls) > 0:
                        if message.content != "":
                            message.content += "\n\t"
                        content += "\n".join(tool_calls)
                    log = format_chat_message(identity, message.role, content)
                case types.ToolMessage():
                    content = message.content
                    if message.error:
                        if content != "":
                            content += "\n\t"
                        content += message.error
                    log = format_chat_message(identity, message.role, content)
                case types.UserMessage():
                    log = format_chat_message(identity, message.role, message.content)
                case _:
                    typing.assert_never(message)
            logger.info(log)


def log_messages(
    identity: types.Identity, messages: Sequence[types.ChatMessage]
) -> None:
    for message in messages:
        log_message(identity, message)


def format_chat_message(identity: types.Identity, role: types.Role, text: str) -> str:
    identity_symbol: str
    match identity:
        case types.Identity.SINGLE:
            identity_symbol = ":dromedary_camel:"
        case types.Identity.PRIVILEGED:
            identity_symbol = ":video_game:"  # game controller
        case types.Identity.QUARANTINED:
            identity_symbol = ":no_entry:"
        case types.Identity.BOUNDED:
            identity_symbol = ":ewe:"
    role_symbol: str
    role_colour: str
    match role:
        case "user":
            role_symbol = ":bust_in_silhouette:"
            role_colour = "green"
        case "tool":
            role_symbol = ":wrench:"
            role_colour = "orange"
        case "assistant":
            role_symbol = ":robot_face:"
            role_colour = "cyan"
        case _:
            typing.assert_never(role)
    return f"{identity_symbol}{role_symbol} [{role_colour}]{text}[/{role_colour}]"


def format_tool_call(call: types.FunctionCall) -> str:
    args = [f"{name}={value}" for name, value in call.args.items()]
    arg_string = ""
    if len(args) > 0:
        arg_string += "\n\t\t" + "\n\t\t".join(args)
    return f"{call.function}({arg_string})"
