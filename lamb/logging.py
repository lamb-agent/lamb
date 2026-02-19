import logging
from collections.abc import Sequence

import rich.logging
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import ChatMessage

from lamb import types

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


def exception(message: str) -> None:
    logger.exception(message)


def log_message(identity: types.Identity, message: ChatMessage) -> None:
    role = types.Role(message["role"])
    match logger.level:
        case logging.DEBUG:
            # log entire message instance
            logger.debug(format_chat_message(identity, role, f"{message}"))
        case _:
            log: str
            match message["role"]:
                case "assistant":
                    content = (message["content"] or [{"content": ""}])[0]["content"]
                    tool_calls = [
                        format_tool_call(call) for call in (message["tool_calls"] or [])
                    ]
                    if len(tool_calls) > 0:
                        if content != "":
                            content += "\n\t"
                        content += "\n".join(tool_calls)
                    log = format_chat_message(identity, role, content)
                case "tool":
                    content = message["content"][0]["content"]
                    if message["error"]:
                        if content != "":
                            content += "\n\t"
                        content += message["error"]
                    log = format_chat_message(identity, role, content)
                case _:
                    log = format_chat_message(
                        identity, role, message["content"][0]["content"]
                    )
            logger.info(log)


def log_messages(identity: types.Identity, messages: Sequence[ChatMessage]) -> None:
    for message in messages:
        log_message(identity, message)


def format_chat_message(identity: types.Identity, role: types.Role, text: str) -> str:
    identity_symbol: str
    match identity:
        case types.Identity.SINGLE:
            identity_symbol = ":camel:"  # dromeda
        case types.Identity.PRIVILEDGED:
            identity_symbol = ":video_game:"  # game controller
        case types.Identity.QUARANTINED:
            identity_symbol = ":no_entry:"
        case types.Identity.BOUNDED:
            identity_symbol = ":ewe:"
    role_symbol: str
    role_colour: str
    match role:
        case types.Role.USER:
            role_symbol = ":bust_in_silhouette:"
            role_colour = "green"
        case types.Role.TOOL:
            role_symbol = ":wrench:"
            role_colour = "orange"
        case types.Role.ASSISTANT:
            role_symbol = ":robot_face:"
            role_colour = "cyan"
        case types.Role.SYSTEM:
            role_symbol = ":book:"
            role_colour = "blue"
    return f"{identity_symbol}{role_symbol} [{role_colour}]{text}[/{role_colour}]"


def format_tool_call(call: FunctionCall) -> str:
    args = [f"{name}={value}" for name, value in call.args.items()]
    arg_string = ""
    if len(args) > 0:
        arg_string += "\n\t\t" + "\n\t\t".join(args)
    return f"{call.function}({arg_string})"
