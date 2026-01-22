# Copied from https://github.com/ethz-spylab/agentdojo/blob/5cea5891fa8e6b13c4299a94691e1ec64d445fcd/src/agentdojo/agent_pipeline/tool_execution.py
import logging
import sys
from ast import literal_eval
from collections.abc import Callable
from dataclasses import replace

import yaml
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.functions_runtime import (
    FunctionReturnType,
)
from agentdojo.types import (
    ChatToolResultMessage,
    text_content_block_from_string,
)
from pydantic import BaseModel

from lamb.types import Transition, State


def is_string_list(s: str):
    try:
        parsed = literal_eval(s)
        return isinstance(parsed, list)
    except (ValueError, SyntaxError):
        return False


def tool_result_to_str(
    tool_result: FunctionReturnType,
    dump_fn: Callable[[dict | list[dict]], str] = yaml.safe_dump,
) -> str:
    """Basic tool output formatter with YAML dump by default. Could work with `json.dumps` as
    `dump_fn`."""
    if isinstance(tool_result, BaseModel):
        return dump_fn(tool_result.model_dump()).strip()

    if isinstance(tool_result, list):
        res_items = []
        for item in tool_result:
            if type(item) in [str, int]:
                res_items += [str(item)]
            elif isinstance(item, BaseModel):
                res_items += [item.model_dump()]
            else:
                raise TypeError(
                    "Not valid type for item tool result: " + str(type(item))
                )

        # If type checking passes, this is guaranteed to be a list of BaseModel
        return dump_fn(res_items).strip()

    return str(tool_result)


def tool_executor(
    state: State,
    tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
) -> State:
    """Executes the tool calls in the last messages for which tool execution is required.

    Args:
        state: the last state
        tool_output_formatter: a function that converts a tool's output into plain text to be fed to the model.
            It should take as argument the tool output, and convert it into a string. The default converter
            converts the output to structured YAML.
    """

    if len(state.messages) == 0:
        return state
    last_message = state.messages[-1]
    if last_message["role"] != "assistant":
        return state
    if last_message["tool_calls"] is None or len(last_message["tool_calls"]) == 0:
        return state

    tool_call_results = []
    for tool_call in last_message["tool_calls"]:
        if tool_call.function == EMPTY_FUNCTION_NAME:
            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string("")],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error="Empty function name provided. Provide a valid function name.",
                )
            )
            continue
        if tool_call.function not in (
            tool.name for tool in state.runtime.functions.values()
        ):
            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string("")],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=f"Invalid tool {tool_call.function} provided.",
                )
            )
            continue

        # Converts type of input lists from string to list type
        for arg_k, arg_v in tool_call.args.items():
            if isinstance(arg_v, str) and is_string_list(arg_v):
                tool_call.args[arg_k] = literal_eval(arg_v)

        tool_call_result, error = state.runtime.run_function(
            state.env, tool_call.function, tool_call.args
        )
        tool_call_id = tool_call.id
        formatted_tool_call_result = tool_output_formatter(tool_call_result)
        tool_call_results.append(
            ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(formatted_tool_call_result)],
                tool_call_id=tool_call_id,
                tool_call=tool_call,
                error=error,
            )
        )
    return replace(state, messages=[*state.messages, *tool_call_results])


def controller(
    state: State,
    tool_executor: Transition,
    llm: Transition,
    max_iters: int = 25,
) -> State:
    nr_messages = len(state.messages)
    assert nr_messages >= 2, (
        "There must be at least the system message and the initial prompt"
    )

    last_message = state.messages[-1]
    next_state: State
    match last_message:
        # Initial user prompt
        case {"role": "user"}:
            next_state = llm(state)
        # evaluate tool calls
        case {"role": "assistant", "tool_calls": calls} if (
            calls is not None and len(calls) > 0
        ):
            tool_state = tool_executor(state)
            next_state = llm(tool_state)
        # assistant finished without calling tools
        case {"role": "assistant"}:
            return state
        case _:
            logging.error(f"Illegal message role: {last_message['role']}")
            sys.exit(1)
    assert len(next_state.messages) > nr_messages, "LLM must populate messages"
    return controller(next_state, tool_executor, llm, max_iters=max_iters - 1)
