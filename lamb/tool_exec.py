# Copied from https://github.com/ethz-spylab/agentdojo/blob/5cea5891fa8e6b13c4299a94691e1ec64d445fcd/src/agentdojo/agent_pipeline/tool_execution.py
from ast import literal_eval
from dataclasses import replace

import agentdojo.functions_runtime as rt
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.types import (
    ChatToolResultMessage,
    text_content_block_from_string,
)

from lamb import types


def is_string_list(s: str) -> bool:
    try:
        parsed = literal_eval(s)
        return isinstance(parsed, list)
    except (ValueError, SyntaxError):
        return False


def default(state: types.State) -> types.State:
    """Execute the tool calls in the last messages for which tool execution is required.

    Args:
        state: the last state

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
                    error="Empty function name provided. "
                    + "Provide a valid function name.",
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

        args = {
            key: state.config.tool_result_formatter.expand(value)
            for key, value in tool_call.args.items()
        }
        tool_call_result, error = state.runtime.run_function(
            state.env, tool_call.function, args
        )
        tool_call_id = tool_call.id
        formatted_tool_call_result = state.config.tool_result_formatter.format(
            tool_call_result, tool_call.function
        )
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


def query_llm(
    tool_llm: rt.Annotated[types.Query, rt.Depends("tool_llm")],
    prompt: str,
) -> str:
    """Query a different LLM.

    You can use this tool to process the content of variables.
    Variables, like <tool_result_1/>, given in the prompt are automatically expanded.
    The result is always a string stored in a new variable.
    You can use it for more tool calls or the final answer to the user.

    :param prompt: The prompt to the LLM. May contain variables.
    """

    # TODO: Add retry mechanism, if the prompt provided does not have enough context.

    return tool_llm(prompt)
