import json
import typing
from collections.abc import Callable

import agentdojo.functions_runtime as rt
import jsonschema
from agentdojo.types import ChatMessage

from lamb import runtime

if typing.TYPE_CHECKING:
    from openai.types.chat.completion_create_params import ResponseFormat

    from lamb.agent import Agent


def query_llm(agent: "Agent", prompt: str) -> str:
    history, core = agent.prompt(prompt)
    response = _get_response(history)
    # expand all variables
    for var, val in core.formatter.var_vals.items():
        response = response.replace(var, val)
    return response


def query_llm_structured(
    agent: "Agent",
    prompt: str,
    schema: dict,
) -> dict:
    if not _check_schema(schema):
        raise ValueError("There must be no strings in the schema.")
    response_format: ResponseFormat = {
        "type": "json_schema",
        "json_schema": {
            "name": "Structured output schema",
            "description": "You must conform to this output in your final answer to the user.",  # noqa: E501
            "schema": schema,
            "strict": True,
        },
    }
    history, core = agent.prompt(prompt, response_format)
    response = _get_response(history)
    # expand all variables
    for var, val in core.formatter.var_vals.items():
        response = response.replace(var, val)

    # TODO: catch errors and prevent them from leaking sensitive information
    result_dict = json.loads(response)
    jsonschema.validate(result_dict, schema)
    return result_dict


def make_query_llm_fn(fn: Callable[[str], str]) -> rt.Function:
    return runtime.make_function(
        "query_llm",
        """Query a different LLM.

        You can use this tool to process the content of variables.
        Variables, like <tool_result_1/>, given in the prompt are automatically expanded.
        The result is always a string stored in a new variable.
        You can use it for more tool calls or the final answer to the user.""",  # noqa: E501
        [("prompt", "str", "The prompt to the LLM. May contain variables.")],
        fn,
    )


def make_query_llm_fn_with_agent(agent: "Agent") -> rt.Function:
    return make_query_llm_fn(lambda prompt: query_llm(agent, prompt))


def make_query_llm_structured_fn_with_agent(agent: "Agent") -> rt.Function:
    return make_query_llm_structured_fn(
        lambda prompt, json_schema: query_llm_structured(agent, prompt, json_schema)
    )


def make_query_llm_structured_fn(fn: Callable[[str, dict], dict]) -> rt.Function:
    return runtime.make_function(
        "query_llm_structured",
        """Query a different LLM and get a structured output.

        You can use this tool to process the content of variables.
        Variables, like <tool_result_1/>, given in the prompt are automatically expanded.

        Use the json_schema argument to control the type of the structured output
        you will receive.
        This means you can make decisions based on the content of that structured output.

        For security reasons, the structured output can NEVER
        contain arbitrary strings (but enum values, ints, floats, bools).
        If the schema allows for arbitrary strings, the tool call will fail.""",  # noqa: E501
        [
            ("prompt", "str", "The prompt to the LLM. May contain variables."),
            (
                "json_schema",
                "dict",
                "A JSON schema that the function output will conform to.",
            ),
        ],
        fn,
    )


def _get_response(history: list[ChatMessage]) -> str:
    assert len(history) >= 3, "LLM must have added a final message"
    last_message = history[-1]
    assert last_message["role"] == "assistant", (
        "Last message must be an assistant message"
    )
    assert last_message["content"] is not None and len(last_message["content"]) >= 1, (
        "Assistant message must have content"
    )
    return last_message["content"][0]["content"]


def _check_schema(schema: dict | list | bool | str | float) -> bool:  # noqa: FBT001
    match schema:
        case bool() | str() | int() | float():
            return True
        case list():
            return all(_check_schema(sub_schema) for sub_schema in schema)
        case {"type": "string"}:
            return False
        case dict():
            return all(_check_schema(sub_schema) for sub_schema in schema.values())
        case _:
            typing.assert_never(schema)
