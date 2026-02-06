import agentdojo.functions_runtime as rt
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)

from lamb import types


def query_llm(
    tool_llm: rt.Annotated[types.TextQuery, rt.Depends("tool_llm")],
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


def query_llm_structured(
    tool_llm: rt.Annotated[types.StructuredQuery, rt.Depends("tool_llm")],
    prompt: str,
    json_schema: dict,
) -> dict:
    """Query a different LLM and get a structured output.

    You can use this tool to process the content of variables.
    Variables, like <tool_result_1/>, given in the prompt are automatically expanded.

    Use the json_schema argument to control the type of the structured output
    you will receive.
    This means you can make decisions based on the content of that structured output.

    For security reasons, the structured output can NEVER
    contain arbitrary strings (but enum values, ints, floats, bools).
    If the schema allows for arbitrary strings, the tool call will fail.

    :param prompt: The prompt to the LLM. May contain variables.
    :param json_schema: A JSON schema that the function output will conform to.

    """

    # TODO: Add retry mechanism, if the prompt provided does not have enough context.

    return tool_llm(prompt, json_schema)
