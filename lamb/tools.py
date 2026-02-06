import agentdojo.functions_runtime as rt

from lamb import types


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
