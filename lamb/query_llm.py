import json
import typing
from collections.abc import Callable

import agentdojo.functions_runtime as rt
import jsonschema
import pydantic
from agentdojo.types import ChatMessage
from pydantic.main import IncEx

from lamb import ifc, runtime, types

if typing.TYPE_CHECKING:
    from openai.types.chat.completion_create_params import ResponseFormat

    from lamb.agent import Agent


class QueryLlmResponse(pydantic.BaseModel):
    response: str | int | float | list | dict
    history: list[ChatMessage]
    ifc_label: ifc.IFCLabel | None

    def model_dump(
        self,
        *,
        mode: str = "python",  # noqa: ARG002
        include: IncEx | None = None,  # noqa: ARG002
        exclude: IncEx | None = None,  # noqa: ARG002
        context: typing.Any | None = None,  # noqa: ARG002
        by_alias: bool | None = None,  # noqa: ARG002
        exclude_unset: bool = False,  # noqa: ARG002
        exclude_defaults: bool = False,  # noqa: ARG002
        exclude_none: bool = False,  # noqa: ARG002
        exclude_computed_fields: bool = False,  # noqa: ARG002
        round_trip: bool = False,  # noqa: ARG002
        warnings: bool | typing.Literal["none", "warn", "error"] = True,  # noqa: ARG002
        fallback: Callable[[typing.Any], typing.Any] | None = None,  # noqa: ARG002
        serialize_as_any: bool = False,  # noqa: ARG002
    ) -> dict[str, typing.Any]:
        """Override pydantic model dump to include the response only."""

        return self.response  # type: ignore


def query_llm(agent: "Agent", prompt: str) -> QueryLlmResponse:
    history, ifc_label = agent.prompt(prompt)
    response = types.get_response(history)
    return QueryLlmResponse(response=response, history=history, ifc_label=ifc_label)


def query_llm_structured(
    agent: "Agent",
    prompt: str,
    schema: dict,
) -> QueryLlmResponse:
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
    history, ifc_label = agent.prompt(prompt, response_format)
    if ifc_label:
        # Structured outputs are trusted,
        # because they don't carry arbitrary strings that could poison the LLM
        ifc_label = ifc_label.set_integ(ifc.Integrity.TRUSTED)
    response = types.get_response(history)

    # TODO: catch errors and prevent them from leaking sensitive information
    result_dict = json.loads(response)
    jsonschema.validate(result_dict, schema)
    return QueryLlmResponse(response=result_dict, history=history, ifc_label=ifc_label)


def make_query_llm_fn(fn: Callable[[str, str], QueryLlmResponse]) -> rt.Function:
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
    return make_query_llm_fn(lambda prompt, ifc_label: query_llm(agent, prompt))  # noqa: ARG005


def make_query_llm_structured_fn_with_agent(agent: "Agent") -> rt.Function:
    return make_query_llm_structured_fn(
        lambda prompt, json_schema, ifc_label: query_llm_structured(  # noqa: ARG005
            agent, prompt, json_schema
        )
    )


def make_query_llm_structured_fn(
    fn: Callable[[str, dict, str], QueryLlmResponse],
) -> rt.Function:
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


def _check_schema(schema: dict | list | bool | str | float) -> bool:  # noqa: FBT001
    match schema:
        case bool() | str() | int() | float():
            return True
        case list():
            return all(_check_schema(sub_schema) for sub_schema in schema)
        # TODO: invest: would {"TYPE": "STRING"} fly?
        case {"type": "string"}:
            return False
        case dict():
            return all(_check_schema(sub_schema) for sub_schema in schema.values())
        case _:
            typing.assert_never(schema)
