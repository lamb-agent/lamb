import json
import typing
from dataclasses import replace

import agentdojo.functions_runtime as rt
import jsonschema
from agentdojo import logging

from lamb import controller, prompts, tool_exec, tool_result, types

if typing.TYPE_CHECKING:
    from openai.types.chat.completion_create_params import ResponseFormat


def quarantined_llm(
    llm: types.Transition,
    runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> types.Query:
    """Create Q-LLM query function with no tool-calling abilities."""

    empty_runtime = rt.FunctionsRuntime([])
    config = types.Config(
        llm=llm,
        system_prompt=prompts.Q_LLM_SYSTEM_PROMPT,
        tool_executor=tool_exec.default,
        tool_result_formatter=tool_result.BasicFormatter(),  # No variables
        tool_llm=None,
    )
    initial_state = types.State(
        runtime=empty_runtime,
        env=rt.EmptyEnv(),
        messages=[types.make_user_prompt(config.system_prompt)],
        config=config,
        response_format=types.TEXT_FORMAT,  # Is overwritten by query call anyway
    )

    def loop(state: types.State) -> types.State:
        return state.next_llm()  # Only call the llm, no agent loop

    return _make_query(initial_state, loop)


def bounded_llm(
    llm: types.Transition,
    runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> types.Query:
    """Create B-LLM query function with tool-calling abilities."""

    config = types.Config(
        llm=llm,
        system_prompt=prompts.B_LLM_SYSTEM_PROMPT,
        tool_executor=tool_exec.default,
        tool_result_formatter=tool_result.BasicFormatter(),  # No variables
        tool_llm=None,
    )
    init_fn = controller.init(config)
    initial_state = init_fn(
        runtime,
        env,
        [],
    )
    return _make_query(initial_state, controller.loop)


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


def _make_query(
    initial_state: types.State,
    loop: types.Transition,
) -> types.Query:
    @typing.overload
    def query(prompt: str) -> str: ...
    @typing.overload
    def query(prompt: str, schema: dict) -> dict: ...
    def query(
        prompt: str,
        schema: dict | None = None,
    ) -> str | dict:
        """Query the LLM with the given prompt.

        The LLM is stateless, i.e. the message history is not preserved.
        """

        user_prompt = types.make_user_prompt(prompt)
        response_format: ResponseFormat = types.TEXT_FORMAT
        if schema:
            if not _check_schema(schema):
                raise ValueError("There must be no strings in the schema.")
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "Structured output schema",
                    "description": "You must conform to this output in your final answer to the user.",
                    "schema": schema,
                    "strict": True,
                },
            }
        state = replace(
            initial_state,
            messages=[*initial_state.messages, user_prompt],
            response_format=response_format,
        )
        final_state = loop(state)
        assert len(final_state.messages) >= 3, "LLM must have added a final message"
        last_message = final_state.messages[-1]
        assert last_message["role"] == "assistant", (
            "Last message must be an assistant message"
        )
        assert (
            last_message["content"] is not None and len(last_message["content"]) >= 1
        ), "Assistant message must have content"
        result = last_message["content"][0]["content"]
        if schema:
            # TODO: catch errors and prevent them from leaking sensitive information
            result_dict = json.loads(result)
            jsonschema.validate(result_dict, schema)
            result = result_dict

        # We know that at runtime, AD uses the tracelogger
        # and we have to override the delegate's messages
        # to manipulate the message logging.
        # TODO: This is very hacky. Find a better way. Maybe custom logging.
        logger = typing.cast("logging.TraceLogger", logging.Logger.get())

        # save old logger state
        old_pipeline_name = logger.context["pipeline_name"]
        old_messages = logger.delegate.messages

        logger.context["pipeline_name"] = "lamb_sub_llm"
        logger.delegate.messages = []
        logger.log(list(final_state.messages))

        # reset to old logger state
        logger.context["pipeline_name"] = old_pipeline_name
        logger.delegate.messages = old_messages

        return result

    return query
