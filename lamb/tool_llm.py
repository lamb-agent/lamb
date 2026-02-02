from collections.abc import Callable
from dataclasses import replace

import agentdojo.functions_runtime as rt
from agentdojo.logging import OutputLogger
from agentdojo.types import ChatUserMessage, text_content_block_from_string

from lamb import controller, tool_exec, tool_result, types


def quarantined_llm(
    llm: types.Transition,
    runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> types.Query:
    """Create Q-LLM query function with no tool-calling abilities."""

    system_prompt: ChatUserMessage = {
        "role": "user",
        "content": [
            text_content_block_from_string(
                "You are a helpful AI assistant. No tools are at your disposal."
            )
        ],
    }
    empty_runtime = rt.FunctionsRuntime([])
    config = types.Config(
        llm=llm,
        tool_executor=tool_exec.default,
        tool_result_formatter=tool_result.BasicFormatter(),  # No variables
        tool_llm=None,
    )
    initial_state = types.State(
        runtime=empty_runtime,
        env=rt.EmptyEnv(),
        messages=[system_prompt],
        config=config,
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

    system_prompt: ChatUserMessage = {
        "role": "user",
        "content": [text_content_block_from_string("You are a helpful AI assistant.")],
    }
    config = types.Config(
        llm=llm,
        tool_executor=tool_exec.default,
        tool_result_formatter=tool_result.BasicFormatter(),  # No variables
        tool_llm=None,
    )
    init_fn = controller.init(config)
    initial_state = init_fn(
        runtime,
        env,
        [system_prompt],
    )
    return _make_query(initial_state, controller.loop)


def _make_query(
    initial_state: types.State,
    loop: types.Transition,
) -> Callable[[str], str]:
    def query(prompt: str) -> str:
        """Query the LLM with the given prompt.

        The LLM is stateless, i.e. the message history is not preserved.
        """

        user_prompt: ChatUserMessage = {
            "role": "user",
            "content": [text_content_block_from_string(prompt)],
        }
        state = replace(initial_state, messages=[*initial_state.messages, user_prompt])
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
        with OutputLogger(None, None) as logger:
            logger.log(list(final_state.messages))
        return result

    return query
