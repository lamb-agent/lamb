import sys
import typing
from collections.abc import Sequence
from dataclasses import replace

import agentdojo.functions_runtime as rt
from agentdojo.types import (
    ChatMessage,
    ChatSystemMessage,
    TextContentBlock,
)

from lamb import logging, tools, types


def init(config: types.Config) -> types.Init:
    system_prompt = types.make_user_prompt(config.system_prompt)

    def _init(
        runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        messages: Sequence[ChatMessage],
    ) -> types.State:
        assert len(messages) == 1, "Messages must only contain user message"
        assert messages[0]["role"] == "user", "Messages must only contain user message"
        user_message = messages[0]

        # TODO: is there a better way with proper typing to edit the env?
        if config.tool_llm:
            runtime.register_function(tools.query_llm)
            runtime.register_function(tools.query_llm_structured)
            object.__setattr__(
                env,
                "tool_llm",
                config.tool_llm(config.llm, config.read_only_tools, env),
            )
        return types.State(
            runtime,
            env,
            [system_prompt, user_message],
            config,
            types.TEXT_FORMAT,
        )

    return _init


def loop(
    state: types.State,
    max_iters: int = 20,
) -> types.State:
    # Prevent infinite loops
    if max_iters <= 0:
        failure_message = ChatSystemMessage(
            role="system",
            content=[
                TextContentBlock(
                    type="text",
                    content="Unfortunately, I failed to fulfill your request",
                )
            ],
        )
        logging.log_message(state.config.identity, failure_message)
        return replace(state, messages=[*state.messages, failure_message])
    nr_messages = len(state.messages)
    assert nr_messages >= 2, (
        "There must be at least the system message and the initial prompt"
    )

    last_message = state.messages[-1]
    logging.log_message(state.config.identity, last_message)
    next_state: types.State
    match last_message:
        # Initial user prompt
        case {"role": "user"}:
            next_state = state.next_llm()
        # evaluate tool calls
        case {"role": "assistant", "tool_calls": calls} if (
            calls is not None and len(calls) > 0
        ):
            tool_state = state.next_tool()
            logging.log_messages(
                state.config.identity, tool_state.messages[nr_messages:]
            )
            next_state = tool_state.next_llm()
        # assistant finished without calling tools
        case {"role": "assistant"}:
            # clean up env again
            # this is required for passing the utility checks
            if state.config.tool_llm:
                object.__delattr__(state.env, "tool_llm")
            return state
        case _:
            logging.exception(f"Illegal message role: {last_message['role']}")
            sys.exit(1)
    assert len(next_state.messages) > nr_messages, "LLM must populate messages"
    return loop(next_state, max_iters=max_iters - 1)
