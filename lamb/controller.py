import sys
from collections.abc import Sequence

import agentdojo.functions_runtime as rt
import agentdojo.logging as log
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    TextContentBlock,
)

from lamb import tools, types


def init(config: types.Config) -> types.Init:
    system_prompt = types.make_user_prompt(config.system_prompt)

    def _init(
        runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        messages: Sequence[ChatMessage],
    ) -> types.State:
        # TODO: is there a better way with proper typing to edit the env?
        if config.tool_llm:
            runtime.register_function(tools.query_llm)
            object.__setattr__(
                env,
                "tool_llm",
                config.tool_llm(config.llm, runtime, env),
            )
        return types.State(runtime, env, [system_prompt, *messages], config)

    return _init


def loop(
    state: types.State,
    max_iters: int = 20,
) -> types.State:
    # Prevent infinite loops
    if max_iters <= 0:
        ChatAssistantMessage(
            role="assistant",
            content=[
                TextContentBlock(
                    type="text",
                    content="Unfortunately, I failed to fulfill your request",
                )
            ],
            tool_calls=[],
        )
        return state
    nr_messages = len(state.messages)
    assert nr_messages >= 2, (
        "There must be at least the system message and the initial prompt"
    )

    last_message = state.messages[-1]
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
            next_state = tool_state.next_llm()
        # assistant finished without calling tools
        case {"role": "assistant"}:
            return state
        case _:
            logger = log.Logger.get()
            logger.log_error(f"Illegal message role: {last_message['role']}")
            sys.exit(1)
    assert len(next_state.messages) > nr_messages, "LLM must populate messages"
    return loop(next_state, max_iters=max_iters - 1)
