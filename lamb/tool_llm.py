import typing
from dataclasses import replace

import agentdojo.functions_runtime as rt
from agentdojo.logging import OutputLogger
from agentdojo.types import ChatUserMessage, text_content_block_from_string

from lamb import tool_exec, tool_result, types


class QuarantinedLlm:
    initial_state: types.State
    loop: types.Transition

    SYSTEM_PROMPT: typing.Final[ChatUserMessage] = {
        "role": "user",
        "content": [
            text_content_block_from_string(
                "You are a helpful AI assistant. No tools are at your disposal."
            )
        ],
    }

    def __init__(self, llm: types.Transition) -> None:
        empty_runtime = rt.FunctionsRuntime([])
        config = types.Config(
            llm=llm,
            tool_executor=tool_exec.default,
            tool_result_formatter=tool_result.BasicFormatter(),  # No variables
            tool_llm=None,
        )
        self.initial_state = types.State(
            runtime=empty_runtime,
            env=rt.EmptyEnv(),
            messages=[self.SYSTEM_PROMPT],
            config=config,
        )
        self.loop = lambda state: state.next_llm()  # Only call the llm, no agent loop

    def query(self, prompt: str) -> str:
        """Query the LLM with the given prompt.

        The LLM is stateless, i.e. the message history is not preserved.
        """

        user_prompt: ChatUserMessage = {
            "role": "user",
            "content": [text_content_block_from_string(prompt)],
        }
        state = replace(
            self.initial_state, messages=[*self.initial_state.messages, user_prompt]
        )
        final_state = self.loop(state)
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
