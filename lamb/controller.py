import sys
from dataclasses import dataclass

from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    TextContentBlock,
)

from lamb import logging, types


@dataclass
class Controller:
    call_llm: types.CallLLM
    call_tools: types.CallTools
    identity: types.Identity
    max_iters: int = 25

    def loop(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        # Prevent infinite loops
        if self.max_iters <= 0:
            failure_message = types.make_assistant_prompt(
                "Unfortunately, I failed to fulfill your request"
            )

            logging.log_message(self.identity, failure_message)
            return [*messages, failure_message]
        nr_messages = len(messages)
        assert nr_messages >= 2, (
            "There must be at least the system message and the initial prompt"
        )

        last_message = messages[-1]
        logging.log_message(self.identity, last_message)
        match last_message:
            # Initial user prompt
            case {"role": "user"}:
                next_message = self.call_llm(messages)
                return self.loop([*messages, next_message])
            # evaluate tool calls
            case {"role": "assistant", "tool_calls": calls} if (
                calls is not None and len(calls) > 0
            ):
                tool_messages = self.call_tools(calls)
                logging.log_messages(self.identity, tool_messages)
                appended_messages = [*messages, *tool_messages]
                next_message = self.call_llm(appended_messages)
                return self.loop([*appended_messages, next_message])
            # assistant finished without calling tools
            case {"role": "assistant"}:
                return messages
            case _:
                logging.exception(f"Illegal message role: {last_message['role']}")
                sys.exit(1)
