from collections.abc import Callable, Sequence
from dataclasses import dataclass

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
import agentdojo.types as ad_types
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params import ResponseFormatJSONSchema, ResponseFormatText

from lamb import tool_result

type Transition = Callable[[State], State]
"""Go from one state to the next."""
type Init = Callable[
    [rt.FunctionsRuntime, rt.TaskEnvironment, Sequence[ad_types.ChatMessage]], State
]
"""Produce an initial state from the runtime, env and messages given by AgentDojo"""
type StructuredQuery = Callable[[str, dict], dict]
type TextQuery = Callable[[str], str]
type Query = TextQuery | StructuredQuery
"""Query an LLM or Agent with a prompt and get a response"""
type ToolLLM = Callable[[Transition, rt.FunctionsRuntime, rt.TaskEnvironment], Query]

TEXT_FORMAT: ResponseFormatText = {"type": "text"}


@dataclass
class Config:
    system_prompt: str
    llm: Transition
    tool_executor: Transition
    tool_result_formatter: tool_result.Formatter
    tool_llm: ToolLLM | None


@dataclass
class State:
    runtime: rt.FunctionsRuntime
    env: rt.TaskEnvironment
    messages: Sequence[ad_types.ChatMessage]
    config: Config
    response_format: ResponseFormat

    def next_llm(self) -> "State":
        return self.config.llm(self)

    def next_tool(self) -> "State":
        return self.config.tool_executor(self)


class PipeElementWrapper(pipeline.BasePipelineElement):
    """Transforms a transition function into a BasePipelineElement."""

    init: Init
    loop: Transition

    def __init__(self, init: Init, loop: Transition) -> None:
        self.init = init
        self.loop = loop

    def query(
        self,
        query: str,
        runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment = rt.EmptyEnv(),  # noqa: B008
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},  # we ignore extra_args # noqa: B006
    ) -> tuple[
        str,
        rt.FunctionsRuntime,
        rt.TaskEnvironment,
        Sequence[ad_types.ChatMessage],
        dict,
    ]:
        initial_state = self.init(runtime, env, messages)
        final_state = self.loop(initial_state)
        return (
            query,
            final_state.runtime,
            final_state.env,
            final_state.messages,
            extra_args,
        )


def make_user_prompt(prompt: str) -> ad_types.ChatMessage:
    return {
        "role": "user",
        "content": [ad_types.text_content_block_from_string(prompt)],
    }
