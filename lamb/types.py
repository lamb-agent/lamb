from collections.abc import Callable, Sequence
from dataclasses import dataclass

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
from agentdojo.types import ChatMessage

from lamb import tool_llm, tool_result

type Transition = Callable[[State], State]
"""Go from one state to the next."""
type Init = Callable[
    [rt.FunctionsRuntime, rt.TaskEnvironment, Sequence[ChatMessage]], State
]
"""Produce an initial state from the runtime, env and messages given by AgentDojo"""


@dataclass
class Config:
    llm: Transition
    tool_executor: Transition
    tool_result_formatter: tool_result.Formatter
    tool_llm: tool_llm.ToolLlm | None


@dataclass
class State:
    runtime: rt.FunctionsRuntime
    env: rt.TaskEnvironment
    messages: Sequence[ChatMessage]
    config: Config

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
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},  # we ignore extra_args # noqa: B006
    ) -> tuple[
        str, rt.FunctionsRuntime, rt.TaskEnvironment, Sequence[ChatMessage], dict
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
