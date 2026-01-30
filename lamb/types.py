from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
from agentdojo.types import ChatMessage

from lamb import tool_result

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
    """Transforms a state transition function into an AgentDojo compatible BasePipelineElement."""

    init: Init
    run: Transition

    def __init__(self, init: Init, run: Transition) -> None:
        self.init = init
        self.run = run

    def query(
        self,
        query: str,
        runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment = rt.EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},  # we ignore extra_args
    ) -> tuple[
        str, rt.FunctionsRuntime, rt.TaskEnvironment, Sequence[ChatMessage], dict
    ]:
        initial_state = self.init(runtime, env, messages)
        final_state = self.run(initial_state)
        return (
            query,
            final_state.runtime,
            final_state.env,
            final_state.messages,
            extra_args,
        )
