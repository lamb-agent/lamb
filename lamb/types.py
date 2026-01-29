from collections.abc import Sequence
from dataclasses import astuple, dataclass
from typing import Callable, TypedDict, cast

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
from agentdojo.types import ChatMessage

from lamb import tool_result

type Transition = Callable[[State], State]


class Config(TypedDict):
    tool_result_formatter: tool_result.Formatter


@dataclass
class State[Env: rt.TaskEnvironment]:
    query: str
    runtime: rt.FunctionsRuntime
    env: Env
    messages: Sequence[ChatMessage]
    extra_args: Config


class PipeElementWrapper(pipeline.BasePipelineElement):
    """Transforms a state transition function into an AgentDojo compatible BasePipelineElement."""

    run: Transition
    config: Config

    def __init__(self, run: Transition, config: Config) -> None:
        self.run = run
        self.config = config

    def query(
        self,
        query: str,
        runtime: rt.FunctionsRuntime,
        env: rt.Env = rt.EmptyEnv(),  # ty:ignore[invalid-parameter-default]
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, rt.FunctionsRuntime, rt.Env, Sequence[ChatMessage], dict]:
        config: Config = cast(
            Config, self.config | extra_args
        )  # TODO: Figure out if there is a safe way to do this
        return astuple(self.run(State(query, runtime, env, messages, config)))
