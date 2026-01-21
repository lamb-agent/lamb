from collections.abc import Sequence
from dataclasses import astuple, dataclass
from typing import Callable

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
from agentdojo.types import ChatMessage

type Llm = Callable[[State], State]


@dataclass
class State[Env: rt.TaskEnvironment]:
    query: str
    runtime: rt.FunctionsRuntime
    env: Env
    messages: Sequence[ChatMessage]
    extra_args: dict


class PipeElementWrapper(pipeline.BasePipelineElement):
    run: Callable[[State], State]

    def __init__(self, run: Callable[[State], State]) -> None:
        self.run = run

    def query(
        self,
        query: str,
        runtime: rt.FunctionsRuntime,
        env: rt.Env = rt.EmptyEnv(),  # ty:ignore[invalid-parameter-default]
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, rt.FunctionsRuntime, rt.Env, Sequence[ChatMessage], dict]:
        return astuple(self.run(State(query, runtime, env, messages, extra_args)))
