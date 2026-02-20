from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
import agentdojo.types as ad_types
from openai.types.shared_params import ResponseFormatText

from lamb.agent import Agent

type CallLLM = Callable[[list[ad_types.ChatMessage]], ad_types.ChatMessage]
type CallTools = Callable[[list[rt.FunctionCall]], list[ad_types.ChatMessage]]
type CheckIFC = Callable[[rt.Function, Mapping[str, rt.FunctionCallArgTypes]], None]
type AgentFn = Callable[[rt.FunctionsRuntime, rt.TaskEnvironment], Agent]

TEXT_FORMAT: ResponseFormatText = {"type": "text"}


@runtime_checkable
class Formatter(Protocol):
    def format(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
    ) -> str: ...
    def expand(
        self,
        tool: rt.Function,
        args: Mapping[str, rt.FunctionCallArgTypes],
    ) -> Mapping[str, rt.FunctionCallArgTypes]: ...


class Identity(Enum):
    SINGLE = "single"
    PRIVILEDGED = "priviledged"
    QUARANTINED = "quarantined"
    BOUNDED = "bounded"


class Role(Enum):
    USER = "user"
    TOOL = "tool"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ADAgentLoop(pipeline.BasePipelineElement):
    """Wrap an agent as a BasePipelineElement."""

    agent_fn: AgentFn

    def __init__(self, agent_fn: AgentFn) -> None:
        self.agent_fn = agent_fn

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
        agent = self.agent_fn(runtime, env)
        messages = agent.prompt(query)
        return (
            query,
            runtime,
            env,
            messages,
            extra_args,
        )


def make_user_prompt(prompt: str) -> ad_types.ChatMessage:
    return {
        "role": "user",
        "content": [ad_types.text_content_block_from_string(prompt)],
    }
