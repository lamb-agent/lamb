# Run with > uv run python -m pytest
import typing

from agentdojo import functions_runtime as rt

from lamb import ifc, runtime, types
from lamb.agent import Agent
from lamb.llm import MockLlm

EMPTY_RUNTIME = rt.FunctionsRuntime([])
EMPTY_ENV = rt.TaskEnvironment()


def make_function(
    name: str, params: list[tuple[str, str]], fn: typing.Callable
) -> rt.Function:
    return runtime.make_function(name, "", [(param, t, "") for param, t in params], fn)


GET_CURRENT_DAY = make_function("get_current_day", [], lambda: "2026-16-03")


def tool_call_msg(call: types.FunctionCall) -> types.AssistantMessage:
    return types.AssistantMessage(
        "",
        [call],
    )


def assistant_msg(msg: str) -> types.AssistantMessage:
    return types.AssistantMessage(
        msg,
        [],
    )


def test_single_empty() -> None:
    messages: list[types.ChatMessage] = [types.AssistantMessage("content", [])]
    a = Agent.single(MockLlm(messages), EMPTY_RUNTIME, EMPTY_ENV)
    history, label = a.prompt("")
    assert history[-1] == messages[-1]
    assert label == ifc.IFCLabel.UH


def test_single_tool_call() -> None:
    frt = rt.FunctionsRuntime([GET_CURRENT_DAY])
    call_0 = types.FunctionCall(GET_CURRENT_DAY.name, {}, "0")
    messages: list[types.ChatMessage] = [
        tool_call_msg(call_0),
        assistant_msg("content"),
    ]
    a = Agent.single(MockLlm(messages), frt, EMPTY_ENV)
    history, label = a.prompt("")
    assert history[-3] == messages[-2]
    assert history[-2] == types.ToolMessage(
        content=GET_CURRENT_DAY.run(),
        error=None,
        tool_call_id=call_0.id,
        tool_call=call_0,
    )
    assert history[-1] == messages[-1]
    assert label == ifc.IFCLabel.UH
