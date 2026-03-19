# Run with > uv run pytest
import typing

from agentdojo import functions_runtime as rt

from agentdojo.task_suite import load_suites

from lamb import ifc, labels, runtime, types
from lamb.agent import Agent
from lamb.llm import MockLlm

if typing.TYPE_CHECKING:
    from agentdojo.default_suites.v1.slack.task_suite import SlackEnvironment
    from agentdojo.default_suites.v1.workspace.task_suite import WorkspaceEnvironment


def empty_rt() -> tuple[rt.FunctionsRuntime, rt.TaskEnvironment]:
    return rt.FunctionsRuntime([]), rt.TaskEnvironment()


def slack_rt() -> tuple[rt.FunctionsRuntime, "SlackEnvironment"]:
    slack_suite = load_suites.get_suite("v1.2", "slack")
    return rt.FunctionsRuntime(
        slack_suite.tools
    ), slack_suite.load_and_inject_default_environment({})


def workspace_rt() -> tuple[rt.FunctionsRuntime, "WorkspaceEnvironment"]:
    ws_suite = load_suites.get_suite("v1.2", "workspace")
    return rt.FunctionsRuntime(
        ws_suite.tools
    ), ws_suite.load_and_inject_default_environment({})


def make_function(
    name: str,
    params: list[tuple[str, str]],
    fn: typing.Callable,
    dependencies: list[str] = [],  # noqa: B006
) -> rt.Function:
    return runtime.make_function(
        name, "", [(param, t, "") for param, t in params], fn, dependencies
    )


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
    runtime, env = empty_rt()
    a = Agent.single(MockLlm(messages), runtime, env)
    history, label = a.prompt("")
    assert history[-1] == messages[-1]
    assert label == ifc.IFCLabel.UH


def test_single_tool_call() -> None:
    runtime, env = workspace_rt()
    call_0 = types.FunctionCall("get_current_day", {}, "0")
    messages: list[types.ChatMessage] = [
        tool_call_msg(call_0),
        assistant_msg("content"),
    ]
    a = Agent.single(MockLlm(messages), runtime, env)
    history, label = a.prompt("")
    assert history[-3] == messages[-2]
    assert history[-2] == types.ToolMessage(
        content="2024-05-15",
        error=None,
        tool_call_id=call_0.id,
        tool_call=call_0,
    )
    assert history[-1] == messages[-1]
    assert label == ifc.IFCLabel.UH


def test_slack_invite() -> None:
    runtime, env = slack_rt()
    fred = "Fred"
    call_0 = types.FunctionCall(
        "invite_user_to_slack",
        {"user": fred, "user_email": "fred@gmail.com"},
        "0",
    )
    messages: list[types.ChatMessage] = [
        tool_call_msg(call_0),
        assistant_msg("content"),
    ]
    a = Agent.lamb_static_ifc(
        MockLlm(messages),
        runtime,
        env,
        labels.ADLabeler(env),
    )
    _, _ = a.prompt("Invite Fred (fred@gmail.com) to slack")
    assert fred in env.slack.users


def test_argument_validation() -> None:
    runtime, env = workspace_rt()
    call_0 = types.FunctionCall(
        "send_email",
        {
            "recipients": ["paul@gmail.com"],
            "subject": "test",
            "body": "test",
            "attachments": "BAD TYPE",
            "cc": [],
            "bcc": [],
        },
        "0",
    )
    messages: list[types.ChatMessage] = [
        tool_call_msg(call_0),
        assistant_msg("content"),
    ]
    a = Agent.lamb_static_ifc(
        MockLlm(messages),
        runtime,
        env,
        labels.ADLabeler(env),
    )
    history, _ = a.prompt("Send an email to paul@gmail.com with a bad attachment")
    assert isinstance(history[-2], types.ToolMessage)
    assert history[-2].error is not None
    assert "ValidationError" in history[-2].error
