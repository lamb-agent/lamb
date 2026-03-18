# Run with > uv run pytest
import typing

from agentdojo import functions_runtime as rt
from agentdojo.default_suites.v1.tools import email_client, slack
from agentdojo.task_suite import load_suites

from lamb import ifc, labels, runtime, types
from lamb.agent import Agent
from lamb.llm import MockLlm

if typing.TYPE_CHECKING:
    from agentdojo.default_suites.v1.slack.task_suite import SlackEnvironment
    from agentdojo.default_suites.v1.workspace.task_suite import WorkspaceEnvironment

EMPTY_RUNTIME = rt.FunctionsRuntime([])
EMPTY_ENV = rt.TaskEnvironment()


def slack_env() -> "SlackEnvironment":
    slack_suite = load_suites.get_suite("v1.2", "slack")
    return slack_suite.load_and_inject_default_environment({})


def workspace_env() -> "WorkspaceEnvironment":
    ws_suite = load_suites.get_suite("v1.2", "workspace")
    return ws_suite.load_and_inject_default_environment({})


def make_function(
    name: str,
    params: list[tuple[str, str]],
    fn: typing.Callable,
    dependencies: list[str] = [],  # noqa: B006
) -> rt.Function:
    return runtime.make_function(
        name, "", [(param, t, "") for param, t in params], fn, dependencies
    )


GET_CURRENT_DAY = make_function("get_current_day", [], lambda: "2026-16-03")

INVITE_USER_TO_SLACK = make_function(
    "invite_user_to_slack",
    [("user", "str"), ("user_email", "str")],
    slack.invite_user_to_slack,
    ["slack"],
)

SEND_EMAIL = make_function(
    "send_email",
    [
        ("recipients", "list"),
        ("subject", "str"),
        ("body", "str"),
        ("attachments", "list"),
        ("cc", "list"),
        ("bcc", "list"),
    ],
    email_client.send_email,
    ["inbox"],
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


def test_slack_invite() -> None:
    frt = rt.FunctionsRuntime([INVITE_USER_TO_SLACK])
    env = slack_env()
    fred = "Fred"
    call_0 = types.FunctionCall(
        INVITE_USER_TO_SLACK.name,
        {"user": fred, "user_email": "fred@gmail.com"},
        "0",
    )
    messages: list[types.ChatMessage] = [
        tool_call_msg(call_0),
        assistant_msg("content"),
    ]
    a = Agent.lamb_static_ifc(
        MockLlm(messages),
        frt,
        env,
        labels.ADLabeler(env),
    )
    _, _ = a.prompt("Invite Fred (fred@gmail.com) to slack")
    assert fred in env.slack.users


def test_argument_validation() -> None:
    frt = rt.FunctionsRuntime([SEND_EMAIL])
    env = workspace_env()
    call_0 = types.FunctionCall(
        SEND_EMAIL.name,
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
        frt,
        env,
        labels.ADLabeler(env),
    )
    history, _ = a.prompt("Send an email to paul@gmail.com with a bad attachment")
    assert isinstance(history[-2], types.ToolMessage)
    assert history[-2].error is not None
    assert "ValidationError" in history[-2].error
