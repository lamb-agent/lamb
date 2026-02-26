from collections.abc import Callable, Sequence
from dataclasses import dataclass

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
from agentdojo.types import ChatAssistantMessage, ChatMessage
from openai.types.chat.completion_create_params import ResponseFormat

import lamb.controller
import lamb.ifc
import lamb.llm
import lamb.prompts
import lamb.query_llm
import lamb.runtime
import lamb.tool_categories
import lamb.tool_exec
import lamb.tool_result
import lamb.types


@dataclass
class AgentCore:
    """The stateful part of an agent.

    Should be created anew for each invokation.
    """

    runtime: lamb.runtime.Runtime
    formatter: lamb.tool_result.VariableFormatter
    ifc_checker: lamb.ifc.IFCChecker | None = None


class Agent:
    make_core: Callable[[], AgentCore]
    system_prompt: str
    model: lamb.llm.Llm
    identity: lamb.types.Identity
    max_iters: int = 25

    def __init__(
        self,
        model: lamb.llm.Llm,
        identity: lamb.types.Identity,
        system_prompt: str,
        make_core: Callable[[], AgentCore],
        max_iters: int = 25,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model
        self.identity = identity
        self.make_core = make_core
        self.max_iters = max_iters

    def assemble_controller(
        self,
        core: AgentCore,
        response_format: ResponseFormat,
    ) -> lamb.controller.Controller:
        tool_llm = lamb.llm.ToolLlm(llm=self.model, runtime=core.runtime)
        executor = lamb.tool_exec.ToolExec(core.runtime, core.formatter)

        def call_llm(messages: list[ChatMessage]) -> ChatMessage:
            return tool_llm.prompt(messages, response_format)

        def call_tools(calls: list[rt.FunctionCall]) -> list[ChatMessage]:
            return executor.exec(calls)

        return lamb.controller.Controller(
            call_llm,
            call_tools,
            self.identity,
            self.max_iters,
        )

    def run(
        self,
        user_prompt: str,
        response_format: ResponseFormat = lamb.types.TEXT_FORMAT,
    ) -> tuple[list[ChatMessage], AgentCore]:
        """Prompt the agent with a user prompt.

        Returns the resulting chat history and the ejected core.
        """

        core = self.make_core()
        controller = self.assemble_controller(core, response_format)
        history = controller.loop(
            [
                lamb.types.make_user_prompt(self.system_prompt),
                lamb.types.make_user_prompt(user_prompt),
            ]
        )
        return history, core

    def prompt(
        self,
        user_prompt: str,
        response_format: ResponseFormat = lamb.types.TEXT_FORMAT,
    ) -> tuple[str, lamb.ifc.IFCLabel | None]:
        """Prompt the agent with a user prompt.

        Returns the resulting chat history and the ejected core.
        """

        history, core = self.run(user_prompt, response_format)
        response = lamb.types.get_response(history)
        label = None
        if core.ifc_checker:
            label = core.ifc_checker.response_label(response)
        # expand all variables
        for var, val in core.formatter.var_vals.items():
            response = response.replace(var, val)
        return response, label

    @staticmethod
    def single(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        formatter: lamb.tool_result.VariableFormatter,
    ) -> "Agent":
        return Agent(
            model=model,
            identity=lamb.types.Identity.SINGLE,
            system_prompt=lamb.prompts.Q_LLM_SYSTEM_PROMPT,  # TODO: single prompt
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime.default(functions_runtime, env),
                formatter=formatter,
            ),
        )

    @staticmethod
    def quarantined(
        model: lamb.llm.Llm,
    ) -> "Agent":
        return Agent(
            model=model,
            identity=lamb.types.Identity.QUARANTINED,
            system_prompt=lamb.prompts.Q_LLM_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime.empty(),
                formatter=lamb.tool_result.VariableFormatter.none(),
            ),
        )

    @staticmethod
    def dual(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Agent":
        q_llm = Agent.quarantined(model)
        query_llm = lamb.query_llm.make_query_llm_fn_with_agent(q_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(
            q_llm
        )

        return Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,  # TODO: need dedicated
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime(
                    functions_runtime, env, [query_llm, query_llm_structured]
                ),
                formatter=lamb.tool_result.VariableFormatter.integrity_based(
                    query_llm.run
                ),
            ),
        )

    @staticmethod
    def dual_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        ifc_checker: lamb.ifc.IFCChecker,
    ) -> "Agent":
        q_llm = Agent.quarantined(model)
        custom_functions = [
            lamb.query_llm.make_query_llm_fn_with_agent(q_llm),
            lamb.query_llm.make_query_llm_structured_fn_with_agent(q_llm),
        ]

        return Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,  # TODO: need dedicated
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime(functions_runtime, env, custom_functions),
                formatter=lamb.tool_result.VariableFormatter.ifc(ifc_checker),
            ),
        )

    @staticmethod
    def bounded(
        model: lamb.llm.Llm,
        make_core: Callable[[], AgentCore],
    ) -> "Agent":
        return Agent(
            model=model,
            identity=lamb.types.Identity.BOUNDED,
            system_prompt=lamb.prompts.B_LLM_SYSTEM_PROMPT,
            make_core=make_core,
        )

    @staticmethod
    def lamb_no_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Agent":
        from lamb.cores.lamb_no_ifc import make_core  # noqa: PLC0415

        return Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            make_core=lambda: make_core(model, functions_runtime, env),
        )

    @staticmethod
    def lamb_dynamic_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Agent":
        from lamb.cores.lamb_dynamic_ifc import make_core  # noqa: PLC0415

        return Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            make_core=lambda: make_core(model, functions_runtime, env),
        )

    @staticmethod
    def lamb_static_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Agent":
        from lamb.cores.lamb_static_ifc import make_core  # noqa: PLC0415

        return Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            make_core=lambda: make_core(model, functions_runtime, env),
        )


def basic_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
    integ = (
        lamb.ifc.Integrity.TRUSTED
        if tool in lamb.tool_categories.TRUSTED_SOURCE
        else lamb.ifc.Integrity.UNTRUSTED
    )
    conf = (
        lamb.ifc.Confidentiality.HIGH
        if tool in lamb.tool_categories.HIGH_CONF_SOURCE
        else lamb.ifc.Confidentiality.LOW
    )
    return lamb.ifc.IFCLabel(integ, conf)


def basic_tool_sink_label(tool: Callable) -> lamb.ifc.IFCLabel:
    integ = (
        lamb.ifc.Integrity.TRUSTED
        if tool in lamb.tool_categories.TRUSTED_SINK
        else lamb.ifc.Integrity.UNTRUSTED
    )
    conf = (
        lamb.ifc.Confidentiality.HIGH
        if tool in lamb.tool_categories.HIGH_CONF_SINK
        else lamb.ifc.Confidentiality.LOW
    )
    return lamb.ifc.IFCLabel(integ, conf)


def filter_tools(
    all_fns: list[rt.Function],
    tools_set: set[Callable],
) -> list[rt.Function]:
    return [fn for fn in all_fns if fn.run in tools_set]


type AgentFn = Callable[[rt.FunctionsRuntime, rt.TaskEnvironment], Agent]


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
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},  # we ignore extra_args # noqa: B006
    ) -> tuple[
        str,
        rt.FunctionsRuntime,
        rt.TaskEnvironment,
        Sequence[ChatMessage],
        dict,
    ]:
        agent = self.agent_fn(runtime, env)
        response, _ = agent.prompt(query)
        messages = [*messages, lamb.types.make_assistant_prompt(response)]
        return (
            query,
            runtime,
            env,
            messages,
            extra_args,
        )
