from collections.abc import Callable, Sequence
from dataclasses import dataclass

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
import openai
from agentdojo.types import ChatMessage
from openai.types.chat.completion_create_params import ResponseFormat

import lamb.controller
import lamb.formatter
import lamb.ifc
import lamb.llm
import lamb.prompts
import lamb.query_llm
import lamb.runtime
import lamb.tool_categories
import lamb.tool_exec
import lamb.types


@dataclass
class AgentCore:
    """The stateful part of an agent.

    Should be created anew for each invocation.
    """

    runtime: lamb.runtime.Runtime
    formatter: lamb.formatter.VariableFormatter
    ifc_checker: lamb.ifc.IFCChecker | None = None


class Agent:
    name: str
    make_core: Callable[[], AgentCore]
    system_prompt: str
    model: lamb.llm.Llm
    identity: lamb.types.Identity
    max_iters: int = 25

    def __init__(
        self,
        name: str,
        model: lamb.llm.Llm,
        identity: lamb.types.Identity,
        system_prompt: str,
        make_core: Callable[[], AgentCore],
        max_iters: int = 25,
    ) -> None:
        self.name = name
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
            try:
                return tool_llm.prompt(messages, response_format)
            except openai.InternalServerError:
                return lamb.types.make_assistant_prompt(
                    "The model failed to provide a correctly formatted tool call."
                )

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
                lamb.types.make_system_prompt(self.system_prompt),
                lamb.types.make_user_prompt(user_prompt),
            ]
        )
        return history, core

    def prompt(
        self,
        user_prompt: str,
        response_format: ResponseFormat = lamb.types.TEXT_FORMAT,
    ) -> tuple[list[ChatMessage], lamb.ifc.IFCLabel]:
        """Prompt the agent with a user prompt.

        Returns the resulting chat history and the ejected core.
        """

        history, core = self.run(user_prompt, response_format)
        response = lamb.types.get_response(history)
        label = lamb.ifc.IFCLabel.top()  # default to TOP
        if core.ifc_checker:
            label = core.ifc_checker.response_label(response)
        # expand all variables
        for var, val in core.formatter.var_vals.items():
            response = response.replace(var, val)

        last_message = history[-1]
        assert last_message["content"] is not None
        last_message["content"][0]["content"] = response
        return history, label

    @staticmethod
    def single(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Agent":
        return Agent(
            name="single",
            model=model,
            identity=lamb.types.Identity.SINGLE,
            system_prompt=lamb.prompts.SINGLE_LLM_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime.default(functions_runtime, env),
                formatter=lamb.formatter.VariableFormatter.none(),
            ),
        )

    @staticmethod
    def quarantined(
        model: lamb.llm.Llm,
    ) -> "Agent":
        return Agent(
            name="dual-quarantined",
            model=model,
            identity=lamb.types.Identity.QUARANTINED,
            system_prompt=lamb.prompts.Q_LLM_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime.empty(),
                formatter=lamb.formatter.VariableFormatter.none(),
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
            name="dual-privileged",
            model=model,
            identity=lamb.types.Identity.PRIVILEGED,
            system_prompt=lamb.prompts.P_LLM_NO_IFC_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime(
                    functions_runtime, env, [query_llm, query_llm_structured]
                ),
                formatter=lamb.formatter.VariableFormatter.integrity_based(
                    query_llm.run
                ),
            ),
        )

    @staticmethod
    def dual_ifc_dynamic(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        from cores.dual_ifc import make_core  # noqa: PLC0415

        return Agent(
            name="dual-ifc-privileged",
            model=model,
            identity=lamb.types.Identity.PRIVILEGED,
            system_prompt=lamb.prompts.P_LLM_IFC_SYSTEM_PROMPT,
            make_core=lambda: make_core(
                model,
                functions_runtime,
                env,
                labeler,
                lamb.ifc.SecretHandling.DYNAMIC,
            ),
        )

    @staticmethod
    def dual_ifc_static(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        from cores.dual_ifc import make_core  # noqa: PLC0415

        return Agent(
            name="dual-ifc-privileged",
            model=model,
            identity=lamb.types.Identity.PRIVILEGED,
            system_prompt=lamb.prompts.P_LLM_IFC_SYSTEM_PROMPT,
            make_core=lambda: make_core(
                model,
                functions_runtime,
                env,
                labeler,
                lamb.ifc.SecretHandling.STATIC,
            ),
        )

    @staticmethod
    def bounded_high_make_core(
        functions_runtime: rt.FunctionsRuntime, env: rt.TaskEnvironment
    ) -> AgentCore:
        all_fns = list(functions_runtime.functions.values())
        read_only_fns = filter_tools(
            all_fns,
            lamb.tool_categories.READ_ONLY
            & lamb.tool_categories.UNTRUSTED_SINK
            & lamb.tool_categories.HIGH_CONF_SINK,
        )
        # no nested llms
        runtime = lamb.runtime.Runtime.default(rt.FunctionsRuntime(read_only_fns), env)

        return AgentCore(
            runtime=runtime,
            # we are already at TOP
            formatter=lamb.formatter.VariableFormatter.none(),
        )

    @staticmethod
    def bounded(
        model: lamb.llm.Llm,
        system_prompt: str,
        make_core: Callable[[], AgentCore],
    ) -> "Agent":
        return Agent(
            name="lamb-bounded",
            model=model,
            identity=lamb.types.Identity.BOUNDED,
            system_prompt=system_prompt,
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
            name="lamb-no-ifc-privileged",
            model=model,
            identity=lamb.types.Identity.PRIVILEGED,
            system_prompt=lamb.prompts.P_LLM_NO_IFC_SYSTEM_PROMPT,
            make_core=lambda: make_core(model, functions_runtime, env),
        )

    @staticmethod
    def lamb_dynamic_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        from lamb.cores.lamb_dynamic_ifc import make_core  # noqa: PLC0415

        return Agent(
            name="lamb-dynamic-ifc-privileged",
            model=model,
            identity=lamb.types.Identity.PRIVILEGED,
            system_prompt=lamb.prompts.P_LLM_IFC_SYSTEM_PROMPT,
            make_core=lambda: make_core(
                model,
                functions_runtime,
                env,
                labeler,
            ),
        )

    @staticmethod
    def lamb_static_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        from lamb.cores.lamb_static_ifc import make_core  # noqa: PLC0415

        return Agent(
            name="lamb-static-ifc-privileged",
            model=model,
            identity=lamb.types.Identity.PRIVILEGED,
            system_prompt=lamb.prompts.P_LLM_IFC_SYSTEM_PROMPT,
            make_core=lambda: make_core(
                model,
                functions_runtime,
                env,
                labeler,
            ),
        )


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
        history, _ = agent.prompt(query)
        messages = [*messages, *history]
        return (
            query,
            runtime,
            env,
            messages,
            extra_args,
        )
