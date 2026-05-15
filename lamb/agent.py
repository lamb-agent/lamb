import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
import openai
from agentdojo import types as ad
from openai.types.chat.completion_create_params import ResponseFormat

import lamb.controller
import lamb.formatter
import lamb.ifc
import lamb.labels
import lamb.llm
import lamb.prompts
import lamb.query_llm
import lamb.runtime
import lamb.tool_exec
import lamb.types


@dataclass
class AgentCore:
    """The stateful part of an agent.

    Should be created anew for each invocation.
    """

    runtime: lamb.runtime.Runtime
    ifc_checker: lamb.ifc.IFCChecker
    formatter: lamb.formatter.VariableFormatter

    def __init__(
        self,
        runtime: lamb.runtime.Runtime,
        ifc_checker: lamb.ifc.IFCChecker,
    ) -> None:
        self.runtime = runtime
        self.ifc_checker = ifc_checker
        self.formatter = lamb.formatter.VariableFormatter.ifc(ifc_checker)


class Agent:
    name: str
    make_core: Callable[[], AgentCore]
    system_prompt: str
    model: lamb.llm.Llm
    identity: lamb.types.Identity
    initial_label: lamb.types.IFCLabelStr
    max_iters: int

    def __init__(
        self,
        name: str,
        model: lamb.llm.Llm,
        identity: lamb.types.Identity,
        system_prompt: str,
        make_core: Callable[[], AgentCore],
        initial_label: lamb.types.IFCLabelStr,
        max_iters: int = 10,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.identity = identity
        self.make_core = make_core
        self.initial_label = initial_label
        self.max_iters = max_iters

    def assemble_controller(
        self,
        core: AgentCore,
        response_format: ResponseFormat,
    ) -> lamb.controller.Controller:
        tool_llm = lamb.llm.ToolLlm(llm=self.model, runtime=core.runtime)
        executor = lamb.tool_exec.ToolExec(core.runtime, core.formatter)

        def call_llm(messages: list[lamb.types.ChatMessage]) -> lamb.types.ChatMessage:
            try:
                response = tool_llm.prompt(messages, response_format)
                if isinstance(response, lamb.types.AssistantMessage):
                    response.label = str(core.ifc_checker.get_model_context())
                return response  # noqa: TRY300
            except openai.InternalServerError:
                return lamb.types.make_assistant_prompt(
                    "The model failed to provide a correctly formatted tool call."
                )

        def call_tools(
            calls: list[lamb.types.FunctionCall],
        ) -> list[lamb.types.ChatMessage]:
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
    ) -> tuple[list[lamb.types.ChatMessage], AgentCore]:
        """Prompt the agent with a user prompt.

        Returns the resulting chat history and the ejected core.
        """

        core = self.make_core()
        controller = self.assemble_controller(core, response_format)
        history = controller.loop(
            [
                lamb.types.make_system_prompt(self.system_prompt),
                lamb.types.make_user_prompt(user_prompt, self.initial_label),
            ]
        )
        return history, core

    def prompt(
        self,
        user_prompt: str,
        response_format: ResponseFormat = lamb.types.TEXT_FORMAT,
    ) -> tuple[list[lamb.types.ChatMessage], lamb.ifc.IFCLabel]:
        """Prompt the agent with a user prompt.

        Returns the resulting chat history and the ejected core.
        """

        history, core = self.run(user_prompt, response_format)
        response = lamb.types.get_response(history)
        label = core.ifc_checker.response_label(response)
        # expand all variables
        for var, val in core.formatter.var_vals.items():
            response = response.replace(var, val)

        last_message = history[-1]
        last_message.content = response
        return history, label

    @staticmethod
    def single(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        ifc_checker = lamb.ifc.IFCChecker.none(labeler)
        return Agent(
            name="single",
            model=model,
            identity=lamb.types.Identity.SINGLE,
            system_prompt=lamb.prompts.SINGLE_LLM_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime.default(functions_runtime, env),
                ifc_checker=ifc_checker,
            ),
            initial_label="TL",
        )

    @staticmethod
    def quarantined(
        model: lamb.llm.Llm,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        ifc_checker = lamb.ifc.IFCChecker.top(labeler)
        return Agent(
            name="dual-quarantined",
            model=model,
            identity=lamb.types.Identity.QUARANTINED,
            system_prompt=lamb.prompts.Q_LLM_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime.empty(),
                ifc_checker=ifc_checker,
            ),
            initial_label="UH",  # TODO: this should actually be passed as a parameter
            max_iters=2,  # we know it can't be more, because no tools are available
        )

    @staticmethod
    def dual(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        q_llm = Agent.quarantined(model, labeler)
        query_llm = lamb.query_llm.make_query_llm_fn_with_agent(q_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(
            q_llm
        )
        ifc_checker = lamb.ifc.IFCChecker.vars_only(labeler)
        return Agent(
            name="dual-privileged",
            model=model,
            identity=lamb.types.Identity.PRIVILEGED,
            system_prompt=lamb.prompts.P_LLM_NO_IFC_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime(
                    functions_runtime, env, [query_llm, query_llm_structured]
                ),
                ifc_checker=ifc_checker,
            ),
            initial_label="TL",
        )

    @staticmethod
    def dual_ifc_dynamic(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        from lamb.cores.dual_ifc import make_core  # noqa: PLC0415

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
                lamb.ifc.IFCPolicy.dynamic(),
            ),
            initial_label="TL",
        )

    @staticmethod
    def dual_ifc_static(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        from lamb.cores.dual_ifc import make_core  # noqa: PLC0415

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
                lamb.ifc.IFCPolicy.static(),
            ),
            initial_label="TL",
        )

    @staticmethod
    def bounded_high_make_core(
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> AgentCore:
        B_HIGH_FILTER = lamb.ifc.ALL_TOOLS - {  # noqa: N806
            lamb.ifc.SystemAccess.PRIVILEGED,
            lamb.ifc.Integrity.TRUSTED,
            lamb.ifc.Confidentiality.LOW,
        }
        tools = list(functions_runtime.functions.values())
        bounded_fns = labeler.filter_tools(tools, B_HIGH_FILTER)
        # no nested llms
        runtime = lamb.runtime.Runtime.default(rt.FunctionsRuntime(bounded_fns), env)

        ifc_checker = lamb.ifc.IFCChecker.top(labeler=labeler)
        return AgentCore(
            runtime=runtime,
            ifc_checker=ifc_checker,
        )

    @staticmethod
    def bounded(
        model: lamb.llm.Llm,
        system_prompt: str,
        initial_label: lamb.types.IFCLabelStr,
        make_core: Callable[[], AgentCore],
    ) -> "Agent":
        return Agent(
            name="lamb-bounded",
            model=model,
            identity=lamb.types.Identity.BOUNDED,
            system_prompt=system_prompt,
            make_core=make_core,
            initial_label=initial_label,
            max_iters=7,  # usually nothing useful comes of it, if it takes longer
        )

    @staticmethod
    def lamb_no_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        labeler: lamb.ifc.Labeler,
    ) -> "Agent":
        from lamb.cores.lamb_no_ifc import make_core  # noqa: PLC0415

        return Agent(
            name="lamb-no-ifc-privileged",
            model=model,
            identity=lamb.types.Identity.PRIVILEGED,
            system_prompt=lamb.prompts.P_LLM_NO_IFC_SYSTEM_PROMPT,
            make_core=lambda: make_core(model, functions_runtime, env, labeler),
            initial_label="TL",
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
            initial_label="TL",
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
            initial_label="TL",
        )


class AgentStr(Enum):
    SINGLE = "single"
    DUAL = "dual"
    LAMB_NO_IFC = "lamb-no-ifc"
    LAMB_DYNAMIC_IFC = "lamb-dynamic-ifc"
    LAMB_STATIC_IFC = "lamb-static-ifc"


@dataclass
class AgentFn[Env: rt.TaskEnvironment]:
    make: Callable[[lamb.types.Suite, rt.FunctionsRuntime, Env], Agent]

    @staticmethod
    def new(
        name: AgentStr,
        model: lamb.llm.Llm,
        labeler_fn: typing.Callable[[lamb.types.Suite, Env], lamb.ifc.Labeler],
    ) -> "AgentFn[Env]":
        match name:
            case AgentStr.SINGLE:
                return AgentFn(
                    lambda suite, runtime, env: Agent.single(
                        model,
                        runtime,
                        env,
                        labeler_fn(suite, env),
                    )
                )
            case AgentStr.DUAL:
                return AgentFn(
                    lambda suite, runtime, env: Agent.dual(
                        model,
                        runtime,
                        env,
                        labeler_fn(suite, env),
                    )
                )
            case AgentStr.LAMB_NO_IFC:
                return AgentFn(
                    lambda suite, runtime, env: Agent.lamb_no_ifc(
                        model,
                        runtime,
                        env,
                        labeler_fn(suite, env),
                    )
                )
            case AgentStr.LAMB_DYNAMIC_IFC:
                return AgentFn(
                    lambda suite, runtime, env: Agent.lamb_dynamic_ifc(
                        model,
                        runtime,
                        env,
                        labeler_fn(suite, env),
                    )
                )
            case AgentStr.LAMB_STATIC_IFC:
                return AgentFn(
                    lambda suite, runtime, env: Agent.lamb_static_ifc(
                        model,
                        runtime,
                        env,
                        labeler_fn(suite, env),
                    )
                )
            case _:
                typing.assert_never(name)


class ADAgentLoop(pipeline.BasePipelineElement):
    """Wrap an agent as a BasePipelineElement."""

    agent_fn: AgentFn
    suite: lamb.types.Suite

    def __init__(self, agent_fn: AgentFn, suite: lamb.types.Suite) -> None:
        self.agent_fn = agent_fn
        self.suite = suite

    def query(
        self,
        query: str,
        runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment = rt.EmptyEnv(),  # noqa: B008
        messages: Sequence[ad.ChatMessage] = [],
        extra_args: dict = {},  # we ignore extra_args # noqa: B006
    ) -> tuple[
        str,
        rt.FunctionsRuntime,
        rt.TaskEnvironment,
        Sequence[ad.ChatMessage],
        dict,
    ]:
        agent = self.agent_fn.make(self.suite, runtime, env)
        history, _ = agent.prompt(query)
        messages = [*messages, *(lamb.types.message_to_ad(msg) for msg in history)]
        return (
            query,
            runtime,
            env,
            messages,
            extra_args,
        )
