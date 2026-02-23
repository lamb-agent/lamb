from collections.abc import Callable
from dataclasses import dataclass

import agentdojo.functions_runtime as rt
from agentdojo.types import ChatMessage
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
    formatter: lamb.types.Formatter


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

    def prompt(
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

    @staticmethod
    def single(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        formatter: lamb.types.Formatter,
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
        read_only_fns = [
            fn
            for fn in functions_runtime.functions.values()
            if fn.run in lamb.tool_categories.READ_ONLY
        ]
        b_llm = Agent.bounded(
            model,
            # no nested llms
            make_core=lambda: AgentCore(
                runtime=lamb.runtime.Runtime.default(
                    rt.FunctionsRuntime(read_only_fns), env
                ),
                formatter=lamb.tool_result.VariableFormatter.integrity_based(),
            ),
        )
        query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(
            b_llm
        )

        return Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
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
    def lamb_dynamic_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Agent":
        all_fns = functions_runtime.functions.values()
        read_only_fns = [
            fn
            for fn in all_fns
            if fn.run in lamb.tool_categories.READ_ONLY
            and fn.run in lamb.tool_categories.UNTRUSTED_SINK
        ]

        def b_make_core() -> AgentCore:
            # no nested llms
            b_runtime = lamb.runtime.Runtime.default(
                rt.FunctionsRuntime(read_only_fns), env
            )

            def b_on_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
                """Restrict B-LLM tools to read-only high conf sinks."""

                assert ifc_label == lamb.ifc.IFCLabel.UH, "B-LLM label must be UH."
                high_conf_sink = [
                    fn
                    for fn in read_only_fns
                    if fn.run in lamb.tool_categories.HIGH_CONF_SINK
                ]
                b_runtime.set_functions(high_conf_sink)

            b_ifc_checker = lamb.ifc.IFCChecker(
                model_context=lamb.ifc.IFCLabel.UL,
                secret_handling=lamb.ifc.SecretHandling.DYNAMIC,
                tool_source_label=basic_tool_source_label,
                tool_sink_label=basic_tool_sink_label,
                on_model_context_change=b_on_context_change,
            )
            return AgentCore(
                runtime=b_runtime,
                formatter=lamb.tool_result.VariableFormatter.ifc(b_ifc_checker),
            )

        b_llm = Agent.bounded(model, b_make_core)
        query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(
            b_llm
        )

        def p_make_core() -> AgentCore:
            runtime = lamb.runtime.Runtime(
                functions_runtime, env, [query_llm, query_llm_structured]
            )

            def p_on_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
                """Restrict P-LLM tools to high conf sinks."""

                assert ifc_label == lamb.ifc.IFCLabel.TH, "P-LLM label must be TH."
                high_conf_sink_fns = [
                    fn
                    for fn in all_fns
                    if fn.run in lamb.tool_categories.HIGH_CONF_SINK
                ]
                read_only_high_conf_fns = [
                    fn
                    for fn in high_conf_sink_fns
                    if fn.run in lamb.tool_categories.READ_ONLY
                ]

                # TODO: fix new b_runtime creation
                # b_runtime.set_functions(read_only_high_conf_fns)
                # b_llm = Agent.bounded(
                #     model,
                #     b_runtime,
                #     # at the top of the lattice there is no point in hiding variables
                #     lamb.tool_result.VariableFormatter.none(),
                # )
                custom_functions = [
                    lamb.query_llm.make_query_llm_fn_with_agent(b_llm),
                    lamb.query_llm.make_query_llm_structured_fn_with_agent(b_llm),
                ]
                high_conf_sink_fns.extend(custom_functions)

                runtime.set_functions(high_conf_sink_fns)

            def p_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
                integ = (
                    lamb.ifc.Integrity.TRUSTED
                    if tool
                    in (
                        lamb.tool_categories.TRUSTED_SOURCE | {query_llm_structured.run}
                    )
                    else lamb.ifc.Integrity.UNTRUSTED
                )
                conf = (
                    lamb.ifc.Confidentiality.HIGH
                    if tool
                    in lamb.tool_categories.HIGH_CONF_SOURCE
                    # TODO: this is conservative for b_llm as high_conf_source
                    | {query_llm.run, query_llm_structured.run}
                    # TODO: ARG_CONF
                    else lamb.ifc.Confidentiality.LOW
                )
                return lamb.ifc.IFCLabel(integ, conf)

            def p_tool_sink_label(tool: Callable) -> lamb.ifc.IFCLabel:
                integ = (
                    lamb.ifc.Integrity.TRUSTED
                    if tool in lamb.tool_categories.TRUSTED_SINK
                    else lamb.ifc.Integrity.UNTRUSTED
                )
                conf = (
                    lamb.ifc.Confidentiality.HIGH
                    if tool in lamb.tool_categories.HIGH_CONF_SINK
                    # TODO: conservative estimate, b_llm is low conf sink
                    # TODO: ARG_CONF
                    else lamb.ifc.Confidentiality.LOW
                )
                return lamb.ifc.IFCLabel(integ, conf)

            p_ifc_checker = lamb.ifc.IFCChecker(
                model_context=lamb.ifc.IFCLabel.TL,
                tool_sink_label=p_tool_sink_label,
                tool_source_label=p_tool_source_label,
                secret_handling=lamb.ifc.SecretHandling.DYNAMIC,
                on_model_context_change=p_on_context_change,
            )
            return AgentCore(
                runtime=runtime,
                formatter=lamb.tool_result.VariableFormatter.ifc(p_ifc_checker),
            )

        return Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            make_core=p_make_core,
        )

    @staticmethod
    def lamb_static_ifc(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Agent":
        all_fns = list(functions_runtime.functions.values())

        def filter_tools(
            all_fns: list[rt.Function],
            tools_set: set[Callable],
        ) -> list[rt.Function]:
            return [fn for fn in all_fns if fn.run in tools_set]

        # no nested llms
        b_high_runtime = lamb.runtime.Runtime.default(
            functions_runtime=rt.FunctionsRuntime(
                filter_tools(
                    all_fns,
                    lamb.tool_categories.READ_ONLY
                    | lamb.tool_categories.UNTRUSTED_SINK
                    | lamb.tool_categories.HIGH_CONF_SINK,
                )
            ),
            env=env,
        )
        b_high_ifc_checker = lamb.ifc.IFCChecker(
            model_context=lamb.ifc.IFCLabel.UH,
            tool_sink_label=basic_tool_sink_label,
            tool_source_label=basic_tool_source_label,
            secret_handling=lamb.ifc.SecretHandling.STATIC,
        )
        b_high_llm = Agent.bounded(
            model,
            make_core=lambda: AgentCore(
                runtime=b_high_runtime,
                formatter=lamb.tool_result.VariableFormatter.ifc(b_high_ifc_checker),
            ),
        )
        b_low_query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_high_llm)
        b_low_runtime = lamb.runtime.Runtime(
            functions_runtime=rt.FunctionsRuntime(
                filter_tools(
                    all_fns,
                    lamb.tool_categories.READ_ONLY
                    | lamb.tool_categories.UNTRUSTED_SINK,
                )
            ),
            env=env,
            custom_functions=[b_low_query_llm],
        )

        def b_low_tool_sink_label(tool: Callable) -> lamb.ifc.IFCLabel:
            if tool == b_low_query_llm:
                return lamb.ifc.IFCLabel.UH
            return basic_tool_sink_label(tool)

        def b_low_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
            if tool == b_low_query_llm:
                return lamb.ifc.IFCLabel.UH
            return basic_tool_sink_label(tool)

        b_low_ifc_checker = lamb.ifc.IFCChecker(
            model_context=lamb.ifc.IFCLabel.UL,
            tool_sink_label=b_low_tool_sink_label,
            tool_source_label=b_low_tool_source_label,
            secret_handling=lamb.ifc.SecretHandling.STATIC,
        )
        b_low_llm = Agent.bounded(
            model,
            make_core=lambda: AgentCore(
                runtime=b_low_runtime,
                formatter=lamb.tool_result.VariableFormatter.ifc(b_low_ifc_checker),
            ),
        )

        p_high_query_llm = b_low_query_llm  # uses b_high_llm agent
        p_high_query_llm_structured = (
            lamb.query_llm.make_query_llm_structured_fn_with_agent(b_high_llm)
        )
        p_high_runtime = lamb.runtime.Runtime(
            functions_runtime=rt.FunctionsRuntime(
                filter_tools(
                    all_fns,
                    lamb.tool_categories.HIGH_CONF_SINK,
                )
            ),
            env=env,
            custom_functions=[p_high_query_llm, p_high_query_llm_structured],
        )

        def p_high_tool_sink_label(tool: Callable) -> lamb.ifc.IFCLabel:
            if tool == p_high_query_llm:
                return lamb.ifc.IFCLabel.UH
            if tool == p_high_query_llm_structured:
                return lamb.ifc.IFCLabel.UH
            return basic_tool_sink_label(tool)

        def p_high_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
            if tool == p_high_query_llm:
                return lamb.ifc.IFCLabel.UH
            if tool == p_high_query_llm_structured:
                return lamb.ifc.IFCLabel.TH
            return basic_tool_sink_label(tool)

        p_high_ifc_checker = lamb.ifc.IFCChecker(
            model_context=lamb.ifc.IFCLabel.TH,
            tool_sink_label=p_high_tool_sink_label,
            tool_source_label=p_high_tool_source_label,
            secret_handling=lamb.ifc.SecretHandling.STATIC,
        )
        p_high_llm = Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=p_high_runtime,
                formatter=lamb.tool_result.VariableFormatter.ifc(p_high_ifc_checker),
            ),
        )

        def p_low_tool_sink_label(tool: Callable) -> lamb.ifc.IFCLabel:
            if tool == p_high_query_llm:
                return lamb.ifc.IFCLabel.UH
            if tool == p_high_query_llm_structured:
                return lamb.ifc.IFCLabel.UH
            return basic_tool_sink_label(tool)

        def p_low_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
            if tool == p_high_query_llm:
                return lamb.ifc.IFCLabel.UH
            if tool == p_high_query_llm_structured:
                return lamb.ifc.IFCLabel.TH
            return basic_tool_sink_label(tool)

        p_low_ifc_checker = lamb.ifc.IFCChecker(
            model_context=lamb.ifc.IFCLabel.TL,
            tool_sink_label=basic_tool_sink_label,
            tool_source_label=b_low_tool_source_label,
            secret_handling=lamb.ifc.SecretHandling.STATIC,
        )
        p_low_runtime = lamb.runtime.Runtime.default(functions_runtime, env)

        return Agent(
            model=model,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            make_core=lambda: AgentCore(
                runtime=p_low_runtime,
                formatter=lamb.tool_result.VariableFormatter.ifc(p_low_ifc_checker),
            ),
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
