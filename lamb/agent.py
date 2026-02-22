from collections.abc import Callable

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


class Agent:
    assemble_controller: Callable[[ResponseFormat], lamb.controller.Controller]
    system_prompt: str

    def __init__(
        self,
        model: lamb.llm.Llm,
        runtime: lamb.runtime.Runtime,
        identity: lamb.types.Identity,
        system_prompt: str,
        formatter: lamb.types.Formatter,
        max_iters: int = 25,
    ) -> None:
        self.system_prompt = system_prompt

        def assemble_controller(
            response_format: ResponseFormat,
        ) -> lamb.controller.Controller:
            tool_llm = lamb.llm.ToolLlm(llm=model, runtime=runtime)
            executor = lamb.tool_exec.ToolExec(runtime, formatter)

            def call_llm(messages: list[ChatMessage]) -> ChatMessage:
                return tool_llm.prompt(messages, response_format)

            def call_tools(calls: list[rt.FunctionCall]) -> list[ChatMessage]:
                return executor.exec(calls)

            return lamb.controller.Controller(
                call_llm,
                call_tools,
                identity,
                max_iters,
            )

        self.assemble_controller = assemble_controller

    def prompt(
        self,
        user_prompt: str,
        response_format: ResponseFormat = lamb.types.TEXT_FORMAT,
    ) -> list[ChatMessage]:
        controller = self.assemble_controller(response_format)
        return controller.loop(
            [
                lamb.types.make_user_prompt(self.system_prompt),
                lamb.types.make_user_prompt(user_prompt),
            ]
        )

    @staticmethod
    def single(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        formatter: lamb.types.Formatter,
    ) -> "Agent":
        return Agent(
            model=model,
            runtime=lamb.runtime.Runtime.default(functions_runtime, env),
            identity=lamb.types.Identity.SINGLE,
            system_prompt=lamb.prompts.Q_LLM_SYSTEM_PROMPT,  # TODO: single prompt
            formatter=formatter,
        )

    @staticmethod
    def quarantined(
        model: lamb.llm.Llm,
    ) -> "Agent":
        return Agent(
            model=model,
            runtime=lamb.runtime.Runtime.empty(),
            identity=lamb.types.Identity.QUARANTINED,
            system_prompt=lamb.prompts.Q_LLM_SYSTEM_PROMPT,
            formatter=lamb.tool_result.VariableFormatter.none(),
        )

    @staticmethod
    def dual(
        model: lamb.llm.Llm,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Agent":
        q_llm = Agent.quarantined(model)
        query_llm = lamb.query_llm.make_query_llm_fn(q_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn(q_llm)

        return Agent(
            model=model,
            runtime=lamb.runtime.Runtime(
                functions_runtime, env, [query_llm, query_llm_structured]
            ),
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,  # TODO: need dedicated
            formatter=lamb.tool_result.VariableFormatter.integrity_based(query_llm.run),
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
            lamb.query_llm.make_query_llm_fn(q_llm),
            lamb.query_llm.make_query_llm_structured_fn(q_llm),
        ]

        return Agent(
            model=model,
            runtime=lamb.runtime.Runtime(functions_runtime, env, custom_functions),
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,  # TODO: need dedicated
            formatter=lamb.tool_result.VariableFormatter.ifc(ifc_checker),
        )

    @staticmethod
    def bounded(
        model: lamb.llm.Llm,
        runtime: lamb.runtime.Runtime,
        formatter: lamb.types.Formatter,
    ) -> "Agent":
        return Agent(
            model=model,
            runtime=runtime,
            identity=lamb.types.Identity.BOUNDED,
            system_prompt=lamb.prompts.B_LLM_SYSTEM_PROMPT,
            formatter=formatter,
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
            lamb.runtime.Runtime.default(rt.FunctionsRuntime(read_only_fns), env),
            lamb.tool_result.VariableFormatter.integrity_based(),
        )
        query_llm = lamb.query_llm.make_query_llm_fn(b_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn(b_llm)

        return Agent(
            model=model,
            runtime=lamb.runtime.Runtime(
                functions_runtime, env, [query_llm, query_llm_structured]
            ),
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            formatter=lamb.tool_result.VariableFormatter.integrity_based(query_llm.run),
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

        def b_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
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

        def b_tool_sink_label(tool: Callable) -> lamb.ifc.IFCLabel:
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

        b_ifc_checker = lamb.ifc.IFCChecker(
            model_context=lamb.ifc.IFCLabel.UL,
            secret_handling=lamb.ifc.SecretHandling.DYNAMIC,
            tool_source_label=b_tool_source_label,
            tool_sink_label=b_tool_sink_label,
            on_model_context_change=b_on_context_change,
        )
        b_llm = Agent.bounded(
            model,
            b_runtime,
            lamb.tool_result.VariableFormatter.ifc(b_ifc_checker),
        )
        query_llm = lamb.query_llm.make_query_llm_fn(b_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn(b_llm)
        runtime = lamb.runtime.Runtime(
            functions_runtime, env, [query_llm, query_llm_structured]
        )

        def p_on_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
            """Restrict P-LLM tools to high conf sinks."""

            assert ifc_label == lamb.ifc.IFCLabel.TH, "P-LLM label must be TH."
            high_conf_sink_fns = [
                fn for fn in all_fns if fn.run in lamb.tool_categories.HIGH_CONF_SINK
            ]
            read_only_high_conf_fns = [
                fn
                for fn in high_conf_sink_fns
                if fn.run in lamb.tool_categories.READ_ONLY
            ]
            b_runtime.set_functions(read_only_high_conf_fns)
            b_llm = Agent.bounded(
                model,
                b_runtime,
                # at the top of the lattice there is no point in hiding variables
                lamb.tool_result.VariableFormatter.none(),
            )
            custom_functions = [
                lamb.query_llm.make_query_llm_fn(b_llm),
                lamb.query_llm.make_query_llm_structured_fn(b_llm),
            ]
            high_conf_sink_fns.extend(custom_functions)

            runtime.set_functions(high_conf_sink_fns)

        def p_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
            integ = (
                lamb.ifc.Integrity.TRUSTED
                if tool
                in (lamb.tool_categories.TRUSTED_SOURCE | {query_llm_structured.run})
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

        return Agent(
            model=model,
            runtime=runtime,
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            formatter=lamb.tool_result.VariableFormatter.ifc(p_ifc_checker),
        )
