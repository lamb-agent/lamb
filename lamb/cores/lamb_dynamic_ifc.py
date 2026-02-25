from collections.abc import Callable

import agentdojo.functions_runtime as rt

import lamb.ifc
import lamb.llm
import lamb.query_llm
import lamb.runtime
import lamb.tool_categories
import lamb.tool_result
from lamb.agent import (
    Agent,
    AgentCore,
    basic_tool_sink_label,
    basic_tool_source_label,
    filter_tools,
)


def make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> AgentCore:
    return p_make_core(model, functions_runtime, env)


def b_low_make_core(
    all_fns: list[rt.Function],
    env: rt.TaskEnvironment,
) -> AgentCore:
    read_only_fns = filter_tools(
        all_fns, lamb.tool_categories.READ_ONLY & lamb.tool_categories.UNTRUSTED_SINK
    )
    # no nested llms
    runtime = lamb.runtime.Runtime.default(rt.FunctionsRuntime(read_only_fns), env)

    def on_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
        """Restrict B-LLM tools to read-only high conf sinks."""

        assert ifc_label == lamb.ifc.IFCLabel.UH, "B-LLM label must be UH."
        high_conf_sink_fns = filter_tools(
            read_only_fns, lamb.tool_categories.HIGH_CONF_SINK
        )
        runtime.set_functions(high_conf_sink_fns)

    b_ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.UL,
        secret_handling=lamb.ifc.SecretHandling.DYNAMIC,
        tool_source_label=basic_tool_source_label,
        tool_sink_label=basic_tool_sink_label,
        on_model_context_change=on_context_change,
    )
    return AgentCore(
        runtime=runtime,
        formatter=lamb.tool_result.VariableFormatter.ifc(b_ifc_checker),
    )


def b_high_make_core(
    all_fns: list[rt.Function],
    env: rt.TaskEnvironment,
) -> AgentCore:
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
        formatter=lamb.tool_result.VariableFormatter.none(),
    )


def p_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> AgentCore:
    all_fns = list(functions_runtime.functions.values())

    b_llm = Agent.bounded(model, lambda: b_low_make_core(all_fns, env))
    query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_llm)
    query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(b_llm)

    runtime = lamb.runtime.Runtime(
        functions_runtime, env, [query_llm, query_llm_structured]
    )

    def p_on_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
        """Restrict P-LLM tools to high conf sinks."""

        assert ifc_label == lamb.ifc.IFCLabel.TH, "P-LLM label must be TH."
        high_conf_sink_fns = filter_tools(all_fns, lamb.tool_categories.HIGH_CONF_SINK)
        b_llm = Agent.bounded(model, lambda: b_high_make_core(all_fns, env))
        query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(
            b_llm
        )
        high_conf_sink_fns.extend([query_llm, query_llm_structured])

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
    return AgentCore(
        runtime=runtime,
        formatter=lamb.tool_result.VariableFormatter.ifc(p_ifc_checker),
    )
