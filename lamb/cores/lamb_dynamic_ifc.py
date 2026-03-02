from collections.abc import Callable

import agentdojo.functions_runtime as rt

import lamb.formatter
import lamb.ifc
import lamb.llm
import lamb.prompts
import lamb.query_llm
import lamb.runtime
import lamb.tool_categories
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
        formatter=lamb.formatter.VariableFormatter.ifc(b_ifc_checker),
    )


# TODO: Can we merge this with that of lamb_static_ifc?
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
        formatter=lamb.formatter.VariableFormatter.none(),
    )


def p_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> AgentCore:
    all_fns = list(functions_runtime.functions.values())

    b_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        lambda: b_low_make_core(all_fns, env),
    )
    query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_llm)
    query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(b_llm)

    runtime = lamb.runtime.Runtime(
        functions_runtime, env, [query_llm, query_llm_structured]
    )

    def p_on_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
        """Restrict P-LLM tools to high conf sinks."""

        assert ifc_label == lamb.ifc.IFCLabel.TH, "P-LLM label must be TH."
        high_conf_sink_fns = filter_tools(all_fns, lamb.tool_categories.HIGH_CONF_SINK)
        b_llm = Agent.bounded(
            model,
            lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
            lambda: b_high_make_core(all_fns, env),
        )
        query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(
            b_llm
        )
        high_conf_sink_fns.extend([query_llm, query_llm_structured])

        runtime.set_functions(high_conf_sink_fns)

    def p_tool_sink_label(tool: Callable) -> lamb.ifc.IFCLabel:
        if tool not in lamb.tool_categories.ALL:
            # b-llm takes care of ifc, so it is safe to forward
            return lamb.ifc.IFCLabel.top()
        return basic_tool_sink_label(tool)

    # TODO: use Function instead of Callable
    def p_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
        if tool not in lamb.tool_categories.ALL:
            # tool is query_llm or query_llm_structured
            # we allow the tool execution
            # and update the label dynamically afterwards
            return lamb.ifc.IFCLabel.TL
        return basic_tool_sink_label(tool)

    p_ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.TL,
        tool_sink_label=p_tool_sink_label,
        tool_source_label=p_tool_source_label,
        secret_handling=lamb.ifc.SecretHandling.DYNAMIC,
        on_model_context_change=p_on_context_change,
    )
    formatter = lamb.formatter.VariableFormatter.ifc(p_ifc_checker)

    query_llm_index = 0

    def p_low_query_llm_tool(prompt: str) -> lamb.query_llm.QueryLlmResponse:
        nonlocal query_llm_index
        response = lamb.query_llm.query_llm(b_llm, prompt)
        label = response.ifc_label
        if label and label.integ == lamb.ifc.Integrity.UNTRUSTED:
            var_name = f"<query_llm_{query_llm_index}:{label}/>"
            query_llm_index += 1
            formatter.var_vals[var_name] = str(response.response)
            p_ifc_checker.var_labels[var_name] = label
            p_ifc_checker.on_model_context_change(label)
        return response

    query_llm_structured_index = 0

    def p_low_query_llm_structured_tool(
        prompt: str, json_schema: dict
    ) -> lamb.query_llm.QueryLlmResponse:
        nonlocal query_llm_structured_index
        response = lamb.query_llm.query_llm_structured(b_llm, prompt, json_schema)
        label = response.ifc_label
        if label and not label.permitted_flow(lamb.ifc.IFCLabel.TL):
            var_name = f"<query_llm_{query_llm_structured_index}:{label}/>"
            query_llm_structured_index += 1
            formatter.var_vals[var_name] = str(response.response)
            p_ifc_checker.var_labels[var_name] = label
        return response

    return AgentCore(
        runtime=runtime,
        formatter=formatter,
    )
