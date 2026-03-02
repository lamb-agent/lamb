from collections.abc import Callable

import agentdojo.functions_runtime as rt

import lamb.ifc
import lamb.llm
import lamb.prompts
import lamb.query_llm
import lamb.runtime
import lamb.tool_categories
import lamb.tool_result
import lamb.types
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
    return p_low_make_core(model, functions_runtime, env)


def b_high_make_core(
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> AgentCore:
    all_fns = list(functions_runtime.functions.values())

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
    return AgentCore(
        runtime=b_high_runtime,
        formatter=lamb.tool_result.VariableFormatter.ifc(b_high_ifc_checker),
    )


def b_low_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> AgentCore:
    all_fns = list(functions_runtime.functions.values())
    b_high_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        make_core=lambda: b_high_make_core(functions_runtime, env),
    )
    b_low_query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_high_llm)
    b_low_runtime = lamb.runtime.Runtime(
        functions_runtime=rt.FunctionsRuntime(
            filter_tools(
                all_fns,
                lamb.tool_categories.READ_ONLY | lamb.tool_categories.UNTRUSTED_SINK,
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
    return AgentCore(
        runtime=b_low_runtime,
        formatter=lamb.tool_result.VariableFormatter.ifc(b_low_ifc_checker),
    )


def p_high_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> AgentCore:
    all_fns = list(functions_runtime.functions.values())
    b_high_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        make_core=lambda: b_high_make_core(functions_runtime, env),
    )
    p_high_query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_high_llm)
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
    return AgentCore(
        runtime=p_high_runtime,
        formatter=lamb.tool_result.VariableFormatter.ifc(p_high_ifc_checker),
    )


def p_low_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> AgentCore:
    b_high_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        make_core=lambda: b_high_make_core(functions_runtime, env),
    )

    b_low_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_VARS_SYSTEM_PROMPT,
        make_core=lambda: b_low_make_core(model, functions_runtime, env),
    )

    p_high_llm = Agent(
        name="lamb-static-ifc-high-priviledged",
        model=model,
        identity=lamb.types.Identity.PRIVILEDGED,
        system_prompt=lamb.prompts.P_HIGH_LLM_SYSTEM_PROMPT,
        make_core=lambda: p_high_make_core(model, functions_runtime, env),
    )

    def p_low_tool_sink_label(tool: Callable) -> lamb.ifc.IFCLabel:
        if tool not in lamb.tool_categories.ALL:
            # each flow is possible because sub-agent is chosen dynamically
            return lamb.ifc.IFCLabel.top()
        return basic_tool_sink_label(tool)

    # TODO: use Function instead of Callable
    def p_low_tool_source_label(tool: Callable) -> lamb.ifc.IFCLabel:
        if tool not in lamb.tool_categories.ALL:
            # tool is query_llm or query_llm_structured
            # we allow the tool execution
            # and update the label dynamically afterwards
            return lamb.ifc.IFCLabel.TL
        return basic_tool_sink_label(tool)

    p_low_ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.TL,
        tool_sink_label=p_low_tool_sink_label,
        tool_source_label=p_low_tool_source_label,
        secret_handling=lamb.ifc.SecretHandling.STATIC,
    )

    p_low_formatter = lamb.tool_result.VariableFormatter.ifc(p_low_ifc_checker)

    def determine_agent(prompt: str) -> Agent:
        label = p_low_ifc_checker.response_label(prompt)
        agent: Agent
        match label:
            case lamb.ifc.IFCLabel.UL:
                agent = b_low_llm
            case lamb.ifc.IFCLabel.UH:
                agent = b_high_llm
            case lamb.ifc.IFCLabel.TH:
                agent = p_high_llm
            case lamb.ifc.IFCLabel.TL:
                # this case shouldn't happen if the LLM is smart
                raise ValueError(
                    "There is no point in calling query_llm without variables."
                )
        return agent

    query_llm_index = 0

    def p_low_query_llm_tool(prompt: str) -> lamb.query_llm.QueryLlmResponse:
        nonlocal query_llm_index
        agent = determine_agent(prompt)
        response = lamb.query_llm.query_llm(agent, prompt)
        label = response.ifc_label
        if label and not label.permitted_flow(lamb.ifc.IFCLabel.TL):
            var_name = f"<query_llm_{query_llm_index}:{label}/>"
            query_llm_index += 1
            p_low_formatter.var_vals[var_name] = str(response.response)
            p_low_ifc_checker.var_labels[var_name] = label
        return response

    query_llm_structured_index = 0

    def p_low_query_llm_structured_tool(
        prompt: str, json_schema: dict
    ) -> lamb.query_llm.QueryLlmResponse:
        nonlocal query_llm_structured_index
        agent = determine_agent(prompt)
        response = lamb.query_llm.query_llm_structured(agent, prompt, json_schema)
        label = response.ifc_label
        if label and not label.permitted_flow(lamb.ifc.IFCLabel.TL):
            var_name = f"<query_llm_{query_llm_structured_index}:{label}/>"
            query_llm_structured_index += 1
            p_low_formatter.var_vals[var_name] = str(response.response)
            p_low_ifc_checker.var_labels[var_name] = label
        return response

    p_low_query_llm = lamb.query_llm.make_query_llm_fn(p_low_query_llm_tool)
    p_low_query_llm_structured = lamb.query_llm.make_query_llm_structured_fn(
        p_low_query_llm_structured_tool
    )

    p_low_runtime = lamb.runtime.Runtime(
        functions_runtime, env, [p_low_query_llm, p_low_query_llm_structured]
    )
    return AgentCore(
        runtime=p_low_runtime,
        formatter=p_low_formatter,
    )
