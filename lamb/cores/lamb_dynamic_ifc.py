import agentdojo.functions_runtime as rt

import lamb.formatter
import lamb.ifc
import lamb.llm
import lamb.prompts
import lamb.query_llm
import lamb.runtime
import lamb.types
from lamb.agent import (
    Agent,
    AgentCore,
)

B_LOW_FILTER = lamb.ifc.ALL_TOOLS - {
    lamb.ifc.SystemAccess.PRIVILEGED,
    lamb.ifc.Integrity.TRUSTED,
}
B_HIGH_FILTER = lamb.ifc.ALL_TOOLS - {
    lamb.ifc.SystemAccess.PRIVILEGED,
    lamb.ifc.Integrity.TRUSTED,
    lamb.ifc.Confidentiality.LOW,
}
P_HIGH_FILTER = lamb.ifc.ALL_TOOLS - {lamb.ifc.Confidentiality.LOW}


def make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
) -> AgentCore:
    return p_make_core(model, functions_runtime, env, labeler)


def b_low_make_core(
    tools: lamb.types.Tools,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
) -> AgentCore:
    bounded_fns = labeler.filter_tools(tools, B_LOW_FILTER)
    # no nested llms
    runtime = lamb.runtime.Runtime.default(rt.FunctionsRuntime(bounded_fns), env)

    def on_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
        """Restrict B-LLM tools to read-only high conf sinks."""

        assert ifc_label == lamb.ifc.IFCLabel.UH, "B-LLM label must be UH."
        high_conf_sink_fns = labeler.filter_tools(bounded_fns, B_HIGH_FILTER)
        runtime.set_functions(high_conf_sink_fns, [])
        # Note that the system prompt does not change and thus the LLM
        # still thinks it has variable passing

    b_ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.UL,
        secret_handling=lamb.ifc.SecretHandling.DYNAMIC,
        labeler=labeler,
        on_model_context_change=on_context_change,
    )
    return AgentCore(
        runtime=runtime,
        formatter=lamb.formatter.VariableFormatter.ifc(b_ifc_checker),
        ifc_checker=b_ifc_checker,
    )


def p_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
) -> AgentCore:
    tools = list(functions_runtime.functions.values())

    b_low_llm = Agent.bounded(
        model=model,
        system_prompt=lamb.prompts.B_LLM_VARS_SYSTEM_PROMPT,
        make_core=lambda: b_low_make_core(tools, env, labeler),
        initial_label="TL",
    )
    b_high_llm = Agent.bounded(
        model=model,
        system_prompt=lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        make_core=lambda: Agent.bounded_high_make_core(functions_runtime, env, labeler),
        initial_label="TH",
    )

    def get_agent(ifc_label: str) -> Agent:
        if lamb.ifc.IFCLabel.from_str(ifc_label).is_high():
            return b_high_llm
        return b_low_llm

    query_llm = lamb.query_llm.make_query_llm_fn(
        lambda prompt, ifc_label: lamb.query_llm.query_llm(get_agent(ifc_label), prompt)
    )
    query_llm_structured = lamb.query_llm.make_query_llm_structured_fn(
        lambda prompt, json_schema, ifc_label: lamb.query_llm.query_llm_structured(
            get_agent(ifc_label), prompt, json_schema
        )
    )

    runtime = lamb.runtime.Runtime(
        functions_runtime, env, [query_llm, query_llm_structured]
    )

    def p_on_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
        """Restrict P-LLM tools to high conf sinks."""

        assert ifc_label == lamb.ifc.IFCLabel.TH, "P-LLM label must be TH."
        high_conf_sink_fns = labeler.filter_tools(tools, P_HIGH_FILTER)
        query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_high_llm)
        query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(
            b_high_llm
        )

        runtime.set_functions(high_conf_sink_fns, [query_llm, query_llm_structured])

    p_ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.TL,
        labeler=labeler,
        secret_handling=lamb.ifc.SecretHandling.DYNAMIC,
        on_model_context_change=p_on_context_change,
    )
    formatter = lamb.formatter.VariableFormatter.ifc(p_ifc_checker)

    return AgentCore(
        runtime=runtime,
        formatter=formatter,
        ifc_checker=p_ifc_checker,
    )
