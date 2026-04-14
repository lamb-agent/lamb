import agentdojo.functions_runtime as rt

import lamb.formatter
import lamb.ifc
import lamb.llm
import lamb.query_llm
import lamb.runtime
from lamb.agent import (
    Agent,
    AgentCore,
)

P_HIGH_FILTER = lamb.ifc.ALL_TOOLS - {lamb.ifc.Confidentiality.LOW}


def make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
    policy: lamb.ifc.IFCPolicy,
) -> AgentCore:
    assert policy.integ == "hide", "Dual LLM must hide untrusted"
    tools = list(functions_runtime.functions.values())

    q_llm = Agent.quarantined(model, labeler)

    def label_response(
        response: lamb.query_llm.QueryLlmResponse,
        ifc_label: str,
    ) -> lamb.query_llm.QueryLlmResponse:
        """Label the q-llm's response with the same label as the input."""

        # we ignore the response's ifc_label
        response.ifc_label = lamb.ifc.IFCLabel.from_str(ifc_label)
        assert response.ifc_label.integ == lamb.ifc.Integrity.UNTRUSTED, (
            f"Unexpected IFC label in q-llm prompt: {ifc_label}"
        )
        return response

    query_llm = lamb.query_llm.make_query_llm_fn(
        lambda prompt, ifc_label: label_response(
            lamb.query_llm.query_llm(q_llm, prompt), ifc_label
        )
    )
    query_llm_structured = lamb.query_llm.make_query_llm_structured_fn(
        lambda prompt, json_schema, ifc_label: label_response(
            lamb.query_llm.query_llm_structured(q_llm, prompt, json_schema),
            ifc_label,
        )
    )
    runtime = lamb.runtime.Runtime(
        functions_runtime, env, [query_llm, query_llm_structured]
    )

    def on_model_context_change(ifc_label: lamb.ifc.IFCLabel) -> None:
        assert ifc_label == lamb.ifc.IFCLabel.TH, "P-LLM label must be TH."
        high_conf_sink_fns = labeler.filter_tools(tools, P_HIGH_FILTER)

        runtime.set_functions(high_conf_sink_fns, [query_llm, query_llm_structured])

    ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.TL,
        labeler=labeler,
        policy=policy,
        on_model_context_change=on_model_context_change,
    )

    return AgentCore(
        runtime=runtime,
        ifc_checker=ifc_checker,
    )
