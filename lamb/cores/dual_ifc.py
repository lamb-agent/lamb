import agentdojo.functions_runtime as rt

import lamb.formatter
import lamb.ifc
import lamb.llm
import lamb.query_llm
import lamb.runtime
import lamb.tool_categories
from lamb.agent import (
    Agent,
    AgentCore,
    filter_tools,
)


def make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
    secret_handling: lamb.ifc.SecretHandling,
) -> AgentCore:
    all_fns = list(functions_runtime.functions.values())

    q_llm = Agent.quarantined(model)

    def label_response(
        response: lamb.query_llm.QueryLlmResponse,
        ifc_label: str,
    ) -> lamb.query_llm.QueryLlmResponse:
        """Label the q-llm's response with the same label as the input."""

        assert response.ifc_label is None, "Q-LLM shouldn't have IFC"
        response.ifc_label = lamb.ifc.IFCLabel.from_str(ifc_label)
        assert response.ifc_label.integ == lamb.ifc.Integrity.UNTRUSTED,f"Unexpected IFC label in q-llm prompt: {ifc_label}"
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
        # TODO: decouple tool filtering from tool_categories
        high_conf_sink_fns = filter_tools(all_fns, lamb.tool_categories.HIGH_CONF_SINK)

        runtime.set_functions(high_conf_sink_fns, [query_llm, query_llm_structured])

    ifc_checker: lamb.ifc.IFCChecker
    match secret_handling:
        case lamb.ifc.SecretHandling.DYNAMIC:
            ifc_checker = lamb.ifc.IFCChecker(
                model_context=lamb.ifc.IFCLabel.TL,
                labeler=labeler,
                secret_handling=lamb.ifc.SecretHandling.DYNAMIC,
                on_model_context_change=on_model_context_change,
            )
        case lamb.ifc.SecretHandling.STATIC:
            ifc_checker = lamb.ifc.IFCChecker(
                model_context=lamb.ifc.IFCLabel.TL,
                labeler=labeler,
                secret_handling=lamb.ifc.SecretHandling.STATIC,
            )

    return AgentCore(
        runtime=runtime,
        formatter=lamb.formatter.VariableFormatter.ifc(ifc_checker),
        ifc_checker=ifc_checker,
    )
