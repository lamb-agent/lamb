import agentdojo.functions_runtime as rt

import lamb.formatter
import lamb.llm
import lamb.prompts
import lamb.query_llm
import lamb.runtime
from lamb import ifc
from lamb.agent import Agent, AgentCore


def make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: ifc.Labeler,
) -> AgentCore:
    bounded_fns = labeler.filter_tools(
        list(functions_runtime.functions.values()),
        ifc.ALL_TOOLS - {ifc.SystemAccess.PRIVILEGED},
    )
    b_ifc_checker = ifc.IFCChecker.none_with_context(labeler, ifc.IFCLabel.UL)
    b_llm = Agent.bounded(
        model,
        system_prompt=lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        # no nested llms
        make_core=lambda: AgentCore(
            runtime=lamb.runtime.Runtime.default(rt.FunctionsRuntime(bounded_fns), env),
            formatter=lamb.formatter.VariableFormatter.ifc(b_ifc_checker),
            ifc_checker=b_ifc_checker,
        ),
        initial_label="UH",
    )
    query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_llm)
    query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(b_llm)

    p_ifc_checker = ifc.IFCChecker.vars_only(labeler)
    return AgentCore(
        runtime=lamb.runtime.Runtime(
            functions_runtime, env, [query_llm, query_llm_structured]
        ),
        formatter=lamb.formatter.VariableFormatter.ifc(p_ifc_checker),
        ifc_checker=p_ifc_checker,
    )
