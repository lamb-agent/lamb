from typing import assert_never

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
P_HIGH_FILTER = lamb.ifc.ALL_TOOLS - {lamb.ifc.Confidentiality.LOW}


def make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
) -> AgentCore:
    return p_low_make_core(model, functions_runtime, env, labeler)


def b_low_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
) -> AgentCore:
    tools = list(functions_runtime.functions.values())
    b_high_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        make_core=lambda: Agent.bounded_high_make_core(functions_runtime, env, labeler),
        initial_label="UH",
    )
    b_low_query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_high_llm)
    b_low_runtime = lamb.runtime.Runtime(
        functions_runtime=rt.FunctionsRuntime(
            labeler.filter_tools(tools, B_LOW_FILTER)
        ),
        env=env,
        custom_functions=[b_low_query_llm],
    )

    b_low_ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.UL,
        labeler=labeler,
        secret_handling=lamb.ifc.SecretHandling.STATIC,
    )
    return AgentCore(
        runtime=b_low_runtime,
        formatter=lamb.formatter.VariableFormatter.ifc(b_low_ifc_checker),
        ifc_checker=b_low_ifc_checker,
    )


def p_high_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
) -> AgentCore:
    tools = list(functions_runtime.functions.values())
    b_high_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        make_core=lambda: Agent.bounded_high_make_core(functions_runtime, env, labeler),
        initial_label="UH",
    )
    p_high_query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_high_llm)
    p_high_query_llm_structured = (
        lamb.query_llm.make_query_llm_structured_fn_with_agent(b_high_llm)
    )
    p_high_runtime = lamb.runtime.Runtime(
        functions_runtime=rt.FunctionsRuntime(
            labeler.filter_tools(
                tools, lamb.ifc.ALL_TOOLS - {lamb.ifc.Confidentiality.LOW}
            )
        ),
        env=env,
        custom_functions=[p_high_query_llm, p_high_query_llm_structured],
    )

    p_high_ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.TH,
        labeler=labeler,
        secret_handling=lamb.ifc.SecretHandling.STATIC,
    )
    return AgentCore(
        runtime=p_high_runtime,
        formatter=lamb.formatter.VariableFormatter.ifc(p_high_ifc_checker),
        ifc_checker=p_high_ifc_checker,
    )


def p_low_make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
    labeler: lamb.ifc.Labeler,
) -> AgentCore:
    b_high_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        make_core=lambda: Agent.bounded_high_make_core(functions_runtime, env, labeler),
        initial_label="UH",
    )

    b_low_llm = Agent.bounded(
        model,
        lamb.prompts.B_LLM_VARS_SYSTEM_PROMPT,
        make_core=lambda: b_low_make_core(model, functions_runtime, env, labeler),
        initial_label="UL",
    )

    p_high_llm = Agent(
        name="lamb-static-ifc-high-priviledged",
        model=model,
        identity=lamb.types.Identity.PRIVILEGED,
        system_prompt=lamb.prompts.P_HIGH_LLM_SYSTEM_PROMPT,
        make_core=lambda: p_high_make_core(model, functions_runtime, env, labeler),
        initial_label="TH",
    )

    p_low_ifc_checker = lamb.ifc.IFCChecker(
        model_context=lamb.ifc.IFCLabel.TL,
        labeler=labeler,
        secret_handling=lamb.ifc.SecretHandling.STATIC,
    )

    p_low_formatter = lamb.formatter.VariableFormatter.ifc(p_low_ifc_checker)

    def determine_agent(ifc_label: str) -> Agent:
        agent: Agent
        match lamb.ifc.IFCLabel.from_str(ifc_label):
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
            case _e:
                assert_never(_e)
        return agent

    def p_low_query_llm_tool(
        prompt: str, ifc_label: str
    ) -> lamb.query_llm.QueryLlmResponse:
        agent = determine_agent(ifc_label)
        return lamb.query_llm.query_llm(agent, prompt)

    def p_low_query_llm_structured_tool(
        prompt: str,
        json_schema: dict,
        ifc_label: str,
    ) -> lamb.query_llm.QueryLlmResponse:
        agent = determine_agent(ifc_label)
        return lamb.query_llm.query_llm_structured(agent, prompt, json_schema)

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
        ifc_checker=p_low_ifc_checker,
    )
