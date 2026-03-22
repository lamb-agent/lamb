import agentdojo.functions_runtime as rt

import lamb.formatter
import lamb.llm
import lamb.prompts
import lamb.query_llm
import lamb.runtime
import lamb.tool_categories
from lamb.agent import Agent, AgentCore


def make_core(
    model: lamb.llm.Llm,
    functions_runtime: rt.FunctionsRuntime,
    env: rt.TaskEnvironment,
) -> AgentCore:
    read_only_fns = [
        fn
        for fn in functions_runtime.functions.values()
        if fn.run in lamb.tool_categories.READ_ONLY
    ]
    b_llm = Agent.bounded(
        model,
        system_prompt=lamb.prompts.B_LLM_NO_VARS_SYSTEM_PROMPT,
        # no nested llms
        make_core=lambda: AgentCore(
            runtime=lamb.runtime.Runtime.default(
                rt.FunctionsRuntime(read_only_fns), env
            ),
            formatter=lamb.formatter.VariableFormatter.none(),
        ),
        initial_label="UH",
    )
    query_llm = lamb.query_llm.make_query_llm_fn_with_agent(b_llm)
    query_llm_structured = lamb.query_llm.make_query_llm_structured_fn_with_agent(b_llm)

    return AgentCore(
        runtime=lamb.runtime.Runtime(
            functions_runtime, env, [query_llm, query_llm_structured]
        ),
        formatter=lamb.formatter.VariableFormatter.integrity_based(query_llm.run),
    )
