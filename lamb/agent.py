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
        custom_functions = [
            lamb.query_llm.make_query_llm_fn(q_llm),
            lamb.query_llm.make_query_llm_structured_fn(q_llm),
        ]

        return Agent(
            model=model,
            runtime=lamb.runtime.Runtime(functions_runtime, env, custom_functions),
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,  # TODO: need dedicated
            formatter=lamb.tool_result.VariableFormatter.integrity_based(),
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
        custom_functions = [
            lamb.query_llm.make_query_llm_fn(b_llm),
            lamb.query_llm.make_query_llm_structured_fn(b_llm),
        ]

        return Agent(
            model=model,
            runtime=lamb.runtime.Runtime(functions_runtime, env, custom_functions),
            identity=lamb.types.Identity.PRIVILEDGED,
            system_prompt=lamb.prompts.P_LLM_SYSTEM_PROMPT,
            formatter=lamb.tool_result.VariableFormatter.integrity_based(),
        )
