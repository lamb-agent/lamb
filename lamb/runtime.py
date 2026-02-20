from collections.abc import Callable, Mapping

import agentdojo.functions_runtime as rt
import pydantic


class Runtime:
    functions_runtime: rt.FunctionsRuntime
    env: rt.TaskEnvironment

    def __init__(
        self,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        custom_functions: list[rt.Function],
    ) -> None:
        for fn in custom_functions:
            functions_runtime.register_function(fn)
        self.functions_runtime = functions_runtime
        self.env = env

    def tools(self) -> list[str]:
        return list(self.functions_runtime.functions.keys())

    def get_function(self, tool: str) -> rt.Function:
        return self.functions_runtime.functions[tool]

    def exec(
        self,
        tool: str,
        args: Mapping[str, rt.FunctionCallArgTypes],
    ) -> tuple[rt.FunctionReturnType, str | None]:
        return self.functions_runtime.run_function(self.env, tool, args)

    def set_functions(self, fns: list[rt.Function]) -> None:
        self.functions_runtime = rt.FunctionsRuntime(fns)

    @staticmethod
    def default(
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
    ) -> "Runtime":
        return Runtime(functions_runtime, env, custom_functions=[])

    @staticmethod
    def empty() -> "Runtime":
        return Runtime(rt.FunctionsRuntime([]), rt.EmptyEnv(), custom_functions=[])


def make_function(
    name: str,
    description: str,
    parameters: list[tuple[str, str, str]],
    fn: Callable,
) -> rt.Function:
    params = {
        name: (arg_type, pydantic.Field(description=description))
        for name, arg_type, description in parameters
    }
    return rt.Function(
        name=name,
        description=description,
        parameters=pydantic.create_model(
            f"Input schema for {name}", field_definitions=params
        ),
        dependencies={},
        run=fn,
        full_docstring="",  # unused
        return_type=None,  # unused
    )
