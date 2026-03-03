from collections.abc import Callable

import agentdojo.functions_runtime as rt
import pydantic

from lamb import types


class Runtime:
    functions_runtime: rt.FunctionsRuntime
    env: rt.TaskEnvironment
    custom_functions: dict[str, Callable]

    def __init__(
        self,
        functions_runtime: rt.FunctionsRuntime,
        env: rt.TaskEnvironment,
        custom_functions: list[rt.Function],
    ) -> None:
        for fn in custom_functions:
            functions_runtime.register_function(fn)
        self.functions_runtime = functions_runtime
        self.custom_functions = {fn.name: fn.run for fn in custom_functions}
        self.env = env

    def tools(self) -> list[str]:
        return list(self.functions_runtime.functions.keys())

    def get_function(self, tool: str) -> rt.Function:
        return self.functions_runtime.functions[tool]

    def exec(
        self,
        tool: str,
        args: types.Args,
    ) -> tuple[rt.FunctionReturnType, str | None]:
        custom = self.custom_functions.get(tool)
        if custom:
            try:
                result = custom(**args)
            except Exception as e:  # noqa: BLE001
                return "", str(e)
            else:
                return result, None

        return self.functions_runtime.run_function(self.env, tool, args)

    def set_functions(
        self, fns: list[rt.Function], custom_functions: list[rt.Function]
    ) -> None:
        self.functions_runtime = rt.FunctionsRuntime(fns)
        for fn in custom_functions:
            self.functions_runtime.register_function(fn)
        self.custom_functions = {fn.name: fn.run for fn in custom_functions}

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
    # NOTE: the typing here is a bit fucked up.
    # params has the correct type expected inside the create_model function,
    # but the kwargs-syntax seems to be the bottleneck,
    # so we have to pass it as dict.
    params: dict = {
        name: (arg_type, pydantic.Field(description=description))
        for name, arg_type, description in parameters
    }
    return rt.Function(
        name=name,
        description=description,
        parameters=pydantic.create_model(f"Input schema for {name}", **params),
        dependencies={},
        run=fn,
        full_docstring="",  # unused
        return_type=None,  # unused
    )
