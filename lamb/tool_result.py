from collections.abc import Callable, Sequence
from typing import Any, Protocol, assert_never, runtime_checkable

import agentdojo.functions_runtime as rt
import yaml
from pydantic import BaseModel


@runtime_checkable
class Formatter(Protocol):
    def format(
        self,
        result: rt.FunctionReturnType,
        tool: str,
    ) -> str: ...
    def expand(self, arg: rt.FunctionCallArgTypes) -> rt.FunctionCallArgTypes: ...


class BasicFormatter:
    def format(self, result: rt.FunctionReturnType, tool: str) -> str:
        return _stringify_result(result)

    def expand(self, arg: rt.FunctionCallArgTypes) -> rt.FunctionCallArgTypes:
        return arg


# TODO: Allow custom function that checks if the result
# should be obscured
class VariableFormatter:
    env: dict[str, str]
    tools: dict[str, int]

    def __init__(self) -> None:
        self.env = {}
        self.tools = {}

    def format(
        self,
        result: rt.FunctionReturnType,
        tool: str,
    ) -> str:
        def contains_str(val: Any) -> bool:
            match val:
                case int() | float() | bool() | None:
                    return False
                case str():
                    return True
                case dict():
                    return any(contains_str(item) for item in val.values())
                case BaseModel():
                    return any(
                        contains_str(field) for field in val.model_dump().values()
                    )
                case Sequence():
                    return any(contains_str(item) for item in val)
                case _:
                    raise TypeError(f"Unknown type for str checking: {type(val)}")

        def make_new_var() -> str:
            assert tool in self.tools, f"Tool {tool} was not added to tools."
            var = f"<{tool}_{self.tools[tool]}/>"
            self.tools[tool] += 1
            return var

        if tool not in self.tools:
            self.tools[tool] = 0

        result_str = _stringify_result(result)
        if contains_str(result):
            new_var = make_new_var()
            self.env[new_var] = result_str
            return new_var
        return result_str

    def expand(self, arg: rt.FunctionCallArgTypes) -> rt.FunctionCallArgTypes:
        """Replace variable references in tool call args with their respective value.

        Assuming the arg is: "Summarize this: <variable1/>"
        and the env contains { "variable1": "Untrusted data" },
        We will get: "Summarize this: Untrusted data".
        """

        def replace_vars(text: str) -> str:
            expanded_text = text
            for var, val in self.env.items():
                expanded_text = expanded_text.replace(var, val)
            return expanded_text

        match arg:
            case int() | float() | bool() | None:
                return arg
            case str():
                return replace_vars(arg)
            case list():
                return [self.expand(item) for item in arg]
            case dict():
                return {key: self.expand(val) for key, val in arg.items()}
            case rt.FunctionCall():
                return arg.model_copy(
                    update={
                        "args": {key: self.expand(val) for key, val in arg.args.items()}
                    }
                )
            case _:
                assert_never(arg)


def _stringify_result(
    val: rt.FunctionReturnType,
    dump_fn: Callable[[dict | list[dict]], str] = yaml.safe_dump,
) -> str:
    match val:
        case int() | float() | bool() | None:
            return str(val)
        case str():
            return val
        case dict():
            return dump_fn(val).strip()
        case BaseModel():
            return dump_fn(val.model_dump()).strip()
        case Sequence():
            res_items = []
            for item in val:
                match item:
                    case int() | float() | bool() | None:
                        res_items += str(item)
                    case str() | dict():
                        res_items += item
                    case BaseModel():
                        res_items += item.model_dump()
                    case _:
                        raise TypeError(
                            f"Not valid type for item tool result: {type(item)}"
                        )
            return dump_fn(res_items).strip()
        case _:
            assert_never(val)
