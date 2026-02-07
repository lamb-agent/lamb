from collections.abc import Callable, Sequence
from typing import Protocol, assert_never, runtime_checkable

import agentdojo.functions_runtime as rt
import yaml
from pydantic import BaseModel


@runtime_checkable
class Formatter(Protocol):
    def format(
        self,
        result: rt.FunctionReturnType,
        tool: rt.Function,
    ) -> str: ...
    def expand(self, arg: rt.FunctionCallArgTypes) -> rt.FunctionCallArgTypes: ...


class BasicFormatter:
    def format(self, result: rt.FunctionReturnType, tool: rt.Function) -> str:
        return _stringify_result(result)

    def expand(self, arg: rt.FunctionCallArgTypes) -> rt.FunctionCallArgTypes:
        return arg


class VariableFormatter:
    env: dict[str, str]
    tools: dict[str, int]
    hide_result: Callable[[Callable], bool]

    def __init__(self, hide_result: Callable[[Callable], bool]) -> None:
        self.env = {}
        self.tools = {}
        self.hide_result = hide_result

    def format(
        self,
        result: rt.FunctionReturnType,
        tool: rt.Function,
    ) -> str:
        tool_name = tool.name

        def make_new_var() -> str:
            assert tool_name in self.tools, f"Tool {tool_name} was not added to tools."
            var = f"<{tool_name}_{self.tools[tool_name]}/>"
            self.tools[tool_name] += 1
            return var

        if tool_name not in self.tools:
            self.tools[tool_name] = 0

        result_str = _stringify_result(result)
        if self.hide_result(tool.run):
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
                        res_items.append(str(item))
                    case str() | dict():
                        res_items.append(item)
                    case BaseModel():
                        res_items.append(item.model_dump())
                    case _:
                        raise TypeError(
                            f"Not valid type for item tool result: {type(item)}"
                        )
            return dump_fn(res_items).strip()
        case _:
            assert_never(val)
