from collections.abc import Callable, Sequence
from functools import reduce
from typing import assert_never

import agentdojo.functions_runtime as rt
import yaml
from pydantic import BaseModel

from lamb import ifc, tool_categories, tools, types


class BasicFormatter(types.Formatter):
    """Leaves arguments and results unchanged."""

    def format(self, tool: rt.Function, result: rt.FunctionReturnType) -> str:
        return _stringify_result(result)

    def expand(
        self, tool: rt.Function, arg: rt.FunctionCallArgTypes
    ) -> rt.FunctionCallArgTypes:
        return arg


class VariableFormatter(types.Formatter):
    """Can hide results in variables and expand them when used in arguments."""

    env: dict[str, str]
    tools: dict[str, int]
    hide_result: Callable[[Callable], bool]

    def __init__(self, hide_result: Callable[[Callable], bool]) -> None:
        self.env = {}
        self.tools = {}
        self.hide_result = hide_result

    def format(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
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

    def expand(
        self,
        tool: rt.Function,
        arg: rt.FunctionCallArgTypes,
    ) -> rt.FunctionCallArgTypes:
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
                return [self.expand(tool, item) for item in arg]
            case dict():
                return {key: self.expand(tool, val) for key, val in arg.items()}
            case rt.FunctionCall():
                return arg.model_copy(
                    update={
                        "args": {
                            key: self.expand(tool, val) for key, val in arg.args.items()
                        }
                    }
                )
            case _:
                assert_never(arg)

    @staticmethod
    def integrity_based() -> "VariableFormatter":
        """Create new variable formatter that hides results from untrusted sources."""

        untrusted_source_fns = {tools.query_llm} | (tool_categories.UNTRUSTED_SOURCE)

        return VariableFormatter(hide_result=untrusted_source_fns.__contains__)


class IFCFormatter(types.Formatter):
    variable_formatter: VariableFormatter
    get_model_context: Callable[[], ifc.IFCLabel]
    env: dict[str, ifc.IFCLabel]

    def __init__(self, get_model_context: Callable[[], ifc.IFCLabel]) -> None:
        def hide_result(tool: Callable) -> bool:
            return not ifc.permitted_tool_result(tool, get_model_context())

        self.env = {}
        self.variable_formatter = VariableFormatter(hide_result)
        self.get_model_context = get_model_context

    def format(self, tool: rt.Function, result: rt.FunctionReturnType) -> str:
        # the formatter's hide_result function already uses ifc
        # to determine if the result should be hidden in a variable
        maybe_variable = self.variable_formatter.format(tool, result)
        if maybe_variable in self.variable_formatter.env:
            # result is a variable, so we store its ifc label
            self.env[maybe_variable] = ifc.tool_source_label(tool.run)
        return maybe_variable

    def expand(
        self, tool: rt.Function, arg: rt.FunctionCallArgTypes
    ) -> rt.FunctionCallArgTypes:
        def extract_vars(text: str) -> set[str]:
            return {var for var in self.env if var in text}

        def find_vars(arg: rt.FunctionCallArgTypes) -> set[str]:
            match arg:
                case int() | float() | bool() | None:
                    return set()
                case str():
                    return extract_vars(arg)
                case list():
                    return reduce(set.union, (find_vars(item) for item in arg))
                case dict():
                    return reduce(set.union, (find_vars(item) for item in arg.values()))
                case rt.FunctionCall():
                    return reduce(
                        set.union, (find_vars(item) for item in arg.args.values())
                    )
                case _:
                    assert_never(arg)

        variable_labels = {self.env[var] for var in find_vars(arg)}
        if not ifc.permitted_tool_call(
            tool.run, self.get_model_context(), variable_labels
        ):
            raise ifc.IFCError(
                f"The follow argument contains a variable that violates the IFC policies: {arg}"  # noqa: E501
            )

        return self.variable_formatter.expand(tool, arg)


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
