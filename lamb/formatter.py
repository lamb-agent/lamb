from collections.abc import Callable, Mapping, Sequence
from typing import assert_never

import agentdojo.functions_runtime as rt
import yaml
from pydantic import BaseModel

from lamb import ifc, tool_categories, types


class VariableFormatter(types.Formatter):
    """Can hide results in variables and expand them when used in arguments."""

    var_vals: dict[str, str]
    tools: dict[str, int]
    hide_result: Callable[[Callable], bool]
    notify_new_var: Callable[[rt.Function, str], None]
    check_call: types.CheckIFC

    def __init__(
        self,
        hide_result: Callable[[Callable], bool],
        notify_new_var: Callable[[rt.Function, str], None] = lambda tool, index: None,
        check_call: types.CheckIFC = lambda tool, args: None,  # noqa: ARG005
    ) -> None:
        self.var_vals = {}
        self.tools = {}
        self.hide_result = hide_result
        self.notify_new_var = notify_new_var
        self.check_call = check_call

    def format(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
    ) -> str:
        tool_name = tool.name

        def make_new_var() -> str:
            assert tool_name in self.tools, f"Tool {tool_name} was not added to tools."
            index = self.tools[tool_name]
            var = f"<{tool_name}_{index}/>"
            self.notify_new_var(tool, var)
            self.tools[tool_name] += 1
            return var

        if tool_name not in self.tools:
            self.tools[tool_name] = 0

        result_str = _stringify_result(result)
        if self.hide_result(tool.run):
            new_var = make_new_var()
            self.var_vals[new_var] = result_str
            return new_var
        return result_str

    def expand(
        self,
        tool: rt.Function,
        args: Mapping[str, rt.FunctionCallArgTypes],
    ) -> Mapping[str, rt.FunctionCallArgTypes]:
        """Replace variable references in tool call args with their respective value.

        Assuming the arg is: "Summarize this: <variable1/>"
        and the env contains { "variable1": "Untrusted data" },
        We will get: "Summarize this: Untrusted data".
        """

        def replace_vars(text: str) -> str:
            expanded_text = text
            for var, val in self.var_vals.items():
                expanded_text = expanded_text.replace(var, val)
            return expanded_text

        def expand_arg(arg: rt.FunctionCallArgTypes) -> rt.FunctionCallArgTypes:
            match arg:
                case int() | float() | bool() | None:
                    return arg
                case str():
                    return replace_vars(arg)
                case list():
                    return [expand_arg(item) for item in arg]
                case dict():
                    return {key: expand_arg(val) for key, val in arg.items()}
                case rt.FunctionCall():
                    return arg.model_copy(update={"args": self.expand(tool, arg.args)})
                case _:
                    assert_never(arg)

        self.check_call(tool, args)
        return {key: expand_arg(val) for key, val in args.items()}

    @staticmethod
    def none() -> "VariableFormatter":
        """Never hide variables."""

        return VariableFormatter(hide_result=lambda _: False)

    @staticmethod
    def integrity_based(query_llm: Callable | None = None) -> "VariableFormatter":
        """Create new variable formatter that hides results from untrusted sources."""

        untrusted_source_fns = tool_categories.UNTRUSTED_SOURCE | {query_llm}

        return VariableFormatter(hide_result=untrusted_source_fns.__contains__)

    @staticmethod
    def ifc(ifc_checker: ifc.IFCChecker) -> "VariableFormatter":
        return VariableFormatter(
            hide_result=ifc_checker.hide_result,
            notify_new_var=ifc_checker.on_new_var,
            check_call=ifc_checker.check,
        )


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
