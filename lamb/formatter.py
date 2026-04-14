import typing
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import assert_never

import agentdojo.functions_runtime as rt
import yaml
from pydantic import BaseModel

from lamb import ifc, query_llm, tool_categories, types


class VariableFormatter(types.Formatter):
    """Can hide results in variables and expand them when used in arguments."""

    var_vals: dict[str, str]
    tools: dict[str, int]
    hide_result: Callable[
        [rt.Function, rt.FunctionReturnType, str],
        tuple[bool, ifc.IFCLabel | None, ifc.IFCLabel | None],
    ]
    check_call: Callable[
        [rt.Function, types.Args, types.Args],
        tuple[ifc.IFCLabel | None, ifc.IFCLabel | None],
    ]

    def __init__(
        self,
        hide_result: Callable[
            [rt.Function, rt.FunctionReturnType, str],
            tuple[bool, ifc.IFCLabel | None, ifc.IFCLabel | None],
        ],
        check_call: Callable[
            [rt.Function, types.Args, types.Args],
            tuple[ifc.IFCLabel | None, ifc.IFCLabel | None],
        ] = lambda _tool, _args_hidden, _args_expanded: (
            None,
            None,
        ),
    ) -> None:
        self.var_vals = {}
        self.tools = {}
        self.hide_result = hide_result
        self.check_call = check_call

    def format(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
    ) -> tuple[str, str, str]:
        tool_name = tool.name

        def get_next_var() -> str:
            assert tool_name in self.tools, f"Tool {tool_name} was not added to tools."
            index = self.tools[tool_name]
            return f"<{tool_name}_{index}/>"

        if tool_name not in self.tools:
            self.tools[tool_name] = 0

        next_var = get_next_var()

        result_str = _stringify_result(result)

        hide, source_label, sink_label = self.hide_result(tool, result, next_var)
        if hide:
            self.var_vals[next_var] = result_str
            self.tools[tool_name] += 1
            return next_var, str(source_label), str(sink_label)

        return (
            result_str,
            str(source_label) if source_label else "",
            str(sink_label) if source_label else "",
        )

    def expand(
        self,
        tool: rt.Function,
        args: types.Args,
    ) -> tuple[types.Args, str, str]:
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

        def expand_arg(arg: types.FunctionCallArgTypes) -> types.FunctionCallArgTypes:
            match arg:
                case int() | float() | bool() | None:
                    return arg
                case str():
                    return replace_vars(arg).strip()  # models love to append new lines
                case list() if types.is_arg_list(arg):
                    return [expand_arg(item) for item in arg]
                case dict() if types.is_arg_dict(arg):
                    return {key: expand_arg(val) for key, val in arg.items()}
                case types.FunctionCall():
                    return replace(arg, args=self.expand(tool, arg.args))
                case _:
                    assert_never(
                        typing.cast("typing.Never", arg)
                    )  # static code analysis can't compute this

        expanded_args = {key: expand_arg(val) for key, val in args.items()}
        tool.parameters.model_validate(expanded_args)
        source_label, sink_label = self.check_call(tool, args, expanded_args)
        # attach ifc label as extra argument for query_llm functions
        expanded_args["ifc_label"] = str(source_label)
        return (
            expanded_args,
            str(source_label) if source_label else "",
            str(sink_label) if source_label else "",
        )

    @staticmethod
    def ifc(ifc_checker: ifc.IFCChecker) -> "VariableFormatter":
        return VariableFormatter(
            hide_result=ifc_checker.hide,
            check_call=ifc_checker.check,
        )


def _stringify_result(  # noqa: C901
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
        case query_llm.QueryLlmResponse():
            return str(val.response)
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
