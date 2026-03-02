# Copied from https://github.com/ethz-spylab/agentdojo/blob/5cea5891fa8e6b13c4299a94691e1ec64d445fcd/src/agentdojo/agent_pipeline/tool_execution.py
from ast import literal_eval
from dataclasses import dataclass

import agentdojo.functions_runtime as rt
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from lamb import ifc, query_llm, types
from lamb.runtime import Runtime


def is_string_list(s: str) -> bool:
    try:
        parsed = literal_eval(s)
        return isinstance(parsed, list)
    except (ValueError, SyntaxError):
        return False


@dataclass
class ToolExec:
    runtime: Runtime
    formatter: types.Formatter

    def exec(self, calls: list[rt.FunctionCall]) -> list[ChatMessage]:
        tool_call_results = []
        for tool_call in calls:
            if tool_call.function not in self.runtime.tools():
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Invalid tool {tool_call.function} provided.",
                    )
                )
                continue

            # Converts type of input lists from string to list type
            for arg_k, arg_v in tool_call.args.items():
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    tool_call.args[arg_k] = literal_eval(arg_v)

            tool = self.runtime.get_function(tool_call.function)
            args = self.formatter.expand(tool, tool_call.args)

            tool_call_result, error = self.runtime.exec(tool_call.function, args)
            history: list[ChatMessage] | None = None
            if error is None and tool.name in ["query_llm", "query_llm_structured"]:
                assert isinstance(tool_call_result, query_llm.QueryLlmResponse)
                history = tool_call_result.history
                tool_call_result = tool_call_result.response
            tool_call_id = tool_call.id
            try:
                formatted_tool_call_result = self.formatter.format(
                    tool,
                    tool_call_result,
                )
                result = ChatToolResultMessage(
                    role="tool",
                    content=[
                        text_content_block_from_string(formatted_tool_call_result)
                    ],
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    error=error,
                )
                result["history"] = history  # type: ignore
                tool_call_results.append(result)
            except ifc.IFCError as e:
                result = ChatToolResultMessage(
                    role="tool",
                    content=[],
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    error=str(e),
                )
                result["history"] = history  # type: ignore
                tool_call_results.append(result)

        return tool_call_results
