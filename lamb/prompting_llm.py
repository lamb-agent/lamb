# Copied from https://github.com/ethz-spylab/agentdojo/blob/5cea5891fa8e6b13c4299a94691e1ec64d445fcd/src/agentdojo/agent_pipeline/llms/prompting_llm.py
import json
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Callable, assert_never

from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _message_to_openai,
    _tool_call_to_openai,
    chat_completion_request,
)
from agentdojo.ast_utils import (
    ASTParsingError,
    create_python_function_from_tool_call,
    parse_tool_calls_from_python_function,
)
import agentdojo.functions_runtime as rt
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from lamb.types import State


class InvalidModelOutputError(Exception): ...


def _message_to_together(
    message: ChatMessage, model_name: str
) -> ChatCompletionMessageParam:
    match message["role"]:
        case "system":
            return ChatCompletionDeveloperMessageParam(
                role="developer", content=get_text_content_as_str(message["content"])
            )
        case "user":
            return ChatCompletionUserMessageParam(
                role="user", content=get_text_content_as_str(message["content"])
            )
        case "assistant":
            if message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
                tool_calls = [
                    _tool_call_to_openai(tool_call)
                    for tool_call in message["tool_calls"]
                ]
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=get_text_content_as_str(message["content"] or []),
                    tool_calls=[],  # not allowed with gemma, TODO: add flag to toggle
                )
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=get_text_content_as_str(message["content"] or []),
            )
        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` should be specified for OpenAI.")
            return ChatCompletionToolMessageParam(
                content=message["error"] or get_text_content_as_str(message["content"]),
                tool_call_id=message["tool_call_id"],
                role="tool",
                name=message["tool_call"].function,  # type: ignore -- this is actually used, and is important!
            )
        case _:
            raise ValueError(f"Invalid message type: {message}")


class PromptingLLM:
    """Function calling for Llama 3 based on [`hamelsmu`](https://github.com/hamelsmu)'s
    [prompt](https://github.com/hamelsmu/replicate-examples/blob/master/cog-vllm-tools/predict.py)."""

    llm: OpenAILLM
    """LLM to use"""

    _MAX_ATTEMPTS = 3
    """Maximum number of attempts at parsing the model output."""

    _tool_calling_prompt = """You are a helpful assistant. You are given a task and a set of possible functions inside <function-definitions> tags.
Calling these functions are optional. Carefully consider the question and determine if one or more functions can be used to answer the question. Place your thoughts and reasoning behind your decision in <function-thoughts> tags.
If the given question lacks the parameters required by the function, point it out in <function-thoughts> tags. Below is a list of function definitions:
<function-definitions>
{funcs}
</function-definitions>

If you wish to call a particular function, specify the name of the function and any arguments in a way that conforms to that function's schema inside <function-call> tags.
Function calls should be in this format: <function-thoughts>Calling func1 would be helpful because of ...</function-thoughts>\
<function-call>[function1(a="1", b=3), function2(foo=["a", "b"], bar=None)]</function-call>, WITHOUT any answer. Pass the arguments in correct format, i.e., \
strings should be enclosed in quotes, lists should be enclosed in square brackets, integers should have no quotes, etc. \
If you do not wish to call any functions, say so in the <function-thoughts> tags followed by <function-call>[]</function-call><answer>...</answer>
If no tools are provided, act as if no tools are available to you, but still provide <function-call>[]</function-call> as part of your output.

If and only if NO function calls are made, answer the question to the best of your ability inside <answer> tags.  If you are unsure of the answer, say so in <answer> tags.

The user will provide the output of the function calls in the <function-result> tags. The function call that output the given result is provided in <function-call> tags. Give your reply based on the output of the function calls. If no function calls are made, the user will provide the answer without any tags.
If the tool returned an error, then the user will provide the error message in <function-error> tags. The function call that returned the error is provided in <function-call> tags. If the tool returns an error, call the tool again with the correct parameters. If the tool returns an error again, without asking the user. You might need to call other tools to solve the task. Be persistent!

Sometimes, you may need to call multiple functions in multiple rounds to get the desired output, based on the results of the previous function calls. When this is the case, then use the <function-thoughts> tags to explain your reasoning for calling the next function, and the <function-call> tags to specify the next function call. \

Once you receive the result of one tool, think if you need to call another tool in the <function-thoughts> tags. If you do, then call the next tool. If you are done and do not need to call another tool, then do explicitly give the answer in <answer> tags. Tell the answer to the user in the message if you don't need other tools.

If you think you will need to call multiple tools in multiple stages, but you don't have yet the information to call all of them, then wait to receive the information back from the tools you can already call."""
    """The system prompt that instructs the LLM on how to use the tools."""

    def __init__(self, llm: OpenAILLM) -> None:
        self.llm = llm

    def _get_system_message(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[ChatSystemMessage | None, Sequence[ChatMessage]]:
        if len(messages) == 0:
            return None, []
        if messages[0]["role"] == "system":
            return messages[0], messages[1:]
        return None, messages

    def _make_tools_prompt(
        self, system_message: ChatSystemMessage | None, tools: Sequence[rt.Function]
    ) -> ChatMessage | None:
        """It creates a system message by instructing the model on how to use tools and
        provides the tools' documentation, placed within the <function-definitions> tags.
        The tool calling prompt is stored in the
        [`_tool_calling_prompt`][agentdojo.agent_pipeline.PromptingLLM._tool_calling_prompt] attribute.

        It preserves the original system message,  by placing it at the end of the tool calling prompt.

        Args:
            system_message: The system message to be adapted.

        Returns:
            The adapted system message.
        """
        if len(tools) == 0:
            return system_message
        tools_docs = ""
        system_prompt = (
            get_text_content_as_str(system_message["content"])
            if system_message is not None
            else ""
        )
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_json_schema(),
            }
            tools_docs += f"<{tool.name}>\n"
            tools_docs += json.dumps(tool_dict, indent=4)
            tools_docs += f"\n</{tool.name}>\n\n"
        tool_calling_prompt = self._tool_calling_prompt.format(funcs=tools_docs)
        message_content = f"{tool_calling_prompt}\n{system_prompt}"
        return ChatUserMessage(
            role="user", content=[text_content_block_from_string(message_content)]
        )

    def _parse_model_output(
        self,
        message: ChatCompletionMessage,
        expand_vars: Callable[[rt.FunctionCallArgTypes], rt.FunctionCallArgTypes],
    ) -> ChatAssistantMessage:
        """Parses the model output by extracting text and/or tool call contents from the message.

        It looks for the function call content within the `<function-call>` tags and extracts it. Each
        function call is expected to look like a python function call with parameters specified by name.
        For example, calling the function `func1` with parameters `a=1` and `b=3` would look like:

            <function-call>func1(a=1, b=3)</function-call>

        Content related to the LLM's thoughts are expected to be in the `<function-thoughts>` tags and are
        returned as part of the assistant message's `content`.

        If no function call is done, the answer is expected to be in the `<answer>` tags.

        Args:
            message: The model output message in OpenAI format.

        Returns:
            The assistant message with the extracted text and tool calls.
        """
        if message.content is None:
            return ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("")],
                tool_calls=None,
            )
        tool_call_pattern = re.compile(
            r"<function-call>(.*?)</function-call>", re.DOTALL
        )
        tool_call_match = tool_call_pattern.search(message.content)

        # Extract the function call content
        tool_call_content = tool_call_match.group(1) if tool_call_match else "[]"
        # Remove the function call section from the original text
        outside_content = (
            re.sub(
                r"<function-call>.*?</function-call>",
                "",
                message.content,
                flags=re.DOTALL,
            )
            .replace("<function-thoughts>", "")
            .replace("</function-thoughts>", "")
            .strip()
        )
        outside_content = re.sub(r"\n\s*\n", "\n\n", outside_content)
        tool_calls = parse_tool_calls_from_python_function(tool_call_content)
        for tool_call in tool_calls:
            args = {
                arg_name: ("..." if arg_value == Ellipsis else expand_vars(arg_value))
                for arg_name, arg_value in tool_call.args.items()
            }
            tool_call.args = args
        if len(tool_calls) == 0:
            answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
            answer_match = answer_pattern.search(outside_content)
            if answer_match is None:
                raise InvalidModelOutputError(
                    "The answer should be in <answer> tags if no tool calls are provided."
                )
            outside_content = answer_match.group(1)

        return ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(outside_content)],
            tool_calls=tool_calls,
        )

    def _tool_message_to_user_message(
        self, tool_message: ChatToolResultMessage
    ) -> ChatUserMessage:
        """It places the output of the tool call in the <function-result> tags and the tool call in the <function-call> tags.
        If an error is returned, it places the error in the <function-error> tags and the tool call in the <function-call> tags.""
        """
        function_call_signature = create_python_function_from_tool_call(
            tool_message["tool_call"]
        )
        function_call = f"<function-call>{function_call_signature}</function-call>"
        if tool_message["error"] is None:
            tool_result = f"<function-result>{get_text_content_as_str(tool_message['content'])}</function-result>"
        else:
            tool_result = f"<function-error>{tool_message['error']}</function-error>"
        return ChatUserMessage(
            role="user",
            content=[text_content_block_from_string(f"{function_call}{tool_result}")],
        )

    def next(
        self,
        state: State,
    ) -> State:
        adapted_messages = [
            self._tool_message_to_user_message(message)
            if message["role"] == "tool"
            else message
            for message in state.messages
        ]
        # TODO: this is specific to Together's API. Look into improving it.
        openai_messages = [
            _message_to_together(message, self.llm.model)
            for message in adapted_messages
            if message["role"] != "system"
        ]
        system_message, other_messages = self._get_system_message(state.messages)
        system_message = self._make_tools_prompt(
            system_message, list(state.runtime.functions.values())
        )
        if system_message is not None:
            converted_system_message = ChatUserMessage(
                role="user",
                # TODO: this is specific to Together's API. Look into improving it.
                content=get_text_content_as_str(system_message["content"] or []),  # type: ignore -- This is correct for TogetherAI's API
            )
            openai_messages = [converted_system_message, *openai_messages]
        completion = chat_completion_request(
            self.llm.client,
            self.llm.model,
            openai_messages,
            [],
            None,
            self.llm.temperature,
        )
        output = ChatAssistantMessage(
            role="assistant",
            content=[
                text_content_block_from_string(
                    completion.choices[0].message.content or ""
                )
            ],
            tool_calls=[],
        )
        if len(state.runtime.functions) == 0 or "<function-call>" not in (
            get_text_content_as_str(output["content"] or [])
        ):
            return replace(state, messages=[*state.messages, output])
        for _ in range(self._MAX_ATTEMPTS):
            try:
                output = self._parse_model_output(
                    completion.choices[0].message,
                    state.extra_args["tool_result_formatter"].expand,
                )
                break
            except (InvalidModelOutputError, ASTParsingError) as e:
                error_message = ChatUserMessage(
                    role="user",
                    content=[
                        text_content_block_from_string(
                            f"Invalid function calling output: {e!s}"
                        )
                    ],
                )
                completion = chat_completion_request(
                    self.llm.client,
                    self.llm.model,
                    [
                        *openai_messages,
                        _message_to_openai(error_message, self.llm.model),
                    ],
                    [],
                    None,
                    self.llm.temperature,
                )
        return replace(state, messages=[*state.messages, output])
