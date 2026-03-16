# Copied from https://github.com/ethz-spylab/agentdojo/blob/5cea5891fa8e6b13c4299a94691e1ec64d445fcd/src/agentdojo/agent_pipeline/llms/openai_llm.py
import json
import typing
from collections.abc import Sequence

import openai
from agentdojo import functions_runtime as rt
from openai.types import chat as openai_chat
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params import FunctionDefinition
from tenacity import (
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from lamb import logging, types


def _tool_call_to_openai(
    tool_call: types.FunctionCall,
) -> openai_chat.ChatCompletionMessageToolCallParam:
    return openai_chat.ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args),
        },
    )


def _content_blocks_to_openai_content_blocks(
    message: types.ChatMessage,
) -> list[openai_chat.ChatCompletionContentPartTextParam]:
    return [
        openai_chat.ChatCompletionContentPartTextParam(
            type="text", text=message.content
        )
    ]


def _message_to_openai(
    message: types.ChatMessage,
) -> openai_chat.ChatCompletionMessageParam:
    match message:
        case types.UserMessage(role=role):
            return openai_chat.ChatCompletionUserMessageParam(
                role=role, content=_content_blocks_to_openai_content_blocks(message)
            )
        case types.AssistantMessage(role=role):
            if len(message.tool_calls) > 0:
                tool_calls = [
                    _tool_call_to_openai(tool_call)
                    for tool_call in message.tool_calls
                ]
                return openai_chat.ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=_content_blocks_to_openai_content_blocks(message),
                    tool_calls=tool_calls,
                )
            return openai_chat.ChatCompletionAssistantMessageParam(
                role="assistant",
                content=_content_blocks_to_openai_content_blocks(message),
            )
        case types.ToolMessage():
            return openai_chat.ChatCompletionToolMessageParam(
                content=message.error
                or _content_blocks_to_openai_content_blocks(message),
                tool_call_id=message.tool_call_id,
                role="tool",
                name=message.tool_call.function,  # type: ignore -- this is actually used, and is important!
            )
        case _:
            typing.assert_never(message)


def _openai_to_tool_call(
    tool_call: openai_chat.ChatCompletionMessageFunctionToolCall,
) -> types.FunctionCall:
    return types.FunctionCall(
        function=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
        id=tool_call.id,
    )


def _assistant_message_to_content(
    message: openai_chat.ChatCompletionMessage,
) -> str:
    return message.content or ""


def _openai_to_assistant_message(
    message: openai_chat.ChatCompletionMessage,
) -> types.AssistantMessage:
    if message.tool_calls is not None:
        tool_calls = [
            _openai_to_tool_call(tool_call)
            for tool_call in message.tool_calls
            if isinstance(tool_call, openai_chat.ChatCompletionMessageFunctionToolCall)
        ]
    else:
        tool_calls = []
    return types.AssistantMessage(
        content=_assistant_message_to_content(message),
        tool_calls=tool_calls,
    )


def _function_to_openai(f: rt.Function) -> openai_chat.ChatCompletionToolParam:
    function_definition = FunctionDefinition(
        name=f.name,
        description=f.description,
        parameters=f.parameters.model_json_schema(),
    )
    return openai_chat.ChatCompletionToolParam(
        type="function", function=function_definition
    )


def prompt(
    model: str,
    runtime: rt.FunctionsRuntime,
    messages: Sequence[types.ChatMessage],
    response_format: ResponseFormat,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    reasoning_effort: openai_chat.ChatCompletionReasoningEffort | None = None,
) -> types.ChatMessage:
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    openai_messages = [_message_to_openai(message) for message in messages]
    openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
    for i, attempt in enumerate(
        Retrying(
            wait=wait_random_exponential(multiplier=1, max=40),
            stop=stop_after_attempt(3),
            reraise=True,
            retry=retry_if_not_exception_type(
                (
                    openai.BadRequestError,
                    openai.UnprocessableEntityError,
                    openai.APIConnectionError,
                )
            ),
        )
    ):
        temp = temperature + i * 0.1
        if i > 0:
            logging.warning(f"Retrying with increased temperature of {temp}.")
        with attempt:
            completion = client.chat.completions.create(
                model=model,
                messages=openai_messages,
                response_format=response_format,
                tools=openai_tools,
                temperature=temp,
                reasoning_effort=reasoning_effort,
            )
            choice = completion.choices[0]

            output = _openai_to_assistant_message(choice.message)
            return output
    raise ValueError("For loop must throw or return.")
