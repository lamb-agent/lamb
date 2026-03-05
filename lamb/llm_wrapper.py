# Copied from https://github.com/ethz-spylab/agentdojo/blob/5cea5891fa8e6b13c4299a94691e1ec64d445fcd/src/agentdojo/agent_pipeline/llms/openai_llm.py
import json
from collections.abc import Sequence
from typing import overload

import openai
from agentdojo import functions_runtime as rt
from agentdojo import types as ad
from openai.types import chat as openai_chat
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params import FunctionDefinition
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


def _tool_call_to_openai(
    tool_call: rt.FunctionCall,
) -> openai_chat.ChatCompletionMessageToolCallParam:
    if tool_call.id is None:
        raise ValueError("`tool_call.id` is required for OpenAI")
    return openai_chat.ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args),
        },
    )


@overload
def _content_blocks_to_openai_content_blocks(
    message: ad.ChatUserMessage | ad.ChatSystemMessage,
) -> list[openai_chat.ChatCompletionContentPartTextParam]: ...


@overload
def _content_blocks_to_openai_content_blocks(
    message: ad.ChatAssistantMessage | ad.ChatToolResultMessage,
) -> list[openai_chat.ChatCompletionContentPartTextParam] | None: ...


def _content_blocks_to_openai_content_blocks(
    message: ad.ChatUserMessage
    | ad.ChatAssistantMessage
    | ad.ChatSystemMessage
    | ad.ChatToolResultMessage,
) -> list[openai_chat.ChatCompletionContentPartTextParam] | None:
    if message["content"] is None:
        return None
    return [
        openai_chat.ChatCompletionContentPartTextParam(
            type="text", text=el["content"] or ""
        )
        for el in message["content"]
    ]


def _message_to_openai(
    message: ad.ChatMessage,
) -> openai_chat.ChatCompletionMessageParam:
    match message["role"]:
        case "system":
            return openai_chat.ChatCompletionDeveloperMessageParam(
                role="developer",
                content=_content_blocks_to_openai_content_blocks(message),
            )
        case "user":
            return openai_chat.ChatCompletionUserMessageParam(
                role="user", content=_content_blocks_to_openai_content_blocks(message)
            )
        case "assistant":
            if message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
                tool_calls = [
                    _tool_call_to_openai(tool_call)
                    for tool_call in message["tool_calls"]
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
        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` should be specified for OpenAI.")
            return openai_chat.ChatCompletionToolMessageParam(
                content=message["error"]
                or _content_blocks_to_openai_content_blocks(message),
                tool_call_id=message["tool_call_id"],
                role="tool",
                name=message["tool_call"].function,  # type: ignore -- this is actually used, and is important!
            )
        case _:
            raise ValueError(f"Invalid message type: {message}")


def _openai_to_tool_call(
    tool_call: openai_chat.ChatCompletionMessageFunctionToolCall,
) -> rt.FunctionCall:
    return rt.FunctionCall(
        function=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
        id=tool_call.id,
    )


def _assistant_message_to_content(
    message: openai_chat.ChatCompletionMessage,
) -> list[ad.MessageContentBlock] | None:
    if message.content is None:
        return None
    return [ad.text_content_block_from_string(message.content)]


def _openai_to_assistant_message(
    message: openai_chat.ChatCompletionMessage,
) -> ad.ChatAssistantMessage:
    if message.tool_calls is not None:
        tool_calls = [
            _openai_to_tool_call(tool_call)
            for tool_call in message.tool_calls
            if isinstance(tool_call, openai_chat.ChatCompletionMessageFunctionToolCall)
        ]
    else:
        tool_calls = None
    return ad.ChatAssistantMessage(
        role="assistant",
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


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type(
        (openai.BadRequestError, openai.UnprocessableEntityError)
    ),
)
def prompt(
    model: str,
    runtime: rt.FunctionsRuntime,
    messages: Sequence[ad.ChatMessage],
    response_format: ResponseFormat,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = 0.0,
    reasoning_effort: openai_chat.ChatCompletionReasoningEffort | None = None,
) -> ad.ChatMessage:
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    openai_messages = [_message_to_openai(message) for message in messages]
    openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
    completion = client.chat.completions.create(
        model=model,
        messages=openai_messages,
        response_format=response_format,
        tools=openai_tools,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    choice = completion.choices[0]

    output = _openai_to_assistant_message(choice.message)
    return output
