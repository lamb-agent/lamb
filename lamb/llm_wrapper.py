import logging
import sys
import time
from collections.abc import Sequence

import agentdojo.agent_pipeline as pipeline
import agentdojo.functions_runtime as rt
import agentdojo.types as ad_types
import google.genai
import google.genai.errors
import openai

from lamb.prompting_llm import PromptingLLM


class LocalLLM(pipeline.BasePipelineElement):
    def __init__(self, model: str):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )
        self.llm = PromptingLLM(client, model)

    def query(
        self,
        query: str,
        runtime: rt.FunctionsRuntime,
        env: rt.Env = rt.EmptyEnv(),  # ty:ignore[invalid-parameter-default]
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, rt.FunctionsRuntime, rt.Env, Sequence[ad_types.ChatMessage], dict]:
        try:
            return self.llm.query(query, runtime, env, messages, extra_args)
        except openai.APIConnectionError:
            logging.error("Ollama must be running")
            sys.exit(1)


class GoogleLLM(pipeline.BasePipelineElement):
    def __init__(self, model: str, api_key: str):
        client = google.genai.Client(api_key=api_key)
        self.llm = pipeline.GoogleLLM(model=model, client=client)

    def query(
        self,
        query: str,
        runtime: rt.FunctionsRuntime,
        env: rt.Env = rt.EmptyEnv(),  # ty:ignore[invalid-parameter-default]
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, rt.FunctionsRuntime, rt.Env, Sequence[ad_types.ChatMessage], dict]:
        try:
            return self.llm.query(query, runtime, env, messages, extra_args)
        except google.genai.errors.ClientError as err:
            match err:
                case google.genai.errors.ClientError(code=429):
                    logging.error("Google API exhausted, waiting...")
                    time.sleep(30)
                    return self.query(query, runtime, env, messages, extra_args)
                case _:
                    logging.error(f"Google API error: {err}")
                    sys.exit(1)


class Gemma(pipeline.BasePipelineElement):
    def __init__(self, api_key: str):
        client = openai.OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )
        self.llm = PromptingLLM(client=client, model="gemma-3-27b-it")

    def query(
        self,
        query: str,
        runtime: rt.FunctionsRuntime,
        env: rt.Env = rt.EmptyEnv(),  # ty:ignore[invalid-parameter-default]
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, rt.FunctionsRuntime, rt.Env, Sequence[ad_types.ChatMessage], dict]:
        try:
            return self.llm.query(query, runtime, env, messages, extra_args)
        except openai.RateLimitError:
            logging.error("Google API exhausted, waiting...")
            time.sleep(30)
            return self.query(query, runtime, env, messages, extra_args)
