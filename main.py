from pathlib import Path

import agentdojo.benchmark
import agentdojo.logging
from agentdojo import agent_pipeline as pipeline
from agentdojo import attacks, task_suite

from lamb import (
    agent,
    bench,  # noqa: F401
    llm,
    logging,
)
from lamb.agent import Agent


def main() -> None:
    suites = task_suite.get_suites("v1.2")
    # we don't actually use the AD logger,
    # but if we don't initialize it AD crashes
    log_dir = Path(".log")
    with agentdojo.logging.OutputLogger(str(log_dir), None):
        for suite_name, suite in suites.items():
            # if suite_name != "slack":
            #     continue
            logging.debug(f"Executing suite {suite_name}")
            lamb_pipeline = create_pipeline()
            attack = attacks.load_attack("tool_knowledge", suite, lamb_pipeline)
            # results = bench.benchmark(log_dir=log_dir, aggregate_by_commit=True)
            results = agentdojo.benchmark.benchmark_suite_with_injections(
                suite=suite,
                attack=attack,
                agent_pipeline=lamb_pipeline,
                logdir=log_dir,
                force_rerun=False,
                injection_tasks=[],
            )
            print(f"Utility of {suite_name}: {results['utility_results']}")
            print(f"Security of {suite_name}: {results['security_results']}")


def create_pipeline() -> pipeline.BasePipelineElement:
    model = llm.Llm.ollama_openai(llm.OllamaModel.GPT_OSS_120B)
    agent_loop = agent.ADAgentLoop(
        lambda runtime, env: Agent.lamb_no_ifc(
            model,
            runtime,
            env,
        )
    )
    lamb_pipeline = pipeline.AgentPipeline([agent_loop])
    # This is an inconsistency in AgentDojo.
    # The benchmark requires the attribute `name` to be set,
    # but the `AgentPipeline` class doesn't have it.
    lamb_pipeline.__setattr__("name", "local_lamb")
    return lamb_pipeline


if __name__ == "__main__":
    main()
