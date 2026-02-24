from pathlib import Path
import agentdojo.logging
from agentdojo import agent_pipeline as pipeline
from agentdojo import benchmark as bench
from agentdojo import task_suite

from lamb import (
    agent,
    llm,
    logging,
)
from lamb.agent import Agent


def main() -> None:
    suites = task_suite.get_suites("v1.2")
    # we don't actually use the AD logger,
    # but if we don't initialize it AD crashes
    logdir = Path(".log")
    with agentdojo.logging.OutputLogger(str(logdir), None):
        for suite_name, suite in suites.items():
            logging.debug(f"Executing suite {suite_name}")
            lamb_pipeline = create_pipeline()
            results = bench.benchmark_suite_without_injections(
                suite=suite,
                agent_pipeline=lamb_pipeline,
                logdir=logdir,
                force_rerun=False,
                user_tasks=["user_task_0"],
            )
            utility = bench.aggregate_results([results["utility_results"]])
            print(f"Utility of {suite_name}: {utility}")


def create_pipeline() -> pipeline.BasePipelineElement:
    model = llm.Llm.local(llm.OllamaModel.GPT_OSS_120B, reasoning="medium")
    agent_loop = agent.ADAgentLoop(
        lambda runtime, env: Agent.lamb_no_ifc(model, runtime, env)
    )
    lamb_pipeline = pipeline.AgentPipeline([agent_loop])
    # This is an inconsistency in AgentDojo.
    # The benchmark requires the attribute `name` to be set,
    # but the `AgentPipeline` class doesn't have it.
    lamb_pipeline.__setattr__("name", "lamb")
    return lamb_pipeline


if __name__ == "__main__":
    main()
