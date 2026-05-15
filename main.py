from pathlib import Path

import typer
from agentdojo.task_suite import register_suite

from lamb import (
    bench,
    labels,
    llm,
)
from lamb.agent import AgentFn, AgentStr
from lamb.types import Suite
from suites.coding import coding_labeler, coding_suite


def main(
    agent: AgentStr,
    port: str = "11434",
    run: str | None = None,
    model: llm.OllamaModel = llm.OllamaModel.GEMMA4,
    log_dir: str = ".log",
    suites: list[Suite] = [Suite.WORKSPACE, Suite.BANKING, Suite.TRAVEL, Suite.SLACK],  # noqa: B006
    force_rerun: bool = False,  # noqa: FBT002
) -> None:
    register_suite(coding_suite.CODING_SUITE_V1, "v1.2")
    m = llm.LiveLlm.ollama_openai(model, port=port)
    agent_fn = AgentFn.new(
        agent,
        m,
        lambda suite, env: (
            coding_labeler.CodingLabeler(env)
            if suite == Suite.CODING
            else labels.ADLabeler(env)
        ),
    )
    bench.benchmark(
        agent_fn=agent_fn,
        model=m.model,
        agent=f"local-{agent.value}-{model.nickname()}{'' if not run else f'r{run}'}",
        log_dir=Path(log_dir),
        attack="tool_knowledge",
        suites=[suite.value for suite in suites],
        user_tasks=None,
        injection_tasks=None,
        suite_version="v1.2",
        n_repeats=1,
        force_rerun=force_rerun,
    )


if __name__ == "__main__":
    typer.run(main)
