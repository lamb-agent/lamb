import typing
from pathlib import Path

import typer

from lamb import (
    bench,
    llm,
)
from lamb.agent import ADAgentLoop, AgentFn, AgentStr


def main(
    agent: str,
    port: str = "11434",
    run: str | None = None,
    model: llm.OllamaModel = llm.OllamaModel.GEMMA4,
    log_dir: str = ".log",
) -> None:
    m = llm.LiveLlm.ollama_openai(model, port=port)
    assert agent in typing.get_args(AgentStr.__value__)
    agent = typing.cast("AgentStr", agent)
    agent_fn = AgentFn.new(agent, m)
    bench.benchmark(
        agent_loop=ADAgentLoop(agent_fn),
        model=m.model,
        agent=f"local-{agent}-{model.nickname()}{'' if not run else f'r{run}'}",
        log_dir=Path(log_dir),
        attack="tool_knowledge",
        suites=None,
        user_tasks=None,
        injection_tasks=None,
        suite_version="v1.2",
        n_repeats=1,
        force_rerun=False,
    )


if __name__ == "__main__":
    typer.run(main)
