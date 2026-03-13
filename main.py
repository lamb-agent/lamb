from pathlib import Path

from lamb import (
    agent,
    bench,
    llm,
)
from lamb.agent import Agent


def main() -> None:
    log_dir = Path(".log")
    model = llm.Llm.ollama_openai(llm.OllamaModel.GPT_OSS_120B)
    bench.benchmark(
        agent_loop=agent.ADAgentLoop(
            lambda runtime, env: Agent.lamb_no_ifc(model, runtime, env)
        ),
        model=model.model,
        agent="lamb-no-ifc",
        log_dir=log_dir,
        attack=None,
        suites=["slack"],
        user_tasks=[],
        injection_tasks=["injection_task_0", "injection_task_1"],
        suite_version="v1.2",
        n_repeats=1,
        force_rerun=False,
    )


if __name__ == "__main__":
    main()
