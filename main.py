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
        agent="lamb-no-ifc",
        log_dir=log_dir,
        attack="tool_knowledge",
        suites=None,
        user_tasks=["user_task_0", "user_task_1", "user_task_2"],
        injection_tasks=["injection_task_0", "injection_task_1", "injection_task_2"],
        suite_version="v1.2",
        n_repeats=1,
        force_rerun=False,
    )


if __name__ == "__main__":
    main()
