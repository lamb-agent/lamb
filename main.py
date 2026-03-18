from pathlib import Path

from lamb import (
    agent,
    bench,
    labels,
    llm,
)
from lamb.agent import Agent


def main() -> None:
    log_dir = Path(".log")
    model = llm.LiveLlm.ollama_openai(llm.OllamaModel.GPT_OSS_120B)
    bench.benchmark(
        agent_loop=agent.ADAgentLoop(
            lambda runtime, env: Agent.lamb_dynamic_ifc(
                model,
                runtime,
                env,
                labels.ADLabeler(env),
            )
        ),
        model=model.model,
        agent="local-lamb-dynamic-ifc",
        log_dir=log_dir,
        attack="tool_knowledge",
        suites=None,
        user_tasks=None,
        injection_tasks=None,
        suite_version="v1.2",
        n_repeats=1,
        force_rerun=False,
    )


if __name__ == "__main__":
    main()
