"""Analyse benchmark logs and report error counts per suite as CSV on stdout.

Usage:
    python -m scripts.error_metrics <root_log_folder>
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from lamb.bench import TaskResults

CATEGORIES = ("user", "injection", "attack")


@dataclass
class ErrorCounts:
    """Tracks timeout / connection / other error counts."""

    timeout: int = 0
    connection: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.timeout + self.connection + self.other

    def record(self, error: str) -> None:
        if "Request timed out" in error:
            self.timeout += 1
        elif "Connection error" in error:
            self.connection += 1
        else:
            self.other += 1

    @classmethod
    def header(cls, prefix: str) -> list[str]:
        return [
            f"{prefix}_errors",
            f"{prefix}_timeout",
            f"{prefix}_connection",
            f"{prefix}_other",
        ]

    def row(self) -> list[int]:
        return [self.total, self.timeout, self.connection, self.other]


CANCELLED_MARKER = "Unfortunately, I failed to fulfill your request"


def _last_message_text(task: TaskResults) -> str:
    """Extract the text content of the last message in a task."""
    if len(task.messages) == 0:
        return ""
    last_msg = task.messages[-1]
    # Messages have a 'content' field that is a list of {type, content} parts
    # content = getattr(last_msg, "content", None)
    if last_msg["content"] is None or len(last_msg["content"]) == 0:
        return ""
    content = last_msg["content"][0]["content"]
    return content


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_log_folder>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Find all benchmark run folders (sub-folders starting with "bench")
    bench_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("bench")
    )

    if not bench_dirs:
        print(
            f"No benchmark folders (starting with 'bench') found in '{root}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    writer = csv.writer(sys.stdout)
    header = ["benchmark_run", "suite", "total_tasks"]
    for prefix in ("total", *CATEGORIES):
        header.extend(ErrorCounts.header(prefix))
    for prefix in ("total", *CATEGORIES):
        header.append(f"{prefix}_cancelled")
    writer.writerow(header)

    for bench_dir in bench_dirs:
        # Each sub-folder of the benchmark run is a suite
        suite_dirs = sorted(d for d in bench_dir.iterdir() if d.is_dir())

        for suite_dir in suite_dirs:
            task_files = sorted(
                f
                for f in suite_dir.glob("*.json")
                if f.stem.startswith(("user-", "injection-", "attack-"))
            )
            total_tasks = 0
            counts: dict[str, ErrorCounts] = {
                cat: ErrorCounts() for cat in ("total", *CATEGORIES)
            }
            cancelled: dict[str, int] = {
                cat: 0 for cat in ("total", *CATEGORIES)
            }

            for task_file in task_files:
                try:
                    with task_file.open() as f:
                        data = json.load(f)
                    task = TaskResults(**data)
                except Exception as e:
                    print(
                        f"Warning: skipping '{task_file}': {e}",
                        file=sys.stderr,
                    )
                    continue

                total_tasks += 1
                if task.error is not None:
                    counts["total"].record(task.error)
                    for cat in CATEGORIES:
                        if task_file.stem.startswith(f"{cat}-"):
                            counts[cat].record(task.error)
                            break
                elif not task.utility:
                    # No error and utility is False → check for early cancellation
                    if CANCELLED_MARKER in _last_message_text(task):
                        cancelled["total"] += 1
                        for cat in CATEGORIES:
                            if task_file.stem.startswith(f"{cat}-"):
                                cancelled[cat] += 1
                                break

            row: list[object] = [bench_dir.name, suite_dir.name, total_tasks]
            for prefix in ("total", *CATEGORIES):
                row.extend(counts[prefix].row())
            for prefix in ("total", *CATEGORIES):
                row.append(cancelled[prefix])
            writer.writerow(row)


if __name__ == "__main__":
    main()

