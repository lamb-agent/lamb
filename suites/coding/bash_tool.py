import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BashRequest:
    command: str
    files: Mapping[str, str]


@dataclass(frozen=True)
class BashResponse:
    output: str
    files: Mapping[str, str]


def run_bash(request: BashRequest) -> BashResponse:
    """Interfaces with the Node.js just-bash script via pipes."""
    try:
        bash_tool_path = Path(__file__).parent / "bash-tool"
        result = subprocess.run(
            ["npx", "ts-node", bash_tool_path / "main.ts"],
            input=json.dumps(asdict(request)),
            capture_output=True,
            text=True,
            check=True,
            cwd=bash_tool_path.resolve(),
        )

        data = json.loads(result.stdout)
        return BashResponse(**data)

    except subprocess.CalledProcessError as e:
        # Handle stderr from the Node process
        error_msg = e.stderr or e.stdout
        raise RuntimeError(f"Node script execution failed: {error_msg}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError("Failed to parse JSON response from Node script") from e
