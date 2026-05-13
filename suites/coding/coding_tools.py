from typing import Annotated

from agentdojo.functions_runtime import Depends


def return_injection(inj: Annotated[dict[str, str], Depends("inj")]) -> dict[str, str]:
    """Returns the injection strings."""

    return inj
