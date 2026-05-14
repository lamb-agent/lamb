from dataclasses import dataclass
from typing import Annotated

from agentdojo.functions_runtime import Depends


class ToolError(Exception):
    pass


@dataclass
class Repo:
    name: str
    public: bool
    files: dict[str, str]


@dataclass
class Skill:
    name: str
    keywords: list[str]
    content: str

    def keyword_matches(self, keywords: list[str]) -> int:
        return len([kw for kw in keywords if kw in self.keywords])


@dataclass
class CodingClient:
    repos: list[Repo]
    active: Repo | None
    prs: list[tuple[str, str]]
    skills: list[Skill]

    def list_repos(self) -> list[str]:
        return [repo.name for repo in self.repos]


## Repo actions


def list_repos(
    client: Annotated[CodingClient, Depends("coding_client")],
) -> list[str]:
    return client.list_repos()


def init_repo(
    client: Annotated[CodingClient, Depends("coding_client")],
    name: str,
    public: bool,
) -> None:
    """Initialize a new repository with the given name. Decide whether it should be public or private.

    Automatically checks out the newly initialized repository.
    """  # noqa: E501

    if name in client.list_repos():
        raise ToolError("Repository with this name already exists.")
    repo = Repo(name=name, public=public, files={})
    client.active = repo
    return client.repos.append(repo)


def checkout_repo(
    client: Annotated[CodingClient, Depends("coding_client")],
    name: str,
) -> None:
    repos = [repo for repo in client.repos if repo.name == name]
    if len(repos) > 1:
        raise ToolError("Failed checkout. Repository does not exist.")
    client.active = repos[0]


def create_pr(
    client: Annotated[CodingClient, Depends("coding_client")],
    description: str,
) -> None:
    if client.active is None:
        raise ToolError("No repository checked out.")
    client.prs.append((client.active.name, description))


## Skill actions


def search_skills(
    client: Annotated[CodingClient, Depends("coding_client")],
    keywords: list[str],
) -> list[str]:
    matches = [(skill, skill.keyword_matches(keywords)) for skill in client.skills]
    matches.sort(key=lambda tup: tup[1], reverse=True)
    return [skill.name for (skill, count) in matches if count > 0]


def learn_skill(
    client: Annotated[CodingClient, Depends("coding_client")],
    name: str,
) -> str:
    skills = [skill for skill in client.skills if skill.name == name]
    if len(skills) > 1:
        raise ToolError("Failed to learn skill. Skill does not exist.")
    return skills[0].content


## Bash action


def bash(
    client: Annotated[CodingClient, Depends("coding_client")],
    command: str,
) -> str:
    return ""
