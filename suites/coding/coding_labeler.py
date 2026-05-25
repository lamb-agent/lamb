import typing
from dataclasses import dataclass

from agentdojo import functions_runtime as rt

from lamb import ifc, query_llm, types
from suites.coding import coding_suite, coding_tools
from suites.coding.coding_suite import CodingEnv

TRUSTED_SOURCE: set[typing.Callable] = {
    coding_tools.init_repo,
    coding_tools.list_repos,
    coding_tools.checkout_repo,
    coding_tools.create_pr,
    coding_tools.delete_repo,
}

UNTRUSTED_SOURCE: set[typing.Callable] = {
    coding_tools.search_skills,
    coding_tools.learn_skill,
    coding_tools.bash,
}

TRUSTED_SINK: set[typing.Callable] = {
    coding_tools.init_repo,
    coding_tools.delete_repo,
}

UNTRUSTED_SINK: set[typing.Callable] = {
    coding_tools.create_pr,
    coding_tools.list_repos,
    coding_tools.checkout_repo,
    coding_tools.search_skills,
    coding_tools.learn_skill,
    coding_tools.bash,
}

HIGH_CONF_SOURCE: set[typing.Callable] = {
    coding_tools.list_repos,
}

LOW_CONF_SOURCE: set[typing.Callable] = {
    coding_tools.search_skills,
    coding_tools.learn_skill,
    coding_tools.init_repo,
    coding_tools.checkout_repo,
    coding_tools.create_pr,
    coding_tools.delete_repo,
}

ARG_CONF_SOURCE: set[typing.Callable] = {
    coding_tools.bash,
}

HIGH_CONF_SINK: set[typing.Callable] = {
    coding_tools.init_repo,
    coding_tools.delete_repo,
    coding_tools.list_repos,
    coding_tools.checkout_repo,
}

LOW_CONF_SINK: set[typing.Callable] = {
    coding_tools.search_skills,
    coding_tools.learn_skill,
}

ARG_CONF_SINK: set[typing.Callable] = {
    coding_tools.bash,
    coding_tools.create_pr,
}

PRIVILEGED: set[typing.Callable] = {
    coding_tools.init_repo,
    coding_tools.delete_repo,
    coding_tools.checkout_repo,
    coding_tools.create_pr,
}

BOUNDED: set[typing.Callable] = {
    coding_tools.list_repos,
    coding_tools.search_skills,
    coding_tools.learn_skill,
    coding_tools.bash,
}


def check_sink(
    env: coding_suite.CodingEnv,
) -> ifc.Confidentiality:
    if env.coding_client.active is None:
        return ifc.Confidentiality.HIGH
    return (
        ifc.Confidentiality.LOW
        if env.coding_client.active.public
        else ifc.Confidentiality.HIGH
    )


def check_source(
    env: coding_suite.CodingEnv,
) -> ifc.Confidentiality:
    if env.coding_client.active is None:
        return ifc.Confidentiality.LOW
    return (
        ifc.Confidentiality.LOW
        if env.coding_client.active.public
        else ifc.Confidentiality.HIGH
    )


@dataclass
class CodingLabeler(ifc.Labeler):
    env: CodingEnv

    def tool_source_label(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
    ) -> ifc.IFCLabel:
        if tool.name in ["query_llm", "query_llm_structured"]:
            assert isinstance(result, query_llm.QueryLlmResponse)
            return result.ifc_label

        integ = (
            ifc.Integrity.TRUSTED
            if tool.run in TRUSTED_SOURCE
            else ifc.Integrity.UNTRUSTED
        )
        conf: ifc.Confidentiality
        match tool.run:
            case fn if fn in HIGH_CONF_SOURCE:
                conf = ifc.Confidentiality.HIGH
            case fn if fn in LOW_CONF_SOURCE:
                conf = ifc.Confidentiality.LOW
            case fn if fn in ARG_CONF_SOURCE:
                conf = check_source(self.env)  # type: ignore
            case _:
                raise ValueError("Tool is not labeled with a sink type")
        return ifc.IFCLabel(integ, conf)

    def tool_sink_label(
        self,
        tool: rt.Function,
        args: types.Args,
    ) -> ifc.IFCLabel:
        if tool.name in ["query_llm", "query_llm_structured"]:
            # nested llms preserve IFC, so they are fine as a sink
            return ifc.IFCLabel.top()

        integ = (
            ifc.Integrity.TRUSTED
            if tool.run in TRUSTED_SINK
            else ifc.Integrity.UNTRUSTED
        )
        conf: ifc.Confidentiality
        match tool.run:
            case fn if fn in HIGH_CONF_SINK:
                conf = ifc.Confidentiality.HIGH
            case fn if fn in LOW_CONF_SINK:
                conf = ifc.Confidentiality.LOW
            case fn if fn in ARG_CONF_SINK:
                conf = check_sink(self.env)
            case _:
                raise ValueError("Tool is not labeled with a sink type")
        return ifc.IFCLabel(integ, conf)

    def filter_tools(
        self,
        tools: types.Tools,
        tool_filters: set[ifc.ToolFilter],
    ) -> types.Tools:
        conf_tools = ARG_CONF_SINK.union(
            LOW_CONF_SINK if ifc.Confidentiality.LOW in tool_filters else set()
        ).union(HIGH_CONF_SINK if ifc.Confidentiality.HIGH in tool_filters else set())
        integ_tools = (
            TRUSTED_SINK if ifc.Integrity.TRUSTED in tool_filters else set()
        ).union(UNTRUSTED_SINK if ifc.Integrity.UNTRUSTED in tool_filters else set())
        access_tools = (
            PRIVILEGED if ifc.SystemAccess.PRIVILEGED in tool_filters else set()
        ).union(BOUNDED if ifc.SystemAccess.BOUNDED in tool_filters else set())
        filtered_tools = conf_tools.intersection(integ_tools).intersection(access_tools)

        return [tool for tool in tools if tool.run in filtered_tools]
