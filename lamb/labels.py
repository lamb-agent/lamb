import typing
from dataclasses import dataclass

from agentdojo import functions_runtime as rt
from agentdojo.default_suites.v1.tools import (
    banking_client,
    calendar_client,
    cloud_drive_client,
    email_client,
    slack,
    web,
)

from lamb import ifc, query_llm, tool_categories, types


class CalendarEnv(typing.Protocol):
    calendar: calendar_client.Calendar


class CloudDriveEnv(typing.Protocol):
    cloud_drive: cloud_drive_client.CloudDrive


HIGH_CONF_IBANS = [
    "UK12345678901234567890",
    "GB29NWBK60161331926819",
    # "US133000000121212121212", # this is used both by the user and the attacker!?
]

HIGH_CONF_DOMAINS = ["@bluesparrowtech.com"]
HIGH_CONF_URLS = ["www.github.com"]


def check_whitelist(
    candidates: list[str],
    allowed: list[str],
) -> ifc.Confidentiality:
    """Return high conf if all candidates are on the whitelist."""

    return (
        ifc.Confidentiality.HIGH
        if all(candidate in allowed for candidate in candidates)
        else ifc.Confidentiality.LOW
    )


def check_domain(
    candidates: list[str],
    allowed: list[str],
) -> ifc.Confidentiality:
    """Return high conf if all candidates end with an allowed domain."""

    return (
        ifc.Confidentiality.HIGH
        if all(
            any(candidate.endswith(domain) for domain in allowed) for candidate in candidates
        )
        else ifc.Confidentiality.LOW
    )


def check_url(
    candidates: list[str],
    allowed: list[str],
) -> ifc.Confidentiality:
    """Return high conf if all candidates start with an allowed domain."""
    return (
        ifc.Confidentiality.HIGH
        if all(
            candidate.startswith(domain)
            for candidate in candidates
            for domain in allowed
        )
        else ifc.Confidentiality.LOW
    )


def check_banking_recipient(
    recipient: str | None = None,
    **_: object,
) -> ifc.Confidentiality:
    if recipient is None:
        return ifc.Confidentiality.HIGH
    return check_whitelist([recipient], HIGH_CONF_IBANS)


def check_event_participants(
    participants: list[str] | None = None,
    **_: object,
) -> ifc.Confidentiality:
    if participants is None:
        return ifc.Confidentiality.HIGH
    return check_domain(participants, HIGH_CONF_DOMAINS)


def check_event_participants_in_event(
    env: CalendarEnv,
    event_id: str,
    **_: object,
) -> ifc.Confidentiality:
    if event_id not in env.calendar.events:
        return ifc.Confidentiality.HIGH
    return check_event_participants(env.calendar.events[event_id].participants)


def check_email(
    env: CloudDriveEnv,
    recipients: list[str],
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    attachments: list[dict[str, str]] | None = None,
    **_: object,
) -> ifc.Confidentiality:
    file_ids = []
    if attachments is not None:
        file_ids = [
            val
            for attachment in attachments
            for key, val in attachment.items()
            if key == "file_id"
        ]
    file_reader_confs = min(
        (check_file_readers(env, file_id).value for file_id in file_ids),
        default=ifc.Confidentiality.HIGH.value,
    )
    all_recipients = [
        rec for recps in [recipients, cc or [], bcc or []] for rec in recps
    ]
    domain_conf = check_domain(all_recipients, HIGH_CONF_DOMAINS).value
    return ifc.Confidentiality(min(file_reader_confs, domain_conf))


def check_file_readers(
    env: CloudDriveEnv,
    file_id: str,
    **_: object,
) -> ifc.Confidentiality:
    if file_id not in env.cloud_drive.files:
        return ifc.Confidentiality.HIGH
    file = env.cloud_drive.files[file_id]
    return check_domain([file.owner, *file.shared_with.keys()], HIGH_CONF_DOMAINS)


def check_file_share(
    email: str,
    **_: object,
) -> ifc.Confidentiality:
    return check_domain([email], HIGH_CONF_DOMAINS)


def check_slack_invite(
    user_email: str,
    **_: object,
) -> ifc.Confidentiality:
    return check_domain([user_email], HIGH_CONF_DOMAINS)


def check_web_url(
    url: str,
    **_: object,
) -> ifc.Confidentiality:
    return check_url([url], HIGH_CONF_URLS)


ARG_CONF_EVAL = {
    banking_client.schedule_transaction: check_banking_recipient,
    banking_client.send_money: check_banking_recipient,
    banking_client.update_scheduled_transaction: check_banking_recipient,
    calendar_client.add_calendar_event_participants: check_event_participants,
    calendar_client.cancel_calendar_event: check_event_participants_in_event,
    calendar_client.create_calendar_event: check_event_participants,
    calendar_client.reschedule_calendar_event: check_event_participants_in_event,
    # cloud drive visibility depends on who the file is shared with
    cloud_drive_client.append_to_file: check_file_readers,
    cloud_drive_client.delete_file: check_file_readers,
    cloud_drive_client.share_file: check_file_share,
    email_client.send_email: check_email,
    slack.invite_user_to_slack: check_slack_invite,
    web.download_file: check_web_url,
    web.get_webpage: check_web_url,
    web.post_webpage: check_web_url,
}


@dataclass
class ADLabeler(ifc.Labeler):
    env: rt.TaskEnvironment

    def tool_source_label(
        self, tool: rt.Function, result: rt.FunctionReturnType
    ) -> ifc.IFCLabel:
        if tool.name in ["query_llm", "query_llm_structured"]:
            assert isinstance(result, query_llm.QueryLlmResponse)
            return result.ifc_label

        integ = (
            ifc.Integrity.TRUSTED
            if tool.run in tool_categories.TRUSTED_SOURCE
            else ifc.Integrity.UNTRUSTED
        )
        conf = (
            ifc.Confidentiality.HIGH
            if tool.run in tool_categories.HIGH_CONF_SOURCE
            else ifc.Confidentiality.LOW
        )
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
            if tool.run in tool_categories.TRUSTED_SINK
            else ifc.Integrity.UNTRUSTED
        )
        all_args = {"env": self.env, **args}
        conf: ifc.Confidentiality
        match tool.run:
            case fn if fn in tool_categories.HIGH_CONF_SINK:
                conf = ifc.Confidentiality.HIGH
            case fn if fn in tool_categories.LOW_CONF_SINK:
                conf = ifc.Confidentiality.LOW
            case fn if fn in tool_categories.ARG_CONF_SINK:
                conf = ARG_CONF_EVAL[fn](**all_args)  # type: ignore
            case _:
                raise ValueError("Tool is not labeled with a sink type")
        return ifc.IFCLabel(integ, conf)

    def filter_tools(
        self,
        tools: types.Tools,
        tool_filters: set[ifc.ToolFilter],
    ) -> types.Tools:
        conf_tools = tool_categories.ARG_CONF_SINK.union(
            tool_categories.HIGH_CONF_SINK
            if ifc.Confidentiality.HIGH in tool_filters
            else set()
        )
        integ_tools = (
            tool_categories.TRUSTED_SINK
            if ifc.Integrity.TRUSTED in tool_filters
            else set()
        ).union(
            tool_categories.UNTRUSTED_SINK
            if ifc.Integrity.UNTRUSTED in tool_filters
            else set()
        )
        access_tools = (
            tool_categories.STATE_CHANGING
            if ifc.SystemAccess.PRIVILEGED in tool_filters
            else set()
        ).union(
            tool_categories.READ_ONLY
            if ifc.SystemAccess.BOUNDED in tool_filters
            else set()
        )
        filtered_tools = conf_tools.intersection(integ_tools).intersection(access_tools)

        return [tool for tool in tools if tool.run in filtered_tools]
