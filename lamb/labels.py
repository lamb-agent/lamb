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
HIGH_CONF_URLS = []


def check_whitelist(
    candidates: list[str],
    allowed: list[str],
) -> ifc.Confidentiality:
    """Return high conf if all canditates are on the whitelist."""

    return (
        ifc.Confidentiality.HIGH
        if all(candidate in allowed for candidate in candidates)
        else ifc.Confidentiality.LOW
    )


def check_domain(
    candidates: list[str],
    allowed: list[str],
) -> ifc.Confidentiality:
    """Return high conf if all canditates end with an allowed domain."""

    return (
        ifc.Confidentiality.HIGH
        if all(
            candidate.endswith(domain) for candidate in candidates for domain in allowed
        )
        else ifc.Confidentiality.LOW
    )


def check_url(
    candidates: list[str],
    allowed: list[str],
) -> ifc.Confidentiality:
    """Return high conf if all canditates start with an allowed domain."""

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
    recipient: str,
    **_: object,
) -> ifc.Confidentiality:
    return check_whitelist([recipient], HIGH_CONF_IBANS)


def check_event_participants(
    participants: list[str],
    **_: object,
) -> ifc.Confidentiality:
    return check_domain(participants, HIGH_CONF_DOMAINS)


def check_event_participants_in_event(
    env: CalendarEnv,
    event_id: str,
    **_: object,
) -> ifc.Confidentiality:
    return check_event_participants(env.calendar.events[event_id].participants)


def check_email(
    env: CloudDriveEnv,
    recipients: list[str],
    cc: list[str],
    bcc: list[str],
    attachments: list[dict[str, str]],
    **_: object,
) -> ifc.Confidentiality:
    file_ids = [
        val
        for attachment in attachments
        for key, val in attachment.items()
        if key == "file_id"
    ]
    file_reader_confs = min(
        check_file_readers(env, file_id).value for file_id in file_ids
    )
    domain_conf = check_domain([*recipients, *cc, *bcc], HIGH_CONF_DOMAINS).value
    return ifc.Confidentiality(min(file_reader_confs, domain_conf))


def check_file_readers(
    env: CloudDriveEnv,
    file_id: str,
    **_: object,
) -> ifc.Confidentiality:
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
}


@dataclass
class ADLabeler(ifc.Labeler):
    env: rt.TaskEnvironment

    def tool_source_label(
        self, tool: rt.Function, result: rt.FunctionReturnType
    ) -> ifc.IFCLabel:
        if tool.name in ["query_llm", "query_llm_structured"]:
            assert isinstance(result, query_llm.QueryLlmResponse)
            assert result.ifc_label, (
                "When IFC functions are called, IFC label must be set"
            )
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
                conf = ifc.Confidentiality.HIGH
            case fn if fn in tool_categories.ARG_CONF_SINK:
                conf = ARG_CONF_EVAL[fn](**all_args)
            case _:
                raise ValueError("Tool is not labeled with a sink type")
        return ifc.IFCLabel(integ, conf)
