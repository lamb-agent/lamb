import typing
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Self

from lamb import tool_categories


class IFCError(Exception):
    pass


class Integrity(Enum):  # noqa: PLW1641
    """Integrity is modelled as the Lattice L={TRUSTED,UNTRUSTED}.

    Information MUST NOT flow from an UNTRUSTED source to a TRUSTED sink.
    """

    TRUSTED = 0
    UNTRUSTED = 1

    def permitted_flow(self, sink: Self) -> bool:
        return self.value <= sink.value

    def join(self, other: Self) -> Self:
        return type(self)(max(self.value, other.value))


class Confidentiality(Enum):
    """Confidentiality is modelled as the Lattice L={LOW,HIGH}.

    Information MUST NOT flow from a HIGH source to a LOW sink.
    """

    LOW = 0
    HIGH = 1

    def permitted_flow(self, sink: Self) -> bool:
        return self.value <= sink.value

    def join(self, other: Self) -> Self:
        return type(self)(max(self.value, other.value))


@dataclass(frozen=True)
class IFCLabel:
    integ: Integrity
    conf: Confidentiality

    def permitted_flow(self, sink: Self) -> bool:
        return self.integ.permitted_flow(sink.integ) and self.conf.permitted_flow(
            sink.conf
        )

    def join(self, other: Self) -> Self:
        return type(self)(
            integ=self.integ.join(other.integ),
            conf=self.conf.join(other.conf),
        )


TOP = IFCLabel(integ=Integrity.UNTRUSTED, conf=Confidentiality.HIGH)
BOT = IFCLabel(integ=Integrity.TRUSTED, conf=Confidentiality.LOW)


def tool_sink_label(tool: typing.Callable) -> IFCLabel:
    conf = (
        Confidentiality.HIGH
        if tool in tool_categories.HIGH_CONF_SINK
        else Confidentiality.LOW  # Currently, .ARG_CONF is LOW conf
    )
    integ = (
        Integrity.TRUSTED
        if tool in tool_categories.TRUSTED_SINK
        else Integrity.UNTRUSTED
    )
    return IFCLabel(integ, conf)


def tool_source_label(tool: typing.Callable) -> IFCLabel:
    conf = (
        Confidentiality.HIGH
        if tool in tool_categories.HIGH_CONF_SOURCE
        else Confidentiality.LOW
    )
    integ = (
        Integrity.TRUSTED
        if tool in tool_categories.TRUSTED_SOURCE
        else Integrity.UNTRUSTED
    )
    return IFCLabel(integ, conf)


def permitted_tool_call(
    tool: typing.Callable,
    model_context: IFCLabel,
    variable_labels: set[IFCLabel],
) -> bool:
    """Tool call check succeeds if the egress using this tool is permitted."""

    sink = tool_sink_label(tool)
    source = reduce(IFCLabel.join, variable_labels, BOT).join(model_context)
    return source.permitted_flow(sink)


def permitted_tool_result(
    tool: typing.Callable,
    model_context: IFCLabel,
) -> bool:
    """Tool result check succeeds if the ingress using this tool is permitted."""

    source = tool_source_label(tool)
    sink = model_context
    return source.permitted_flow(sink)
