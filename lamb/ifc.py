from collections.abc import Callable
from os import stat
import typing
from enum import Enum
from functools import reduce
from typing import Self, assert_never

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


class IFCLabel(Enum):
    integ: Integrity
    conf: Confidentiality

    TL = (Integrity.TRUSTED, Confidentiality.LOW)
    TH = (Integrity.TRUSTED, Confidentiality.HIGH)
    UL = (Integrity.UNTRUSTED, Confidentiality.LOW)
    UH = (Integrity.UNTRUSTED, Confidentiality.HIGH)

    @staticmethod
    def bot() -> "IFCLabel":
        return IFCLabel.TL

    def __str__(self) -> str:
        """Short tuple representation."""

        match self:
            case IFCLabel.TL:
                return "TL"
            case IFCLabel.TH:
                return "TH"
            case IFCLabel.UL:
                return "UL"
            case IFCLabel.UH:
                return "UH"
            case _:
                assert_never(self)

    def __init__(self, integ: Integrity, conf: Confidentiality) -> None:
        self.integ = integ
        self.conf = conf

    def permitted_flow(self, sink: Self) -> bool:
        return self.integ.permitted_flow(sink.integ) and self.conf.permitted_flow(
            sink.conf
        )

    def join(self, other: Self) -> Self:
        return type(self)(
            self.integ.join(other.integ),
            self.conf.join(other.conf),
        )


def tools_for_model_context(context: IFCLabel) -> set[Callable]:
    match context:
        case IFCLabel.TL:
            return tool_categories.ALL
        case IFCLabel.TH:
            return tool_categories.HIGH_CONF_SINK  # TODO: add arg_conf
        case IFCLabel.UL:
            return tool_categories.READ_ONLY & tool_categories.UNTRUSTED_SINK
        case IFCLabel.UH:
            return (
                tool_categories.READ_ONLY
                & tool_categories.HIGH_CONF_SINK
                & tool_categories.UNTRUSTED_SINK
            )  # TODO: add arg_conf
        case _:
            assert_never(context)


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
    source = reduce(IFCLabel.join, variable_labels, IFCLabel.bot()).join(model_context)
    return source.permitted_flow(sink)


def permitted_tool_result(
    tool: typing.Callable,
    model_context: IFCLabel,
) -> bool:
    """Tool result check succeeds if the ingress using this tool is permitted."""

    source = tool_source_label(tool)
    sink = model_context
    return source.permitted_flow(sink)
