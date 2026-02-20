import typing
from collections.abc import Callable, Mapping
from enum import Enum
from functools import reduce
from typing import Self, assert_never

import agentdojo.functions_runtime as rt

from lamb import tool_categories


class IFCError(Exception):
    pass


class Integrity(Enum):
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

    def upgrade_conf(self) -> Self:
        return type(self)(self.integ, Confidentiality.HIGH)


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
    source = reduce(IFCLabel.join, variable_labels, model_context)
    return source.permitted_flow(sink)


def permitted_tool_result(
    tool: typing.Callable,
    model_context: IFCLabel,
) -> bool:
    """Tool result check succeeds if the ingress using this tool is permitted."""

    source = tool_source_label(tool)
    sink = model_context
    return source.permitted_flow(sink)


class SecretHandling(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"


class IFCChecker:
    model_context_label: IFCLabel
    var_labels: dict[str, IFCLabel]
    secret_handling: SecretHandling
    on_model_context_change: Callable[[IFCLabel], None]

    def __init__(
        self,
        model_context: IFCLabel,
        secret_handling: SecretHandling = SecretHandling.STATIC,
        on_model_context_change: Callable[[IFCLabel], None] = lambda label: None,  # noqa: ARG005
    ) -> None:
        self.var_labels = {}
        self.model_context_label = model_context
        self.secret_handling = secret_handling
        self.on_model_context_change = on_model_context_change

    def hide_result(self, tool: Callable) -> bool:
        return not permitted_tool_result(tool, self.model_context_label)

    def format_fn(self, tool: rt.Function, index: int) -> str:
        label = tool_source_label(tool.run)
        var = f"<{tool.name}_{index}:{label}/>"
        self.var_labels[var] = label
        return var

    def check(
        self,
        tool: rt.Function,
        args: Mapping[str, rt.FunctionCallArgTypes],
    ) -> None:
        def extract_vars(text: str) -> set[str]:
            return {var for var in self.var_labels if var in text}

        def find_vars(arg: rt.FunctionCallArgTypes) -> set[str]:
            match arg:
                case int() | float() | bool() | None:
                    return set()
                case str():
                    return extract_vars(arg)
                case list():
                    return reduce(set.union, (find_vars(item) for item in arg))
                case dict():
                    return reduce(set.union, (find_vars(item) for item in arg.values()))
                case rt.FunctionCall():
                    return reduce(
                        set.union, (find_vars(item) for item in arg.args.values())
                    )
                case _:
                    assert_never(arg)

        for name, arg in args.items():
            variable_labels = {self.var_labels[var] for var in find_vars(arg)}
            if not permitted_tool_call(
                tool.run, self.model_context_label, variable_labels
            ):
                if (
                    self.secret_handling == SecretHandling.DYNAMIC
                    and permitted_tool_call(
                        tool.run,
                        self.model_context_label.upgrade_conf(),
                        variable_labels,
                    )
                ):
                    self.model_context_label = self.model_context_label.upgrade_conf()
                    self.on_model_context_change(self.model_context_label)
                    continue

                raise IFCError(
                    f"The argument `{name}` contains a variable that violates the IFC policies"  # noqa: E501
                )
