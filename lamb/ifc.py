import typing
from collections.abc import Callable, Mapping
from enum import Enum
from functools import reduce
from typing import Self, assert_never

import agentdojo.functions_runtime as rt


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
    def top() -> "IFCLabel":
        return IFCLabel.UH

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


class SecretHandling(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"


class IFCChecker:
    model_context: IFCLabel
    var_labels: dict[str, IFCLabel]
    secret_handling: SecretHandling
    on_model_context_change: Callable[[IFCLabel], None]
    tool_sink_label: Callable[[Callable], IFCLabel]
    tool_source_label: Callable[[Callable], IFCLabel]

    def __init__(
        self,
        model_context: IFCLabel,
        tool_sink_label: Callable[[Callable], IFCLabel],
        tool_source_label: Callable[[Callable], IFCLabel],
        secret_handling: SecretHandling = SecretHandling.STATIC,
        on_model_context_change: Callable[[IFCLabel], None] = lambda label: None,  # noqa: ARG005
    ) -> None:
        self.var_labels = {}
        self.model_context = model_context
        self.tool_sink_label = tool_sink_label
        self.tool_source_label = tool_source_label
        self.secret_handling = secret_handling
        self.on_model_context_change = on_model_context_change

    def permitted_tool_call(
        self,
        tool: typing.Callable,
        model_context: IFCLabel,
        variable_labels: set[IFCLabel],
    ) -> bool:
        """Tool call check succeeds if the egress using this tool is permitted."""

        sink = self.tool_sink_label(tool)
        source = reduce(IFCLabel.join, variable_labels, model_context)
        return source.permitted_flow(sink)

    def hide_result(
        self, tool: rt.Function, result: rt.FunctionReturnType, var: str
    ) -> bool:
        source = self.tool_source_label(tool.run)
        sink = self.model_context
        do_hide = not source.permitted_flow(sink)
        if do_hide:
            self.var_labels[var] = source
        return do_hide

    def check(
        self,
        tool: rt.Function,
        args: Mapping[str, rt.FunctionCallArgTypes],
    ) -> None:
        for name, arg in args.items():
            variable_labels = {self.var_labels[var] for var in self.find_vars(arg)}
            if not self.permitted_tool_call(
                tool.run, self.model_context, variable_labels
            ):
                if (
                    self.secret_handling == SecretHandling.DYNAMIC
                    and self.permitted_tool_call(
                        tool.run,
                        self.model_context.upgrade_conf(),
                        variable_labels,
                    )
                ):
                    # NOTE: For the single_llm upgrading the integrity
                    # might also be a valid use-case
                    self.model_context = self.model_context.upgrade_conf()
                    self.on_model_context_change(self.model_context)
                    continue

                raise IFCError(
                    f"The argument `{name}` contains a variable that violates the IFC policies"  # noqa: E501
                )

    def find_vars(self, arg: rt.FunctionCallArgTypes) -> set[str]:
        """Find a subset of variables from the given set of variables
        that is present in the given arg.
        """

        def extract_vars(text: str) -> set[str]:
            return {var for var in self.var_labels if var in text}

        match arg:
            case int() | float() | bool() | None:
                return set()
            case str():
                return extract_vars(arg)
            case list():
                return reduce(set.union, (self.find_vars(item) for item in arg))
            case dict():
                return reduce(
                    set.union, (self.find_vars(item) for item in arg.values())
                )
            case rt.FunctionCall():
                return reduce(
                    set.union, (self.find_vars(item) for item in arg.args.values())
                )
            case _:
                assert_never(arg)

    def response_label(self, response: str) -> IFCLabel:
        variables = self.find_vars(response)
        labels = {self.var_labels[var] for var in variables}
        joined_label = reduce(IFCLabel.join, labels, self.model_context)
        return joined_label
