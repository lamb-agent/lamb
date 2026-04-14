from dataclasses import dataclass
import typing
from collections.abc import Callable
from enum import Enum
from functools import reduce
from typing import Protocol, Self, assert_never

import agentdojo.functions_runtime as rt

from lamb import types


class SystemAccess(Enum):
    """System access is modelled as the lattice L={PRIVILEGED,BOUNDED}.

    Tools that can potentially harm the system
    should be considered PRIVILEGED.
    """

    PRIVILEGED = 0
    BOUNDED = 1


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


type ToolFilter = SystemAccess | Confidentiality | Integrity

ALL_TOOLS: set[ToolFilter] = {
    SystemAccess.PRIVILEGED,
    SystemAccess.BOUNDED,
    Confidentiality.HIGH,
    Confidentiality.LOW,
    Integrity.TRUSTED,
    Integrity.UNTRUSTED,
}


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

    @staticmethod
    def from_str(string: str) -> "IFCLabel":
        match string:
            case "TL":
                return IFCLabel.TL
            case "TH":
                return IFCLabel.TH
            case "UL":
                return IFCLabel.UL
            case "UH":
                return IFCLabel.UH
            case errorLabel:
                raise ValueError(f"Unexpected IFCLabel: {errorLabel}")

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

    def set_conf(self, conf: Confidentiality) -> Self:
        return type(self)(self.integ, conf)

    def set_integ(self, integ: Integrity) -> Self:
        return type(self)(integ, self.conf)

    def is_high(self) -> bool:
        return self.conf == Confidentiality.HIGH


@dataclass
class IFCPolicy:
    conf: typing.Literal["hide", "taint"]
    integ: typing.Literal["hide", "taint"]
    violation: typing.Literal["error", "ignore"]

    @staticmethod
    def dynamic() -> "IFCPolicy":
        return IFCPolicy(conf="taint", integ="hide", violation="error")

    @staticmethod
    def static() -> "IFCPolicy":
        return IFCPolicy(conf="hide", integ="hide", violation="error")


class Labeler(Protocol):
    def tool_source_label(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
    ) -> IFCLabel: ...
    def tool_sink_label(
        self,
        tool: rt.Function,
        args: types.Args,
    ) -> IFCLabel: ...
    def filter_tools(
        self,
        tools: types.Tools,
        tool_filters: set[ToolFilter],
    ) -> types.Tools: ...


class IFCError(Exception):
    source: IFCLabel
    sink: IFCLabel

    def __init__(
        self,
        message: str,
        source: IFCLabel,
        sink: IFCLabel,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.sink = sink


class IFCChecker:
    model_context: IFCLabel
    var_labels: dict[str, IFCLabel]
    policy: IFCPolicy
    on_model_context_change: Callable[[IFCLabel], None]
    labeler: Labeler

    def __init__(
        self,
        model_context: IFCLabel,
        labeler: Labeler,
        policy: IFCPolicy = IFCPolicy(conf="hide", integ="hide", violation="error"),  # noqa: B008
        on_model_context_change: Callable[[IFCLabel], None] = lambda label: None,  # noqa: ARG005
    ) -> None:
        self.var_labels = {}
        self.model_context = model_context
        self.labeler = labeler
        self.policy = policy
        self.on_model_context_change = on_model_context_change

    def get_model_context(self) -> IFCLabel:
        return self.model_context

    def hide(
        self,
        tool: rt.Function,
        result: rt.FunctionReturnType,
        var: str,
    ) -> tuple[bool, IFCLabel, IFCLabel]:
        source = self.labeler.tool_source_label(tool, result)
        sink = self.model_context
        conf_permitted = source.conf.permitted_flow(sink.conf)
        integ_permitted = source.integ.permitted_flow(sink.integ)
        do_hide: bool
        match (conf_permitted, integ_permitted, self.policy):
            case (False, True, IFCPolicy(conf="taint")):
                self.model_context = self.model_context.set_conf(Confidentiality.HIGH)
                self.on_model_context_change(self.model_context)
                do_hide = False
            case (True, False, IFCPolicy(integ="taint")):
                self.model_context = self.model_context.set_integ(Integrity.UNTRUSTED)
                self.on_model_context_change(self.model_context)
                do_hide = False
            case (False, False, IFCPolicy(conf="taint", integ="taint")):
                self.model_context = self.model_context.set_conf(Confidentiality.HIGH)
                self.model_context = self.model_context.set_integ(Integrity.UNTRUSTED)
                self.on_model_context_change(self.model_context)
                do_hide = False
            case _:
                self.var_labels[var] = source
                do_hide = True
        return do_hide, source, self.model_context

    def check(
        self,
        tool: rt.Function,
        hidden_args: types.Args,
        expanded_args: types.Args,
    ) -> tuple[IFCLabel, IFCLabel]:
        sink = self.labeler.tool_sink_label(tool, expanded_args)
        total_source = self.model_context
        for name, arg in hidden_args.items():
            # we iterate over the arguments for a better error message,
            # we could just join all from the start
            variable_labels = {self.var_labels[var] for var in self.find_vars(arg)}
            source = reduce(IFCLabel.join, variable_labels, self.model_context)
            total_source = total_source.join(source)
            if not source.permitted_flow(sink) and self.policy.violation == "error":
                raise IFCError(
                    message=f"The argument `{name}` contains a variable that violates the IFC policies",  # noqa: E501
                    source=total_source,
                    sink=sink,
                )
            # TODO: should we log in case of not permitted but ignore,
            # to remember for the stats whether we would have had an ifc violation?
        return total_source, sink

    def find_vars(self, arg: types.FunctionCallArgTypes) -> set[str]:
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
            case list() if types.is_arg_list(arg):
                return reduce(set.union, (self.find_vars(item) for item in arg), set())
            case dict() if types.is_arg_dict(arg):
                return reduce(
                    set.union, (self.find_vars(item) for item in arg.values()), set()
                )
            case types.FunctionCall():
                return reduce(
                    set.union,
                    (self.find_vars(item) for item in arg.args.values()),
                    set(),
                )
            case _:
                assert_never(
                    typing.cast("typing.Never", arg)
                )  # static code analysis can't compute this

    def response_label(self, response: str) -> IFCLabel:
        variables = self.find_vars(response)
        labels = {self.var_labels[var] for var in variables}
        joined_label = reduce(IFCLabel.join, labels, self.model_context)
        return joined_label

    @staticmethod
    def none(labeler: Labeler) -> "IFCChecker":
        return IFCChecker(
            model_context=IFCLabel.bot(),
            labeler=labeler,
            policy=IFCPolicy(conf="taint", integ="taint", violation="ignore"),
        )

    @staticmethod
    def none_with_context(labeler: Labeler, context: IFCLabel) -> "IFCChecker":
        return IFCChecker(
            model_context=context,
            labeler=labeler,
            policy=IFCPolicy(conf="taint", integ="taint", violation="ignore"),
        )

    @staticmethod
    def vars_only(labeler: Labeler) -> "IFCChecker":
        return IFCChecker(
            model_context=IFCLabel.bot(),
            labeler=labeler,
            policy=IFCPolicy(conf="taint", integ="hide", violation="ignore"),
        )

    @staticmethod
    def top(labeler: Labeler) -> "IFCChecker":
        return IFCChecker(
            model_context=IFCLabel.top(),
            labeler=labeler,
            policy=IFCPolicy(conf="taint", integ="taint", violation="error"),
        )
