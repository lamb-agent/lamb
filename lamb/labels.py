from collections.abc import Callable

from agentdojo import functions_runtime as rt

from lamb import ifc, query_llm, tool_categories


class ADLabeler(ifc.Labeler):
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
            if tool in tool_categories.TRUSTED_SOURCE
            else ifc.Integrity.UNTRUSTED
        )
        conf = (
            ifc.Confidentiality.HIGH
            if tool in tool_categories.HIGH_CONF_SOURCE
            else ifc.Confidentiality.LOW
        )
        return ifc.IFCLabel(integ, conf)

    def tool_sink_label(self, tool: Callable) -> ifc.IFCLabel:
        if tool not in tool_categories.ALL:
            # nested llms preserve IFC, so they are fine as a sink
            return ifc.IFCLabel.top()

        # TODO: ARG_CONF
        integ = (
            ifc.Integrity.TRUSTED
            if tool in tool_categories.TRUSTED_SINK
            else ifc.Integrity.UNTRUSTED
        )
        conf = (
            ifc.Confidentiality.HIGH
            if tool in tool_categories.HIGH_CONF_SINK
            else ifc.Confidentiality.LOW
        )
        return ifc.IFCLabel(integ, conf)
