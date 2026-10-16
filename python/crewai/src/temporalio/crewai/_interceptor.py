"""Workflow interceptor that scopes registered Crews to the workflow run."""

from __future__ import annotations

from typing import Any

from crewai import Crew

from temporalio import workflow
from temporalio.worker import (
    ExecuteWorkflowInput,
    Interceptor,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowOutboundInterceptor,
)

_workflow_crews: dict[str, dict[str, Crew]] = {}


class CrewAIInterceptor(Interceptor):
    """Interceptor that registers a workflow's crews for the run."""

    def __init__(self, crews: dict[str, Crew]) -> None:
        """Initialize with the crews to scope to each workflow run."""
        self._crews = crews

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor]:
        """Return the inbound interceptor class used to scope crews per run."""
        crews = self._crews

        class Inbound(WorkflowInboundInterceptor):
            def init(self, outbound: WorkflowOutboundInterceptor) -> None:
                run_id = outbound.info().run_id
                _workflow_crews[run_id] = crews
                super().init(outbound)

            async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
                try:
                    return await self.next.execute_workflow(input)
                finally:
                    _workflow_crews.pop(workflow.info().run_id, None)

        return Inbound


def crew(name: str) -> Crew:
    """Retrieve a registered ``Crew`` by name.

    Args:
        name: Crew name as registered with :class:`CrewAIPlugin`.
    """
    crews = _workflow_crews.get(workflow.info().run_id)
    if crews is None:
        raise RuntimeError(
            "crew() must be called from inside a workflow running under CrewAIPlugin"
        )
    if name not in crews:
        raise KeyError(f"Crew {name!r} not found. Available crews: {list(crews)}")
    return crews[name]
