"""Activity wrapper for running a CrewAI Task's execution as a Temporal activity.

CrewAI's own async executor runs its (synchronous) LLM-call loop via
``asyncio.to_thread``, which is unsafe inside a deterministic Temporal
workflow (real threads aren't reproducible on replay). So instead of
intercepting at the LLM-call level, this module wraps a whole ``Task``
execution -- the entire "agent reasons, calls the LLM, uses tools, produces
an output" unit -- as a single Temporal activity. The activity body calls
CrewAI's own (unpatched) execution method directly on the original ``Task``
object, so all of CrewAI's own behavior (guardrails, callbacks, output
files, events) keeps working exactly as if no plugin were involved, just
running on an activity worker instead of inline in the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from crewai import Task
from crewai.tasks.task_output import TaskOutput

from temporalio import workflow

# The real implementation, captured once at import time -- before any
# patching -- so the activity body always reaches CrewAI's actual logic,
# no matter how many times CrewAIPlugin instances install the patch below.
_ORIGINAL_AEXECUTE_CORE: Callable[..., Any] = Task._aexecute_core  # type: ignore[attr-defined]

# task id -> activity callable, registered by CrewAIPlugin at construction time.
_task_activities: dict[str, Callable[..., Any]] = {}
# task id -> kwargs for workflow.execute_activity (timeouts, retry policy, etc.)
_task_activity_options: dict[str, dict[str, Any]] = {}

_PATCH_INSTALLED = "_temporalio_crewai_patched"


@dataclass
class TaskActivityInput:
    """Input for a Task-execution activity."""

    context: str | None


@dataclass
class TaskActivityOutput:
    """Output from a Task-execution activity.

    ``task_output`` is ``TaskOutput.model_dump(mode="json")`` with the
    ``pydantic`` field removed; the workflow side reattaches it using the
    task's own ``output_pydantic`` class, which it already has.
    """

    task_output: dict[str, Any]


def make_task_activity(task: Task) -> Callable[[TaskActivityInput], Any]:
    """Build the Temporal activity body that runs ``task``'s real execution.

    The activity calls CrewAI's own, unpatched ``_aexecute_core`` directly
    on ``task`` -- the same live object the workflow-side code holds a
    reference to -- so guardrails, callbacks, output-file writing and
    event emission all run exactly as CrewAI intends, just inside the
    activity instead of inline in the workflow.
    """

    async def activity_fn(input: TaskActivityInput) -> TaskActivityOutput:
        output: TaskOutput = await _ORIGINAL_AEXECUTE_CORE(
            task, None, input.context, None
        )
        data = output.model_dump(mode="json")
        return TaskActivityOutput(task_output=data)

    return activity_fn


def register_task(
    task: Task,
    activity: Callable[..., Any],
    activity_options: dict[str, Any],
) -> None:
    """Register ``task`` so the patched ``_aexecute_core`` routes it through ``activity``."""
    key = str(task.id)
    _task_activities[key] = activity
    _task_activity_options[key] = activity_options


async def _patched_aexecute_core(
    self: Task, agent: Any, context: str | None, tools: Any
) -> TaskOutput:
    """Replacement for ``Task._aexecute_core`` installed by ``CrewAIPlugin``.

    Routes execution through this task's registered Temporal activity if
    one was registered; otherwise falls back to CrewAI's normal behavior
    unchanged (e.g. for a ``Task`` that wasn't given to ``CrewAIPlugin``).
    """
    key = str(self.id)
    activity = _task_activities.get(key)
    if activity is None:
        return await _ORIGINAL_AEXECUTE_CORE(self, agent, context, tools)

    opts = _task_activity_options.get(key, {})
    result: TaskActivityOutput = await workflow.execute_activity(
        activity, TaskActivityInput(context=context), **opts
    )
    data = dict(result.task_output)
    pydantic_payload = data.pop("pydantic", None)
    task_output = TaskOutput(**data)
    if pydantic_payload is not None and self.output_pydantic is not None:
        task_output.pydantic = self.output_pydantic.model_validate(pydantic_payload)
    self.output = task_output
    return task_output


def install_patch() -> None:
    """Install the ``Task._aexecute_core`` patch, once per process."""
    if getattr(Task._aexecute_core, _PATCH_INSTALLED, False):  # type: ignore[attr-defined]
        return
    setattr(_patched_aexecute_core, _PATCH_INSTALLED, True)
    Task._aexecute_core = _patched_aexecute_core  # type: ignore[method-assign]
