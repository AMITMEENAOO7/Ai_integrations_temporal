"""Minimal, side-effect-free replacement for ``Crew.akickoff()``.

``Crew.akickoff()`` performs real I/O as part of its own bookkeeping on
every call -- resolving a platform-specific telemetry path, emitting
console output through its event bus, and locking/writing a checkpoint
file for its task-output storage -- none of which is safe to run inside a
deterministic Temporal workflow. So instead of calling it, this module
reimplements just the ``Process.sequential`` task loop: interpolate
``inputs`` into each task (CrewAI's own, side-effect-free string
templating), build each task's context from previous outputs with
``Crew._get_context`` (a pure static method), and run each task's
execution directly -- which :class:`CrewAIPlugin` has already patched to
run as a Temporal activity.
"""

from __future__ import annotations

from typing import Any

from crewai import Crew, Process
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput


async def kickoff(crew: Crew, inputs: dict[str, Any] | None = None) -> TaskOutput:
    """Run ``crew``'s tasks in sequence, returning the final task's output.

    Only ``Process.sequential`` is supported, with every task running in
    the order given (no ``ConditionalTask``, no ``async_execution=True``):
    this is a minimal reimplementation of CrewAI's own sequential-process
    loop, not a call to ``Crew.kickoff()``, so it doesn't replicate those
    features. Every task in ``crew`` must have been registered with a
    :class:`CrewAIPlugin` so its execution runs as a Temporal activity.
    """
    if crew.process != Process.sequential:
        raise ValueError(
            f"temporalio.crewai.kickoff() only supports Process.sequential, "
            f"got {crew.process!r}."
        )
    for task in crew.tasks:
        if isinstance(task, ConditionalTask):
            raise ValueError(
                f"temporalio.crewai.kickoff() doesn't support ConditionalTask "
                f"(task {task.name or task.description!r})."
            )
        if task.async_execution:
            raise ValueError(
                f"temporalio.crewai.kickoff() doesn't support "
                f"async_execution=True (task {task.name or task.description!r})."
            )

    if inputs:
        for task in crew.tasks:
            task.interpolate_inputs_and_add_conversation_history(inputs)

    task_outputs: list[TaskOutput] = []
    for task in crew.tasks:
        context = Crew._get_context(task, task_outputs)
        output = await task._aexecute_core(task.agent, context, task.tools)  # type: ignore[attr-defined]
        task_outputs.append(output)

    return task_outputs[-1]
