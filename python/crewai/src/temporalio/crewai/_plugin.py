"""CrewAI plugin for running CrewAI Task executions as Temporal activities."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from crewai import Crew

from temporalio import activity
from temporalio.crewai._activity import install_patch, make_task_activity, register_task
from temporalio.crewai._interceptor import CrewAIInterceptor
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


class CrewAIPlugin(SimplePlugin):
    """CrewAI plugin for Temporal SDK.

    .. warning::
        This package is experimental and may change in future versions.
        Use with caution in production environments.

    This plugin runs each `CrewAI <https://github.com/crewAIInc/crewAI>`_
    ``Task``'s execution (the agent's full reasoning/tool-use loop for that
    task) as a Temporal Activity, giving your agent workflows durable
    execution, automatic retries, and timeouts.

    Workflows must call :func:`temporalio.crewai.kickoff` instead of
    ``Crew.kickoff()`` / ``Crew.akickoff()``: CrewAI's own kickoff does
    real I/O as part of its bookkeeping (telemetry, event-bus console
    output, checkpoint-storage file locking) that isn't safe to run
    inside a deterministic workflow, so :func:`kickoff` reimplements just
    the ``Process.sequential`` task loop -- the only process supported by
    this version.

    Each ``Task`` given to this plugin must already have an ``agent``
    assigned; dynamically assigning a different agent at kickoff time
    (e.g. a hierarchical process reassigning a task) isn't supported in
    this version and executes with the task's own ``agent``.

    Args:
        crews: Crews to make available to workflows, keyed by name.
            Workflows retrieve them with :func:`crew`. Every task in each
            crew is wrapped to run as a Temporal Activity.
        default_activity_options: Activity options applied to every
            wrapped task, overridable per-task via ``activity_options``.
        activity_options: Per-task activity options, keyed by the task's
            ``name`` (set via ``Task(name=...)``). Required for a task
            with no ``name`` if `default_activity_options`` doesn't cover
            it, since there's otherwise no way to address it.
    """

    def __init__(
        self,
        crews: dict[str, Crew] | None = None,
        default_activity_options: dict[str, Any] | None = None,
        activity_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the CrewAI plugin with crews to run as Temporal activities."""
        crews = crews or {}
        activities: list[Any] = []

        for crew_name, crew in crews.items():
            # CrewAI's telemetry/tracing lazily resolves a platform-specific
            # user-data file path (via `appdirs`) the first time a crew
            # kicks off. That path resolution isn't safe to run inside a
            # deterministic workflow (e.g. it touches pywin32 internals on
            # Windows), so this plugin always disables it for registered
            # crews.
            crew.tracing = False
            for task in crew.tasks:
                if task.agent is None:
                    raise ValueError(
                        f"Task {task.description!r} in crew {crew_name!r} has no "
                        f"agent assigned. Assign an agent before passing the crew "
                        f"to CrewAIPlugin."
                    )
                task_opts = dict(default_activity_options or {})
                if task.name is not None and task.name in (activity_options or {}):
                    task_opts.update((activity_options or {})[task.name])

                activity_name = f"{crew_name}.{task.name or task.id}"
                task_activity = activity.defn(name=activity_name)(
                    make_task_activity(task)
                )
                activities.append(task_activity)
                register_task(task, task_activity, task_opts)

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to the CrewAI plugin.")
            if isinstance(runner, SandboxedWorkflowRunner):
                return replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "crewai",
                        "litellm",
                        "rich",
                        "appdirs",
                    ),
                )
            return runner

        super().__init__(
            "crewai.CrewAIPlugin",
            activities=activities,
            workflow_runner=workflow_runner,
            interceptors=[CrewAIInterceptor(crews)],
        )
        install_patch()
