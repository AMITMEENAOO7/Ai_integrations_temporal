"""CrewAI plugin for Temporal SDK.

.. warning::
    This package is experimental and may change in future versions.
    Use with caution in production environments.

This plugin runs `CrewAI <https://github.com/crewAIInc/crewAI>`_ Task
executions as Temporal Activities, giving your AI agent crews durable
execution, automatic retries, and timeouts.

Installed from the ``temporalio-crewai`` distribution as ``temporalio.crewai``.
"""

from temporalio.crewai._interceptor import crew
from temporalio.crewai._orchestration import kickoff
from temporalio.crewai._plugin import CrewAIPlugin

__all__ = [
    "CrewAIPlugin",
    "crew",
    "kickoff",
]
