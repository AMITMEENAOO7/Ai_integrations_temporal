# CrewAI Plugin for Temporal Python SDK

> Release stage: Pre-release.

This Temporal [Plugin](https://docs.temporal.io/develop/plugins-guide) runs [CrewAI](https://github.com/crewAIInc/crewAI) `Task` executions (the agent's full reasoning/tool-use loop for that task) as Temporal Activities, giving your agent crews durable execution, automatic retries, and timeouts.

## Installation

```sh
uv add temporalio-crewai
```

## Usage

```python
from crewai import Agent, Crew, Process, Task

from temporalio import workflow
from temporalio.crewai import CrewAIPlugin, crew, kickoff

agent_a = Agent(role="writer", goal="write", backstory="a writer", llm=my_llm)
agent_b = Agent(role="editor", goal="edit", backstory="an editor", llm=my_llm)
task_a = Task(name="write", description="Write about {topic}.", expected_output="A draft.", agent=agent_a)
task_b = Task(name="edit", description="Edit the draft.", expected_output="A final version.", agent=agent_b, context=[task_a])
my_crew = Crew(agents=[agent_a, agent_b], tasks=[task_a, task_b], process=Process.sequential)

plugin = CrewAIPlugin(crews={"my-crew": my_crew})


@workflow.defn
class MyWorkflow:
    def __init__(self) -> None:
        self.crew = crew("my-crew")

    @workflow.run
    async def run(self, topic: str) -> str:
        result = await kickoff(self.crew, inputs={"topic": topic})
        return result.raw
```

## Workflows must call `kickoff()`, not `Crew.kickoff()` / `Crew.akickoff()`

CrewAI's own kickoff does real I/O as part of its own bookkeeping on every call -- resolving a platform-specific telemetry path, emitting console output through its event bus, and locking/writing a checkpoint file for its task-output storage -- none of which is safe to run inside a deterministic Temporal workflow.

`temporalio.crewai.kickoff()` is a minimal reimplementation of just the `Process.sequential` task loop: it interpolates `inputs` into each task (CrewAI's own, side-effect-free string templating), builds each task's `context` from previous outputs the same way `Crew.kickoff()` does, and runs each task directly. Because it's a reimplementation rather than a call into `Crew.kickoff()`, it doesn't support everything that entry point does:

- Only `Process.sequential` is supported (not `Process.hierarchical`).
- `ConditionalTask` and tasks with `async_execution=True` aren't supported.
- Crew-level memory, planning, `output_log_file`, and token-usage bookkeeping aren't populated.
- `crew.tracing` is forced to `False` for every crew given to `CrewAIPlugin` (tracing's own telemetry path resolution isn't workflow-safe).

Everything *inside* a single task's execution -- tool use, structured (`output_pydantic`/`output_json`) output, guardrails, memory lookups, callbacks -- runs unmodified, because it executes inside the task's Activity, not inside the workflow.

## Task requirements

Every `Task` given to `CrewAIPlugin` must already have an `agent` assigned at definition time. Dynamically assigning a different agent at kickoff time (e.g. a hierarchical process reassigning a task) isn't supported.

## Activity options

```python
from datetime import timedelta
from temporalio.common import RetryPolicy
from temporalio.crewai import CrewAIPlugin

plugin = CrewAIPlugin(
    crews={"my-crew": my_crew},
    default_activity_options={
        "start_to_close_timeout": timedelta(seconds=60),
        "retry_policy": RetryPolicy(maximum_attempts=3),
    },
    # Per-task overrides, keyed by Task(name=...):
    activity_options={"write": {"start_to_close_timeout": timedelta(seconds=120)}},
)
```

## Python 3.14

`crewai` pulls in `chromadb`, whose `pydantic.v1` compatibility shim doesn't yet work on Python 3.14. This plugin's CI matrix is capped at 3.13 until that's fixed upstream.

## Running Tests

Install dependencies:

```sh
make sync
```

Run the test suite:

```sh
make test
```

Tests start a local Temporal dev server automatically -- no external server needed.
