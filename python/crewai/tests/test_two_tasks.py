from datetime import timedelta
from typing import Any
from uuid import uuid4

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.llms.base_llm import BaseLLM
from crewai.tools.base_tool import BaseTool
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel

from temporalio import workflow
from temporalio.client import Client
from temporalio.crewai import CrewAIPlugin, crew, kickoff
from temporalio.worker import Worker


class FakeLLM(BaseLLM):
    """Deterministic stand-in LLM: always answers immediately, no tool calls."""

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        return f"{self.additional_params['reply']} (from {self.model})"


@workflow.defn
class TwoTasksWorkflow:
    def __init__(self) -> None:
        self.crew = crew("my-crew")

    @workflow.run
    async def run(self, input: str) -> Any:
        result = await kickoff(self.crew, inputs={"topic": input})
        return result.raw


async def test_two_tasks(client: Client):
    agent_a = Agent(
        role="writer",
        goal="write",
        backstory="a writer",
        llm=FakeLLM(model="fake-a", additional_params={"reply": "draft"}),
    )
    agent_b = Agent(
        role="editor",
        goal="edit",
        backstory="an editor",
        llm=FakeLLM(model="fake-b", additional_params={"reply": "final"}),
    )
    task_a = Task(
        name="write",
        description="Write about {topic}.",
        expected_output="A draft.",
        agent=agent_a,
    )
    task_b = Task(
        name="edit",
        description="Edit the draft.",
        expected_output="A final version.",
        agent=agent_b,
        context=[task_a],
    )
    my_crew = Crew(
        agents=[agent_a, agent_b],
        tasks=[task_a, task_b],
        process=Process.sequential,
        tracing=False,
    )

    task_queue = f"my-crew-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[TwoTasksWorkflow],
        plugins=[
            CrewAIPlugin(
                crews={"my-crew": my_crew},
                default_activity_options={
                    "start_to_close_timeout": timedelta(seconds=10)
                },
            )
        ],
    ):
        result = await client.execute_workflow(
            TwoTasksWorkflow.run,
            "ducks",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
        )

    assert result == "final (from fake-b)"
