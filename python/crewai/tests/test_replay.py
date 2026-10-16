import warnings
from datetime import timedelta
from uuid import uuid4

from crewai import Agent, Crew, Process, Task

from temporalio.client import Client
from temporalio.crewai import CrewAIPlugin
from temporalio.worker import Replayer, Worker
from tests.test_two_tasks import FakeLLM, TwoTasksWorkflow


async def test_replay(client: Client):
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
    )

    task_queue = f"my-crew-replay-{uuid4()}"
    plugin = CrewAIPlugin(
        crews={"my-crew": my_crew},
        default_activity_options={"start_to_close_timeout": timedelta(seconds=10)},
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[TwoTasksWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            TwoTasksWorkflow.run,
            "ducks",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()

    with warnings.catch_warnings(record=True) as recorder:
        warnings.filterwarnings(
            "always", message=r"Module .* was imported after initial workflow load"
        )
        await Replayer(
            workflows=[TwoTasksWorkflow],
            plugins=[plugin],
        ).replay_workflow(await handle.fetch_history())

    # Sandbox imports during an activation count toward the deadlock timeout
    assert not [
        str(w.message)
        for w in recorder
        if "was imported after initial workflow load" in str(w.message)
    ]
