import asyncio
import json
from strands import Agent
from strands.multiagent import GraphBuilder
from strands_tools import calculator
from strands.hooks import BeforeNodeCallEvent, BeforeToolCallEvent
from sympy.solvers.ode.systems import NonlinearError

def my_callback(event: BeforeNodeCallEvent) -> None:
    print(f"[node] -> {event.node_id}")

def my_callback_2(event: BeforeToolCallEvent) -> None:
    print(f"[tool] -> {event.tool_use['name']}")

agent1 = Agent(
    name="searcher",
    callback_handler=None,
    system_prompt="""
    You are a lookup assistant. When asked about a location, distance, date,
    or other factual item, return ONLY the requested fact in one short sentence.
    Do not perform any arithmetic. Do not add explanations or alternatives.
    """,
)

agent2 = Agent(
    name="calculator",
    tools=[calculator],
    callback_handler=None,
    system_prompt="""
    You are a calculation assistant. You will receive a fact (e.g. a distance)
    from the previous agent and the user's original question.
    Use the calculator tool exactly once to compute the requested quantity,
    then state the result in one sentence. Do not provide alternative scenarios
    or extra commentary.
    """,
)
agent2.add_hook(my_callback_2)

builder = GraphBuilder()
builder.add_node(agent1, agent1.name)
builder.add_node(agent2, agent2.name)
builder.add_edge(agent1.name, agent2.name)
builder.set_max_node_executions(10)
builder.set_entry_point(agent1.name)

graph = builder.build()

graph.hooks.add_callback(event_type=None, callback=my_callback)

coroutine = graph.invoke_async(
    """
    I want you to tell how much it would take me to walk the
    distance from Thessaloniki to Athens if I walk at a speed of
    5 km/h. Return only the number of hours.
    """
)
print("PRINTING RESULT: \n\n--------------------------------\n\n")
final = asyncio.run(coroutine)
print(final.results["calculator"].result.message["content"][0]["text"])
print("--------------------------------\n\n")
