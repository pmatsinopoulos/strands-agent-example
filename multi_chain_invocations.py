import json
from dataclasses import asdict
from strands import Agent
from strands.hooks import AfterInvocationEvent


MAX_ITERATIONS = 3
iterations = 0

async def iterative_refinement(event: AfterInvocationEvent):
    """Re-invoke the agent up to MAX_ITERATIONS times for iterative refinement."""
    global iterations
    if iterations < MAX_ITERATIONS and event.result:
        iterations += 1
        event.resume = f"Review your previous response and improve it. Iteration {iterations} of {MAX_ITERATIONS}."

agent = Agent()
agent.add_hook(event_type=AfterInvocationEvent, callback=iterative_refinement)

result = agent(prompt="Draft a haiku about programming")

print("\n--------------------------------\n")
print(json.dumps(asdict(result), indent=2, default=str))
print("\n--------------------------------\n")

print("\n")

print(json.dumps(agent.messages, indent=2, default=str))
