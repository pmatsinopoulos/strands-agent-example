from strands import Agent
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent
from strands_tools import calculator

# Register individual callbacks
def my_callback(event: BeforeInvocationEvent) -> None:
    print(f"Before invocation callback triggered event: {event}")

# Type inference: If your callback has a type hint, the event type is inferred
def typed_callback(event: BeforeToolCallEvent) -> None:
    print(f"Tool called: {event.tool_use['name']}")


agent = Agent(
    system_prompt="""
    You are the best calculator in the world.
    You can use the calculator tool to calculate
    any mathematical expression.
    """,
    tools=[calculator]
)
agent.add_hook(my_callback, BeforeInvocationEvent)
agent.add_hook(typed_callback) # Event type inferred from type hint

result = agent(
    prompt="What is the result of 123 * 456 / (23 + 4) * 8?"
)
