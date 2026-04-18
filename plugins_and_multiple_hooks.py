from strands import Agent
from strands.agent import AgentResult
from strands.plugins import Plugin, hook
from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent
from strands_tools import calculator
import json

class LoggingPlugin(Plugin):
    name = "logging-plugin"

    @hook
    def log_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        print(f"[before tool call] Calling tool: {event.tool_use['name']}")

    @hook
    def log_after_tool_call(self, event: AfterToolCallEvent) -> None:
        print(f"[after tool call] Tool call completed: {event.tool_use['name']}")

agent = Agent(
    plugins=[LoggingPlugin()],
    callback_handler=None,
    tools=[calculator]
)

result: AgentResult = agent(prompt="What is the result of 123 * 456 / (23 + 4) * 8? Just give me the number, no more than that.")

print(json.dumps(result, indent=2, default=str))

print("--------------------------------\n\n")

print(f"Stop reason: {result.stop_reason}")
print(f"Message: {result.message}")
print(f"Metrics: {result.metrics}")
print(f"State: {result.state}")
print(f"Interrupts: {result.interrupts}")
print(f"Structured output: {result.structured_output}")
