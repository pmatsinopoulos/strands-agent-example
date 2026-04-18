from strands_tools import calculator
from strands import Agent
import json

agent = Agent(
    tools=[calculator]
)

# Direct tool call with recording (default behavior)
agent.tool.calculator(expression="123 * 456")

# Direct too call without recording
agent.tool.calculator(expression="765 / 987", record_direct_tool_call=False)

print(json.dumps(agent.messages, indent=2, default=str))
