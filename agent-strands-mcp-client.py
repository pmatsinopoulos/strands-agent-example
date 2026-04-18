from strands import Agent
from strands.agent import AgentResult
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
import json


mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["strands-agents-mcp-server"]
        )
    )
)

agent = Agent(
    tools=[mcp_client]
)

result: AgentResult = agent(prompt="How do I create a custom tool in Strands Agents?")

print(json.dumps(result.metrics.get_summary(), indent=2, default=str))
