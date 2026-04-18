import json
from strands import Agent
from strands.agent import AgentResult


agent = Agent()

result: AgentResult = agent(prompt="Tell me about First World War. Only a sentence or two.")

print(json.dumps(result.metrics.get_summary(), indent=2, default=str))

print("--------------------------------")

# get history
print(json.dumps(agent.messages, indent=2, default=str))
