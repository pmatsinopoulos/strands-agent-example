import json
from strands import Agent
from strands.agent import AgentResult


agent = Agent(
    messages=[
        {
          "role": "user", "content": [
              {
                  "text": "Hello, my name is Panos."
              }
          ]
        },
        {
            "role": "assistant", "content": [
                {
                    "text": "Hi there! How can I help you today?"
                }
            ]
        }
    ]
)

result: AgentResult = agent(prompt="What's my name?")

print(json.dumps(result.metrics.get_summary(), indent=2, default=str))

print("--------------------------------")

print(json.dumps(agent.messages, indent=2, default=str))
