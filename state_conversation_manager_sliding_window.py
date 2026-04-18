from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands import Agent
from strands.agent import AgentResult
import json

agent = Agent(
    conversation_manager=SlidingWindowConversationManager(
        window_size=2, # Maximum number of messages to keep
    ),
    messages=[
        {
            "role": "user", "content": [
                {
                    "text": "You are a helpful assistant that can search internet."
                }
            ]
        },
        {
            "role": "assistant", "content": [
                {
                    "text": "Sure, how can I help you today?"
                }
            ]
        },
        {
            "role": "user", "content": [
                {
                    "text": "I want you to tell me when did 2nd World War finish."
                }
            ]
        },
        {
            "role": "assistant", "content": [
                {
                    "text": "The 2nd World War finished in September 2, 1945."
                }
            ]
        }
    ]
)

result: AgentResult = agent(prompt="What did 2nd World War start?")

print(json.dumps(agent.messages, indent=2, default=str))
