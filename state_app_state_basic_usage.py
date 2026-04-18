import json
from strands import Agent

# Create an agent with initial state
agent = Agent(
    state={
        "user_preferences": {
            "theme": "dark"
        },
        "session_count": 0
    }
)

# Access state values
theme = agent.state.get("user_preferences")
print(f"Current theme: {theme}")

# Set new state values
agent.state.set("last_action", "login")
agent.state.set("session_count", 1)

# Get entire state
all_state = agent.state.get()
print(json.dumps(all_state, indent=2, default=str))

agent.state.delete("last_action")
print(json.dumps(agent.state.get(), indent=2, default=str))
