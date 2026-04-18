from strands import Agent
import json

def custom_callback_handler(**kwargs):
    # Access request state
    if "request_state" in kwargs:
        state = kwargs["request_state"]
        # Use or modify state as needed
        if "counter" not  in state:
            state["counter"] = 0
        state["counter"] += 1
        print(f"Callback handler event count: {state["counter"]}")

agent = Agent(
    callback_handler=custom_callback_handler
)

result = agent(prompt="Hi there!")

print("--------------------------------")
print(json.dumps(result.state, indent=2, default=str))

result =agent(prompt="hello again!")
print("--------------------------------")
print(json.dumps(result.state, indent=2, default=str))
