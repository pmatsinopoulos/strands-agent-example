from strands import Agent, tool
from strands.agent import AgentResult
from strands.hooks import AfterToolCallEvent
import json

@tool
def calculator(expression: str) -> str:
    return eval(expression)


agent = Agent(
    tools=[calculator],
    callback_handler=None,
    system_prompt="""
    You are a clever assistant that can use the calculator to carry out mathematical calculations.
    """
)

def after_tool_callback_event(event: AfterToolCallEvent) -> None:
    print(f"[after tool callback event] Tool use: {event.tool_use}")
    if event.tool_use["name"] == "calculator":
        # Add formatting to calculator results
        original_content = event.result["content"][0]["text"]
        event.result["content"][0]["text"] = f"Calculation Result: {original_content}"

agent.add_hook(after_tool_callback_event)

result: AgentResult = agent(
    prompt="What is the result of 123 * 456 / (23 + 4) * 8? Give me only the number, in a human readable format."
)

print(json.dumps([m for m in agent.messages if m["role"] == "user" and "toolResult" in m["content"][0]], indent=2, default=str))
print("--------------------------------\n")
print(json.dumps(result, indent=2, default=str))
