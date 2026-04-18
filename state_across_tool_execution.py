from strands import Agent, ToolContext, tool

@tool(context=True)
def track_user_action(action: str, tool_context: ToolContext) -> str:
    """
    Track user actions in agent state.

    Args:
        action (str): The action to track.
        tool_context (ToolContext): The tool context. This is used to get access
        to the agent state.
    """
    # Get current action acount
    action_count = tool_context.agent.state.get("action_count") or 0

    # Update state
    tool_context.agent.state.set("action_count", action_count + 1)
    tool_context.agent.state.set("last_action", action)

    return f"Action '{action}' recorded. Total actions: {action_count}"

@tool(context=True)
def get_user_stats(tool_context: ToolContext) -> str:
    """
    Get user statistics from agent state.
    """
    action_count = tool_context.agent.state.get("action_count") or 0
    last_action = tool_context.agent.state.get("last_action") or "none"

    return f"Actions performed: {action_count}. Last action: {last_action}"


agent = Agent(
    tools=[track_user_action, get_user_stats]
)

# Use tools that modify and read state

agent("Track that I logged in")
agent("Track that I viewed my profile")
print(f"Actions taken: {agent.state.get('action_count')}")
print(f"Last action: {agent.state.get('last_action')}")
