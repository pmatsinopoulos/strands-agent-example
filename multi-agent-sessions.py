from strands import Agent
from strands.multiagent import GraphBuilder
from strands.session.file_session_manager import FileSessionManager

# Create agents
agent1 = Agent(name="researcher")
agent2 = Agent(name="writer")

# Create a session manager to the graph
session_manager = FileSessionManager(
    session_id="multi-agent-session",
    storage_dir="./.sessions",
)

builder = GraphBuilder()
builder.add_node(agent1, "researcher")
builder.add_node(agent2, "writer")
builder.add_edge("researcher", "writer")
builder.set_session_manager(session_manager)
builder.set_max_node_executions(10) # cap total not runs
builder.set_execution_timeout(120.0)
builder.set_node_timeout(60.0)

graph = builder.build()

# Use the graph - all orchestrator state is persisted
result = graph("Research and write about AI")
