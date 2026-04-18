from strands import Agent
from strands.session.file_session_manager import FileSessionManager

# Create a session manager with a unique session ID
session_manager = FileSessionManager(
    session_id="test-session",
    storage_dir="./.sessions",
)

# Create an agent with the session manager
agent = Agent(
    session_manager=session_manager
)

# Use the agent - all messages and state are automatically persisted
agent("What's the current foreign policy between these two countries today?") # This conversation is persisted
