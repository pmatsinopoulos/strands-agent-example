from strands import tool
from strands.hooks import BeforeToolCallEvent
from strands import Agent
import logging
import json

def log_with_context(event: BeforeToolCallEvent) -> None:
    """Log tool invocations with context from invocation state."""

    print(f"[log with context]Invocation state: {event.invocation_state}")

    # Access invocation state from the event
    user_id = event.invocation_state.get("user_id", "unknown")
    session_id = event.invocation_state.get("session_id")

    # Access non-JSON serializable objects like database connections
    db_connection = event.invocation_state.get("database_connection")
    logger_instance = event.invocation_state.get("custom_logger")

    # Use custom logger if provided, otherwise use default
    logger = logger_instance if logger_instance else logging.getLogger(__name__)

    logger.info(
        f"User {user_id} in session {session_id} "
        f"invoking tool: {event.tool_use['name']} "
        f"with DB connection: {db_connection is not None}"
    )

@tool
def my_tool(data: str) -> str:
    return f"Processed data: {data}"

agent = Agent(
    tools=[my_tool],
    callback_handler=None
)
agent.add_hook(log_with_context)

# Execute with context including non-serializable objects
import sqlite3
custom_logger = logging.getLogger("custom")
db_conn = sqlite3.connect(":memory:")

result = agent(
    prompt="Process the data: 'Hello, world!'",
    invocation_state={
        "user_id": "user123",
        "session_id": "session456",
        "database_connection": db_conn,
        "custom_logger": custom_logger
    },
)

print(json.dumps(result, indent=2, default=str))
