import threading
import time

from strands import Agent
from strands.agent import AgentResult


def timeout_watchdog(agent: Agent, timeout: float) -> None:
    """Cancel the agent after a timeout."""
    time.sleep(timeout)
    agent.cancel()

agent = Agent()

# Cancel from a background thread after 30 seconds
watchdog = threading.Thread(
    target=timeout_watchdog,
    args=(agent, 30.0)
)
watchdog.start()

result: AgentResult = agent(prompt="Analyze this large dataset")
watchdog.join()

if result.stop_reason == "cancelled":
    print("Agent was cancelled due to timeout")
