import json
import logging
from dataclasses import asdict
from strands import Agent
from strands.hooks import AfterInvocationEvent
from strands_tools import http_request

logging.basicConfig(
  level=logging.INFO,
  format="%(levelname)s | %(name)s | %(message)s",
  handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

resume_count = 0

async def summarize_after_tools(event: AfterInvocationEvent):
    """Resume once to ask the model to summarize its work."""
    global resume_count

    if resume_count == 0 and event.result and event.result.stop_reason == "end_turn":
        resume_count += 1
        logger.info(f"[summarize after tools] Resuming to ask the model to summarize its work. (resume_count: {resume_count})")
        event.resume = "Now summarize what you just did in one sentence."


agent = Agent(
  tools=[http_request],
  callback_handler=None
)

agent.add_hook(event_type=AfterInvocationEvent, callback=summarize_after_tools)


# The agent processes the initial request, then automatically
# performs a second invocation to generate the summary.
result = agent(prompt="Look up the weather in Seattle")

print("\n--------------------------------\n")
print(json.dumps(asdict(result), indent=2, default=str))
print("\n--------------------------------\n")

print("\n")

print(json.dumps(agent.messages, indent=2, default=str))
