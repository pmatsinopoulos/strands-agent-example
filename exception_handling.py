import asyncio
import dataclasses
from typing import Any
from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.models import BedrockModel
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class TransientToolError(Exception):
    """Raised when a tool call fails due to a transient error."""
    pass

class InvalidExpressionError(Exception):
    """Raised when an invalid expression is provided to the calculator tool."""
    pass

_flaky_calls = 0

@tool
def calculator(expression: str) -> str:
    global _flaky_calls

    if expression == "boom":
        raise RuntimeError("programmer error") # -> re-raise
    if expression == "flaky":
        _flaky_calls += 1
        if _flaky_calls <= 2:
            raise TransientToolError("transient tool error") # -> event.retry = True
        return "flaky-recovered"
    if not all(c in "0123456789+-*/(). " for c in expression):
        raise InvalidExpressionError(expression) # -> let model retry
    return str(eval(expression)) # success


class ToolExceptionPolicy(HookProvider):
    """Policy for handling tool exceptions."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_counts: dict[str, int] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(event_type=AfterToolCallEvent, callback=self._check_exception)

    async def _check_exception(self, event: AfterToolCallEvent) -> None:
        event_dict = {f.name: getattr(event, f.name) for f in dataclasses.fields(event)}
        logger.debug(f"[check exception] ****************************** Event:\n{json.dumps(event_dict, indent=2, default=str)}")

        if event.exception is None:
            logger.info(f"[check exception] Tool call succeeded: {event.tool_use['name']}")
            return # Tool succeeded

        if isinstance(event.exception, RuntimeError):
            event.result["content"][0]["text"] = "I'm sorry, this is a runtime error from the tool."
            return # Let this pass to the user as normal process

        if isinstance(event.exception, TransientToolError):
            tool_use_id = event.tool_use["toolUseId"]
            attempts = self.retry_counts.get(tool_use_id, 0)
            if attempts >= self.max_retries:
                logger.error(f"[check exception] Transient tool error persisted after {self.max_retries} retries; giving up.")
                event.result["content"][0]["text"] = (
                    f"Tool {event.tool_use['name']} failed after {self.max_retries} retries: "
                    f"{event.exception}"
                )
                return # Let this pass to the user as normal process

            self.retry_counts[tool_use_id] = attempts + 1
            delay = 2 ** (attempts + 1)
            logger.info(f"[check exception] Transient tool error encountered (attempt {attempts + 1}/{self.max_retries}); retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            event.retry = True
            return # Let model retry the tool call

        logger.error(f"[check exception] Unexpected exception: {event.exception}")
        raise event.exception # Propagate unexpected errors

model = BedrockModel(
    model_id="apac.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="ap-southeast-2",
    temperature=0.5,
)

# Usage
agent = Agent(
    model=model,
    tools=[calculator],
    hooks=[ToolExceptionPolicy()],
    system_prompt="You are a concise, helpful assistant. Answer in one sentence."
)

for case in ["123 * 456", "boom", "flaky", "abc"]:
    print(f"\n=== case: {case!r} ===")
    try:
        result = agent.tool.calculator(expression=case)
        logger.info(f"result: {result}")
    except Exception as e:
        print(f"escaped to caller: {type(e).__name__}: {e}")
