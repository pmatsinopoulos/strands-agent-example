from typing import Any
from pydantic import ValidationError
from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.models import BedrockModel
import json

class InvalidExpressionError(Exception):
    """Raised when an invalid expression is provided to the calculator tool."""
    pass

@tool
def calculator(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.
    """
    raise InvalidExpressionError("This is a custom error")

class PropagateUnexpectedExceptions(HookProvider):
    """Re-raise unexpected exceptions instead of returning them to the model."""

    def __init__(self, allowed_exceptions: tuple[type[Exception],...]=(ValueError,)):
        self.allowed_exceptions = allowed_exceptions

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(event_type=AfterToolCallEvent, callback=self._check_exception)

    def _check_exception(self, event: AfterToolCallEvent) -> None:
        if event.exception is None:
            print(f"[check exception] Tool call succeeded: {event.tool_use['name']}")
            return # Tool succeeded
        if  isinstance(event.exception, self.allowed_exceptions):
            print(f"[check exception] Allowed exception: {event.exception}")
            return # Let model retry these
        print(f"[check exception] Unexpected exception: {event.exception}")
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
    hooks=[PropagateUnexpectedExceptions(
        allowed_exceptions=(ValueError, ValidationError)
    )],
    system_prompt="You are a concise, helpful assistant. Answer in one sentence."
)

result = agent(prompt="What is the result of 123 * 456 / (23 + 4) * 8?")

print("\n\n--------------------------------\n\n")
print(json.dumps(result, indent=2, default=str))
print("\n\n--------------------------------\n\n")
