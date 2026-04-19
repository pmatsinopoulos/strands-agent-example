from typing import Any
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry
from strands_tools import calculator
from strands import Agent
from strands.agent import AgentResult
import json

class ConstantToolArguments(HookProvider):
    def __init__(
        self,
        fixed_tool_arguments: dict[str, dict[str, Any]]
    ):
        """
        Initialize fixed parameter values for tools.

        Args:
            fixed_tool_arguments (dict[str, dict[str, Any]]): A dictionary mapping tool names
            to dictionaries of parameter names and their fixed values. These values will override
            any values provided by the agent when the tool is invoked.
        """
        self.fixed_tool_arguments = fixed_tool_arguments

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(event_type=BeforeToolCallEvent, callback=self._fix_tool_arguments)

    def _fix_tool_arguments(self, event: BeforeToolCallEvent) -> None:
        print(f"[fix tool arguments] Tool use: {event.tool_use}")
        tool_name: str = event.tool_use["name"]
        if tool_name not in self.fixed_tool_arguments:
            return

        fixed_args: dict[str, Any] = self.fixed_tool_arguments[tool_name]

        tool_input: dict[str, Any] = event.tool_use["input"]
        tool_input.update(fixed_args)

agent = Agent(
    tools=[calculator],
    hooks=[
        ConstantToolArguments(
            fixed_tool_arguments={
                "calculator": {
                    "precision": 4,
                    "mode": "evaluate",
                }
            }
        )
    ],
    system_prompt="""
    You are a helpful assistant that does calculations. Don't use your training knowledge to answer answer calculation questions.
    Use the calculator tool
    """,
)
result: AgentResult = agent(prompt="What is the result of 3.12341234 + 5.23452345? Answer only with the number and don't try to reason about how you got the answer. Trust the calculator and don't try to challenge its results.")

print(json.dumps(result, indent=2, default=str))
