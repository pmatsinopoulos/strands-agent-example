from threading import Lock
from time import sleep
from typing import Any
from strands import Agent, tool
from strands.agent import AgentResult
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent, HookProvider, HookRegistry


class LimitToolCounts(HookProvider):
    """
    Limits the number of times tools can be called per agent invocation.
    """
    def __init__(
        self,
        max_tool_calls: dict[str, int]
    ):
        """
        Initializer.

        Args:
            max_tool_calls (dict[str, int]): A dictionary mapping tool names to the maximum number of times that tool can be called.
            If a tool is not specified in it, the tool can be called as many times as needed.
        """
        self.max_tool_calls = max_tool_calls
        self.tool_calls: dict[str, int] = {}
        self._lock = Lock()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(event_type=BeforeInvocationEvent, callback=self._reset_counts)
        registry.add_callback(event_type=BeforeToolCallEvent, callback=self._check_tool_call_limit)

    def _reset_counts(self, event: BeforeInvocationEvent) -> None:
        with self._lock:
            self.tool_calls.clear()

    def _check_tool_call_limit(self, event: BeforeToolCallEvent) -> None:
        with self._lock:
            tool_name = event.tool_use["name"]
            if tool_name not in self.max_tool_calls:
                return

            if tool_name not in self.tool_calls:
                self.tool_calls[tool_name] = 0

            if self.tool_calls[tool_name] > self.max_tool_calls[tool_name]:
                event.cancel_tool = (
                    f"Tool {tool_name} has been called {self.tool_calls[tool_name]} times, which exceeds the maximum allowed of {self.max_tool_calls[tool_name]}."
                )
                return

            self.tool_calls[tool_name] += 1


@tool
def sleep_it(seconds: int) -> None:
    """
    Sleep for a given number of seconds.
    """
    sleep(seconds)

agent = Agent(
    tools=[sleep_it],
    hooks=[
        LimitToolCounts(
            max_tool_calls={
                "sleep_it": 3
            }
        )
    ],
    system_prompt="""
    You can use the sleep_it tool to sleep for a given number of seconds.
    """
)

result: AgentResult = agent(prompt="I want you to sleep for 1 second, 2 seconds, 3 seconds and finally 4 seconds.")
