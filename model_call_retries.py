"""
Example: retrying model calls when a transient ServiceUnavailable error occurs.

The retry behaviour is implemented as a HookProvider that subscribes to
AfterModelCallEvent. When the model call raises an exception whose string
representation contains "ServiceUnavailable", we set ``event.retry = True`` to
ask Strands to discard the failed response and re-invoke the model. We use
exponential backoff between attempts and reset the counter once a request
either succeeds or finishes (BeforeInvocationEvent fires per-request).

To make the retry behaviour observable without depending on a real outage,
the example wraps a real BedrockModel in a small ``FlakyModel`` proxy that
raises a fake ServiceUnavailable exception on the first ``fail_first_n``
stream calls and then forwards to the underlying model.
"""

import asyncio
import logging
from collections.abc import AsyncIterable
from typing import Any

from strands import Agent
from strands.hooks import (
    AfterModelCallEvent,
    BeforeInvocationEvent,
    HookProvider,
    HookRegistry,
)
from strands.models import BedrockModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ServiceUnavailableError(Exception):
    """Stand-in for a transient ServiceUnavailable error from a model provider."""


class RetryOnServiceUnavailable(HookProvider):
    """Retry model calls when a ServiceUnavailable error is raised.

    Args:
        max_retries: Maximum number of retry attempts per agent invocation
            before giving up and letting the exception propagate.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_count = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._reset_counts)
        registry.add_callback(AfterModelCallEvent, self._handle_retry)

    def _reset_counts(self, event: BeforeInvocationEvent | None = None) -> None:
        self.retry_count = 0

    async def _handle_retry(self, event: AfterModelCallEvent) -> None:
        # Successful call - nothing to do; reset for safety.
        if event.exception is None:
            self._reset_counts()
            return

        if "ServiceUnavailable" not in str(event.exception):
            return

        if self.retry_count >= self.max_retries:
            logger.warning(
                "ServiceUnavailable persisted after %d retries; giving up.",
                self.retry_count,
            )
            return

        self.retry_count += 1
        delay = 2 ** self.retry_count
        logger.info(
            "ServiceUnavailable encountered (attempt %d/%d); retrying in %ds...",
            self.retry_count,
            self.max_retries,
            delay,
        )
        await asyncio.sleep(delay)
        event.retry = True


class FlakyModel(Model):
    """Wraps a real Model and fails the first ``fail_first_n`` stream calls.

    Used purely to demonstrate ``RetryOnServiceUnavailable``. Every other
    method (config, structured_output, etc.) is delegated to the wrapped model.
    """

    def __init__(self, inner: Model, fail_first_n: int = 2):
        self._inner = inner
        self._fail_first_n = fail_first_n
        self._call_count = 0

    @property
    def stateful(self) -> bool:
        return self._inner.stateful

    def update_config(self, **model_config: Any) -> None:
        self._inner.update_config(**model_config)

    def get_config(self) -> Any:
        return self._inner.get_config()

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.structured_output(*args, **kwargs)

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        self._call_count += 1
        if self._call_count <= self._fail_first_n:
            raise ServiceUnavailableError(
                f"ServiceUnavailable: simulated transient failure "
                f"(call #{self._call_count})"
            )

        async for event in self._inner.stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice=tool_choice,
            system_prompt_content=system_prompt_content,
            invocation_state=invocation_state,
            **kwargs,
        ):
            yield event


real_model = BedrockModel(
    model_id="apac.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="ap-southeast-2",
    temperature=0.5,
)
flaky_model = FlakyModel(inner=real_model, fail_first_n=2)

retry_hook = RetryOnServiceUnavailable(max_retries=3)

agent = Agent(
    model=flaky_model,
    hooks=[retry_hook],
    system_prompt="You are a concise, helpful assistant. Answer in one sentence.",
)

result = agent(prompt="What is the capital of France?")

print("\n--- Summary ---")
print(f"Model stream calls made: {flaky_model._call_count}")
print(f"Retries triggered:       {retry_hook.retry_count}")
print(f"Final answer:            {result.message}")
