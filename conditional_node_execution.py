"""Conditional node execution in a Strands Graph.

This example shows TWO complementary mechanisms for skipping nodes:

1) Conditional edges via ``GraphBuilder.add_edge(..., condition=...)``.
   This is the deterministic, first-class way: when the predicate returns
   False the destination node is *not* scheduled and the graph keeps
   running normally.

2) The ``BeforeNodeCallEvent`` hook (the snippet from the docs). Hooks
   cannot truly "skip" a node, but they can *cancel* it via
   ``event.cancel_node`` (which marks it FAILED and stops the graph).
   The original docs example only printed a message; here we wire it up
   so you can actually observe the cancellation.

The graph we build:

                         +-----------+
                         | classifier|
                         +-----+-----+
                               |
              "math"  /        |        \\  "text"
                     v         |         v
              +-------------+  |  +----------------+
              | math_solver |  |  | text_analyzer  |
              +-------------+  |  +----------------+

The classifier emits either the word ``MATH`` or ``TEXT``. Conditional
edges then route the work to exactly one downstream node; the other one
is skipped cleanly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from strands import Agent
from strands.hooks import BeforeNodeCallEvent, HookProvider, HookRegistry
from strands.models import BedrockModel
from strands.multiagent import GraphBuilder
from strands.multiagent.graph import GraphState

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("strands").setLevel(logging.WARNING)
log = logging.getLogger("conditional_demo")


# ---------------------------------------------------------------------------
# 1. Hook-based "skip" (the snippet from the Strands documentation)
# ---------------------------------------------------------------------------
class ConditionalNodeExecution(HookProvider):
    """Cancel a node before it runs if a user-supplied predicate is True.

    NOTE: ``BeforeNodeCallEvent`` does not have a real "skip" capability.
    Setting ``event.cancel_node`` cancels the node with status FAILED and
    stops the whole graph. For genuine skipping use conditional edges
    (see ``build_graph`` below).
    """

    def __init__(self, skip_conditions: dict[str, Callable[[dict[str, Any]], bool]]):
        self.skip_conditions = skip_conditions

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(
            event_type=BeforeNodeCallEvent,
            callback=self.check_execution_conditions,
        )

    def check_execution_conditions(self, event: BeforeNodeCallEvent) -> None:
        node_id = event.node_id
        if node_id not in self.skip_conditions:
            return

        condition_func = self.skip_conditions[node_id]
        if condition_func(event.invocation_state or {}):
            log.info("Hook cancelling node %r based on invocation_state", node_id)
            event.cancel_node = f"cancelled by ConditionalNodeExecution hook: {node_id}"


# ---------------------------------------------------------------------------
# 2. Build a 3-node graph with conditional edges (the recommended approach)
# ---------------------------------------------------------------------------
def make_model() -> BedrockModel:
    return BedrockModel(
        model_id="apac.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="ap-southeast-2",
        temperature=0.0,
    )


def _last_text_from_node(state: GraphState, node_id: str) -> str:
    """Return the textual output of a completed node (or '' if missing)."""
    node_result = state.results.get(node_id)
    if node_result is None:
        return ""
    return str(node_result.result).strip()


def classified_as(label: str) -> Callable[[GraphState], bool]:
    """Build a condition that fires when the classifier output equals ``label``."""

    def predicate(state: GraphState) -> bool:
        classification = _last_text_from_node(state, "classifier").upper()
        decision = label.upper() in classification
        log.info(
            "Edge condition: classifier said %r -> route to %r? %s",
            classification,
            label,
            decision,
        )
        return decision

    return predicate


def build_graph(extra_hooks: list[HookProvider] | None = None):
    model = make_model()

    classifier = Agent(
        name="classifier",
        model=model,
        system_prompt=(
            "You are a router. Read the user's request and answer with a "
            "SINGLE word, either MATH or TEXT. MATH means the request is a "
            "calculation or numeric question. TEXT means anything else. "
            "Respond with just the word, nothing else."
        ),
        callback_handler=None,
    )

    math_solver = Agent(
        name="math_solver",
        model=model,
        system_prompt=(
            "You are a careful mathematician. Solve the math problem from the "
            "original task and return only the numeric answer with a brief "
            "one-sentence explanation."
        ),
        callback_handler=None,
    )

    text_analyzer = Agent(
        name="text_analyzer",
        model=model,
        system_prompt=(
            "You are a writing assistant. Summarise the original task in one "
            "concise sentence."
        ),
        callback_handler=None,
    )

    builder = GraphBuilder()
    builder.add_node(classifier, node_id="classifier")
    builder.add_node(math_solver, node_id="math_solver")
    builder.add_node(text_analyzer, node_id="text_analyzer")

    builder.add_edge("classifier", "math_solver", condition=classified_as("MATH"))
    builder.add_edge("classifier", "text_analyzer", condition=classified_as("TEXT"))

    builder.set_entry_point("classifier")

    if extra_hooks:
        builder.set_hook_providers(extra_hooks)

    return builder.build()


# ---------------------------------------------------------------------------
# 3. Run the demo
# ---------------------------------------------------------------------------
def run(task: str, *, hook_skip: dict[str, Callable[[dict[str, Any]], bool]] | None = None) -> None:
    print("\n" + "=" * 72)
    print(f"TASK: {task}")
    print("=" * 72)

    hooks: list[HookProvider] = []
    if hook_skip:
        hooks.append(ConditionalNodeExecution(hook_skip))

    graph = build_graph(extra_hooks=hooks or None)

    try:
        result = graph(task, invocation_state={"demo": True})
    except RuntimeError as exc:
        print(f"Graph execution stopped: {exc}")
        return

    print(f"\nStatus: {result.status.value}")
    print(f"Executed nodes ({len(result.execution_order)}): "
          f"{[n.node_id for n in result.execution_order]}")

    skipped = {n for n in graph.nodes} - {n.node_id for n in result.execution_order}
    if skipped:
        print(f"Skipped nodes: {sorted(skipped)}")

    for node_id, node_result in result.results.items():
        print(f"\n--- {node_id} ({node_result.status.value}) ---")
        print(str(node_result.result).strip())


if __name__ == "__main__":
    # Demo A: math request -> conditional edge routes to math_solver,
    # text_analyzer is cleanly skipped.
    run("What is 17 * 23 + 4?")

    # Demo B: text request -> conditional edge routes to text_analyzer,
    # math_solver is cleanly skipped.
    run("Write me a short poem about the sea.")

    # Demo C: same text request, but the hook unconditionally cancels the
    # text_analyzer node before it runs. Because hook cancellation is a
    # FAILURE rather than a skip, the graph stops with a RuntimeError.
    run(
        "Write me a short poem about the sea.",
        hook_skip={"text_analyzer": lambda inv_state: True},
    )
