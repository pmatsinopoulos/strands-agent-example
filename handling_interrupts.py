import json
import logging
from dataclasses import asdict
from strands import Agent, tool
from strands.hooks import AfterInvocationEvent, AfterToolCallEvent, BeforeToolCallEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@tool
def send_email(to: str, body: str) -> str:
    """Send an email.

    Args:
        to (str): The email address to send the email to.
        body (str): The body of the email.
    """
    return f"Email sent to {to}"

async def require_approval(event: BeforeToolCallEvent):
    """Interrupt before sending emails to require approval."""
    logger.info(f"[require_approval] *****************************")
    logger.info(f"[require_approval] Tool use: {event.tool_use}")
    logger.info(f"[require_approval] *****************************")
    if event.tool_use["name"] == "send_email":
        approval = event.interrupt("email_approval", reason="Approve this email?")
        logger.info(f"[require_approval] Approval: {approval}")
        logger.info(f"[require_approval] *****************************")
        if approval != "approved":
            event.cancel_tool = "Email not approved"

async def after_tool_call(event: AfterToolCallEvent):
    """After tool call call."""
    logger.info(f"[after tool call] *****************************")
    logger.info(f"[after tool call] Tool use: {event.tool_use}")
    logger.info(f"[after tool call] *****************************")
    if event.tool_use["name"] == "send_email":
        logger.info(f"[after tool call] Tool call completed: {event.tool_use['name']}")

async def auto_approve(event: AfterInvocationEvent):
    """Automatically approve all interrupted tool calls."""
    logger.info(f"[auto_approve] *****************************")
    logger.info(f"[auto_approve] Result: {event.result}, Stop reason: {event.result.stop_reason}")
    logger.info(f"[auto_approve] *****************************")
    if event.result and event.result.stop_reason == "interrupt":
        logger.info(f"[auto_approve] *****************************")
        logger.info(f"[auto_approve] Interrupts: {event.result.interrupts}")
        logger.info(f"[auto_approve] *****************************")

        # invocation ends because of an interrupt
        responses = [
            {
                "interruptResponse": {
                    "interruptId": interrupt.id,
                    "response": "approved"
                }
            }
            for interrupt in event.result.interrupts
        ]
        # resume the invocation by automatically handling the interrupt
        # setting to +resume+ the list of interrupt responses
        event.resume = responses

agent = Agent(
    callback_handler=None,
    tools=[send_email],
    system_prompt="You are a helpful assistant that can send emails."
)
agent.add_hook(event_type=BeforeToolCallEvent, callback=require_approval)
agent.add_hook(event_type=AfterInvocationEvent, callback=auto_approve)
agent.add_hook(event_type=AfterToolCallEvent, callback=after_tool_call)

# The interrupt is automatically handled by the hook
# the caller receives the final result directly
result = agent(prompt="Send an email to alice@example.com with the body 'hello Alice'.")

print("\n--------------------------------\n")
print(json.dumps(asdict(result), indent=2, default=str))
print("\n--------------------------------\n")

print("\n")

print(json.dumps(agent.messages, indent=2, default=str))
