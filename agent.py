import json
import logging
from strands import Agent, tool
from strands.agent import AgentResult
from strands.models import BedrockModel
from strands_tools import calculator, current_time

# Enables Strands debug log level
logging.getLogger("strands").setLevel(logging.DEBUG)

# Set the logging format and streams logs to stderr
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# Define a custom tool as a Python function using the @tool decorator
@tool
def letter_counter(word: str, letter: str) -> int:
    """
    Count occurrences of a specific letter in a word.

    Args:
        word (str): The word to count the letter in.
        letter (str): The specific letter to count.

    Returns:
        int: The number of occurrences of the letter in the word.
    """
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0

    if len(letter) != 1:
        raise ValueError(f"Letter must be a single character. It was {letter}")

    return word.lower().count(letter.lower())

# Create an agent with tools from the community-driven strands-tools package
# as well as our custom letter_counter tool
#
# This Agent uses AWS Amazon Bedrock runtime by default. Also, it selects an
# Anthropic model by default. The documentation says it selects a Claude 4 model.

model_id = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
region_name = "ap-southeast-2"
temperature = 0.5

model = BedrockModel(
    model_id=model_id,
    region_name=region_name,
    temperature=temperature,
)
agent = Agent(
    model=model,
    tools=[calculator, current_time, letter_counter],
    callback_handler=None,
)

# Ask the agent a question that uses the available tools
prompt = """
I have 4 requests:

1. What is the time right now?
2. Calculate 3111696 / 74088
3. Tell me how many letter R's are in the word "strawberry"
"""

result: AgentResult = agent(prompt=prompt)

print(json.dumps(result.metrics.get_summary(), indent=2, default=str))

print(f"Model used by agent: {agent.model.get_config()}")
