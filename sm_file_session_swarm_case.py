from strands.session.file_session_manager import FileSessionManager
from strands import Agent
from strands.multiagent import Swarm
from strands_tools import calculator, shell, file_write

session_manager = FileSessionManager(
    session_id="user-123",
    storage_dir="./.sessions",
)

agent = Agent(session_manager=session_manager)

agent("Hello. My name is Panos. What is yours?")

multi_session_manager = FileSessionManager(
    session_id="orchestrator-456",
    storage_dir="./.sessions",
)


agent1 = Agent(name="researcher", system_prompt="You are a helpful assistant that can search the internal knowledge base")
agent2 = Agent(name="markdown_writer", system_prompt="You are a helpful assistant that can create markdown content in it memory.")
agent3 = Agent(name="calculator", tools=[calculator])
agent4 = Agent(name="shell_executor", tools=[shell, file_write], system_prompt="You can execute commands in the shell. For example, you can dump a text from memory into a file.")

swarm = Swarm(
    nodes=[agent1, agent2, agent3, agent4],
    session_manager=multi_session_manager
)

swarm("Find the best 100m runner in the world. Generate a short CV of them in an md file in current directory. Calculate their age and print it.")
