# build ai agent that ties the llm and the search tool together

from langchain.agents import create_agent
from hr_assistant import config   


def create_hr_agent(llm, tools):
    """
    Create an AI agent that can answer questions about HR policies using the provided LLM and tools.
    """
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt = config.SYSTEM_PROMPT
    )