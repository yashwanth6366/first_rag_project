# connects to the llm 

from langchain_groq import ChatGroq
from hr_assistant import config

def get_llm():
    """
    Get the LLM model from Groq.
    """
    return ChatGroq(
        model=config.LLM_MODEL_name,
        temperature=0,
    )