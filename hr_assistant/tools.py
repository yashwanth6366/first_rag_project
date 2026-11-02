# wrap the retriever as a tool the agent can call

from langchain.tools import tool

def create_search_tool(retriever):
    """
    return a tool function that can be used by the agent to search the HR policy documents.
    """
    @tool
    def search_hr_policy(questions:str)-> str:
        """
        Search the HR policy documents for relevant information based on the provided questions.
        """
        matching_chunks = retriever.invoke(questions)
        return "\n\n ".join([chunk.page_content for chunk in matching_chunks])
    return search_hr_policy
    
