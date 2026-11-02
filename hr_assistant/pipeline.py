# connect all the componets together into one ready to use agent

from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.document_loader import load_documents
from hr_assistant.llm import get_llm
from hr_assistant.splitter import split_into_chunks
from hr_assistant.tools import create_search_tool
from hr_assistant.vectore_store import (
    build_vector_store,
    get_retriever,
    load_vector_store,
    save_vector_store,
    vector_store_exists,
)


def build_vector_store_from_documents(file_path: str = config.DATA_FILE_PATH):
    """
    load+split+embed the document.
    """
    if vector_store_exists():
        print("Vector store already exists. Loading from disk...")
        return load_vector_store()

    print("Vector store does not exist. Building from documents...")
    documents = load_documents(file_path)
    chunks = split_into_chunks(documents)
    print(f"loaded '{file_path}' and split into {len(chunks)} chunks.")


    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    print("Vector store built and saved to disk.")
    return vector_store


def build_hr_assistant(file_path: str = config.DATA_FILE_PATH):
    """
    Build the full rag agent.
    """
    config.check_api_keys()

    vector_store = build_vector_store_from_documents(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)

    llm = get_llm()
    agent = create_hr_agent(llm,[search_tool])
    return agent

def ask(agent, question: str) -> str:
    """
    Ask agent a question and return the response as a plain text.
    """
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content
