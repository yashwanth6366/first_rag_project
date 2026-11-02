# store the data in faiss vector store.

import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model
from hr_assistant.document_loader import load_documents
from hr_assistant.splitter import split_into_chunks


#building the vector store

def build_vector_store(chunks):
    """
    embed the chunks and build a FAISS vector store.
    """
    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings_model)


# save the vector store

def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH):
    """
    Save the FAISS vector store.
    """
    vector_store.save_local(path)


def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    """
    Load the FAISS vector store.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector store not found at {path}. Please build it first.")
    embeddings_model = get_embeddings_model()
    return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)

# check if the vector store exists, if not build it.

def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:
    """
    Check if the FAISS vector store exists.
    """
    return os.path.exists(os.path.join(path, "index.faiss"))

# retriever

def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """
    Get a retriever from the FAISS vector store.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})