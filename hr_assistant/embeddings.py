

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

def get_embeddings_model():
    """
    Get the Jina embeddings model using the provided API key.
    
    Returns:
        JinaEmbeddings: An instance of the JinaEmbeddings class.
    """
    return JinaEmbeddings(model_name=config.Embedding_MODEL_name) 