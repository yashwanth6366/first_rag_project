# read the data

from langchain_community.document_loaders import TextLoader
from hr_assistant import config

def load_documents(file_path: str = config.DATA_FILE_PATH):
    """
    Load documents from a text file.
    """ 
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()
