import os
from dotenv import load_dotenv
load_dotenv()

## environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

## DATA PATH
DATA_FILE_PATH = os.path.join("DATA", "hr_policy.txt")

##VECTOR STORE
VECTOR_STORE_PATH = os.path.join("data", "faiss_index")

## models

LLM_MODEL_name = "openai/gpt-oss-120b"

Embedding_MODEL_name = "jina-embeddings-v4" 

## chunk size/ text splitting config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


## Retrival reults

TOP_K_RESULTS = 3

 ## SYSTEM INSTRUCTIONS
SYSTEM_PROMPT =  """
    You are a friendly hr assiatent
    Always use the search_hr_policy_tool to look up information.
    facts before answering any questions. 
    If it isn't in the search results, say "I don't know" instead of making up an answer instead of making up an answer. 
"""


## CHECK API KEYS

def check_api_keys()-> None:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please set it in your environment variables.")
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set. Please set it in your environment variables.")