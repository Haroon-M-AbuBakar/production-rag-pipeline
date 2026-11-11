import os
from dotenv import load_dotenv

# Load environment variables from a local .env file if present
load_dotenv()

class Settings:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-dummy-key-not-used")
    
    # Qdrant Vector Store Config
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = "documents"
    
    # Model Configurations
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    GENERATION_MODEL: str = "gpt-4o-mini"
    
    # RAG Hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

settings = Settings()

if not settings.OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is missing from environment variables.")