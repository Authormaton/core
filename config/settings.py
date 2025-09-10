
"""
Settings for Authormaton core service (Pinecone, embedding, upload limits).
Automatically loads environment variables from .env for local development.
"""
import os
from pydantic_settings import BaseSettings
from pydantic import SecretStr, ValidationError
import sys
try:
    from dotenv import load_dotenv
    # Only load .env if running interactively or as __main__ (not in production)
    if os.environ.get("ENV", "dev") != "prod":
        load_dotenv()
except ImportError:
    pass

class Settings(BaseSettings):
    pinecone_api_key: SecretStr
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_index_name: str = "authormaton-core"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    embed_batch_size: int = 128
    max_upload_mb: int = 25

try:
    settings = Settings()
except ValidationError as e:
    print(f"[FATAL] Invalid or missing environment variables: {e}", file=sys.stderr)
    sys.exit(1)
