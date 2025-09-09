"""
Service for storing and retrieving vectors using Pinecone (scaffold).
"""

# To use: pip install pinecone-client
import os
from typing import List

class VectorDBClient:
    def __init__(self, api_key: str, environment: str):
        import pinecone
        pinecone.init(api_key=api_key, environment=environment)
        self.index = None

    def create_index(self, index_name: str, dimension: int):
        import pinecone
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dimension)
        self.index = pinecone.Index(index_name)

    def upsert_vectors(self, vectors: List[List[float]], ids: List[str]):
        if self.index:
            self.index.upsert(vectors=[(id, vec) for id, vec in zip(ids, vectors)])

    def query(self, vector: List[float], top_k: int = 5):
        if self.index:
            return self.index.query(vector=vector, top_k=top_k)
        return None
