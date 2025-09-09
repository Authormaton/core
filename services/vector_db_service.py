"""
Service for storing and retrieving vectors using Pinecone (scaffold).
"""

# To use: pip install pinecone-client
import os
from typing import List

class VectorDBClient:
    def __init__(self, api_key: str, environment: str = "us-east-1", cloud: str = "aws", region: str = "us-east-1"):
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=api_key)  # environment unused for serverless
        self.cloud = cloud
        self.region = region
        self.index = None

    def create_index(self, index_name: str, dimension: int):
        from pinecone import ServerlessSpec
        existing = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self.index = self.pc.Index(index_name)

    def upsert_vectors(self, vectors: List[List[float]], ids: List[str]):
        # Input validation
        if vectors is None or ids is None:
            raise ValueError("vectors and ids must not be None.")
        if not isinstance(vectors, list) or not isinstance(ids, list):
            raise ValueError("vectors and ids must be lists.")
        if len(vectors) != len(ids):
            raise ValueError(f"vectors and ids must be lists of the same length. Got {len(vectors)} vectors and {len(ids)} ids.")
        if self.index:
            self.index.upsert(vectors=[(id, vec) for id, vec in zip(ids, vectors)])

    def query(self, vector: List[float], top_k: int = 5):
        if self.index:
            return self.index.query(vector=vector, top_k=top_k)
        return None
