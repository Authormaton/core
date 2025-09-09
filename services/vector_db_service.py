"""
Service for storing and retrieving vectors using Pinecone (scaffold).
"""

# To use: pip install pinecone-client
import os
from typing import List

class VectorDBClient:
    def __init__(self, api_key: str, dimension: int = None, cloud: str = "aws", region: str = "us-east-1", environment: str = None):
        """
        Initialize Pinecone client for serverless (cloud/region) or legacy (environment).
        :param api_key: Pinecone API key
        :param dimension: Vector dimension (required for index ops)
        :param cloud: Serverless cloud provider (default 'aws')
        :param region: Serverless region (default 'us-east-1')
        :param environment: Legacy environment (optional, unused for serverless)
        """
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=api_key)  # For serverless, environment is ignored
        self.cloud = cloud
        self.region = region
        self.environment = environment
        self.index = None
        self.dimension = dimension  # Must be set for upsert/query; can be set at index creation

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
            raise TypeError("vectors and ids must be lists.")
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length.")
        if self.dimension is not None and any(len(v) != self.dimension for v in vectors):
            raise ValueError("vector dimensionality mismatch with index.")
        if not self.index:
            raise RuntimeError("Index is not initialized. Call create_index first.")
        # Upsert format may vary by pinecone-client version.
        self.index.upsert(vectors=[(id, vec) for id, vec in zip(ids, vectors)])

    def query(self, vector: List[float], top_k: int = 5):
        if not self.index:
            raise RuntimeError("Index is not initialized. Call create_index first.")
        if self.dimension is not None and len(vector) != self.dimension:
            raise ValueError("query vector dimensionality mismatch with index.")
        return self.index.query(vector=vector, top_k=top_k)
