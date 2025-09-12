
"""
Service for storing and retrieving vectors using Pinecone (scaffold).
"""

# To use: pip install pinecone-client
from typing import List
from config.settings import settings

class VectorDBClient:
    def __init__(self, dimension: int = None, index_name: str = None):
        """
        Initialize Pinecone client for serverless (cloud/region).
        Loads API key and config from settings.
        """
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        self.cloud = settings.pinecone_cloud
        self.region = settings.pinecone_region
        self.index_name = index_name or settings.pinecone_index_name
        self.index = None
        self.dimension = dimension or settings.embedding_dimension

    def create_index(self, index_name: str = None, dimension: int = None):
        from pinecone import ServerlessSpec
        idx_name = index_name or self.index_name
        dim = dimension or self.dimension
        existing = [idx.name for idx in self.pc.list_indexes()]
        if idx_name not in existing:
            self.pc.create_index(
                name=idx_name,
                dimension=dim,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self.index = self.pc.Index(idx_name)

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
        self.index.upsert(vectors=[(id, vec) for id, vec in zip(ids, vectors)])

    def query(self, vector: List[float], top_k: int = 5):
        if not self.index:
            raise RuntimeError("Index is not initialized. Call create_index first.")
        if self.dimension is not None and len(vector) != self.dimension:
            raise ValueError("query vector dimensionality mismatch with index.")
        return self.index.query(vector=vector, top_k=top_k)

# Interactive test block
if __name__ == "__main__":
    vdb = VectorDBClient()
    vdb.create_index()
    print("Index created successfully!")
    # Example upsert and query (uncomment to use):
    # ids = ["id1", "id2"]
    # vectors = [[0.0]*vdb.dimension, [1.0]*vdb.dimension]
    # vdb.upsert_vectors(vectors, ids)
    # print("Upserted vectors.")
    # result = vdb.query([0.0]*vdb.dimension)
    # print("Query result:", result)
