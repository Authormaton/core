"""
Service for ranking text passages by relevance to a query.
Uses embedding-based similarity scoring.
"""

from __future__ import annotations

import logging
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional

from services.chunking_service import chunk_text
from services.embedding_service import embed_texts
from services.web_fetch_service import FetchedDoc

logger = logging.getLogger(__name__)

@dataclass
class RankedEvidence:
    """Represents a ranked evidence passage with metadata."""
    id: int  # 1-based ID used for citations
    url: str
    title: Optional[str] = None
    site_name: Optional[str] = None
    passage: str = ""
    score: float = 0.0
    published_at: Optional[str] = None  # ISO format date string

class RankingService:
    """
    Service for ranking passages by relevance to a query.
    Uses embedding-based similarity scoring.
    """
    
    def __init__(self, ideal_passage_length: int = 1000, overlap: int = 200):
        """
        Initialize the ranking service.
        
        Args:
            ideal_passage_length: Target length of passages in characters
            overlap: Overlap between passages in characters
        """
        self.ideal_passage_length = ideal_passage_length
        self.overlap = overlap
    
    def _split_into_passages(self, doc: FetchedDoc) -> List[tuple[str, str, Optional[str], Optional[str], Optional[str]]]:
        """
        Split a document into passages with metadata.
        
        Args:
            doc: The document to split
            
        Returns:
            List of tuples: (passage, url, title, site_name, published_at)
        """
        if not doc.text:
            return []
        
        # Use chunking service to split the text
        passages = chunk_text(
            doc.text, 
            max_length=self.ideal_passage_length, 
            overlap=self.overlap
        )
        
        # Return passages with metadata
        return [(p, doc.url, doc.title, doc.site_name, doc.published_at) for p in passages]
    
    def _compute_similarity(self, query_embedding: List[float], passage_embeddings: List[List[float]]) -> List[float]:
        """
        Compute cosine similarity between query and passages.
        
        Args:
            query_embedding: Query embedding vector
            passage_embeddings: List of passage embedding vectors
            
        Returns:
            List of similarity scores
        """
        # Convert to numpy arrays for efficient computation
        query_vec = np.array(query_embedding)
        passage_vecs = np.array(passage_embeddings)
        
        # Normalize vectors with zero-norm guards
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return [0.0] * len(passage_embeddings)
        query_vec = query_vec / q_norm
        p_norms = np.linalg.norm(passage_vecs, axis=1, keepdims=True)
        # Avoid divide-by-zero for degenerate embeddings
        p_norms[p_norms == 0] = 1.0
        passage_vecs = passage_vecs / p_norms

        # Compute cosine similarity
        similarities = np.dot(passage_vecs, query_vec)

        return similarities.tolist()
    
    async def rank_documents(self, query: str, docs: List[FetchedDoc], 
                           max_context_chars: int = 20000) -> List[RankedEvidence]:
        """
        Split documents into passages, rank them by relevance to query,
        and return top passages within context budget.
        
        Args:
            query: The search query
            docs: List of fetched documents
            max_context_chars: Maximum total character budget for evidence
            
        Returns:
            List of RankedEvidence objects
        """
        if not query or not docs:
            return []
        
        # Split all documents into passages
        all_passages = []
        for doc in docs:
            passages = self._split_into_passages(doc)
            all_passages.extend(passages)
        
        if not all_passages:
            logger.warning("No passages extracted from documents")
            return []
        
        # Unpack passages and metadata
        passages = [p[0] for p in all_passages]
        urls = [p[1] for p in all_passages]
        titles = [p[2] for p in all_passages]
        site_names = [p[3] for p in all_passages]
        published_dates = [p[4] for p in all_passages]
        
        # Get embeddings for query and passages
        start_time = time.time()
        try:
            # Embed query
            query_embedding = embed_texts([query])[0]
            
            # Embed passages
            passage_embeddings = embed_texts(passages)
            
            # Compute similarity scores
            similarity_scores = self._compute_similarity(query_embedding, passage_embeddings)
        except Exception as e:
            logger.error(f"Error during embedding or similarity computation: {str(e)}")
            # Fallback: assign decreasing scores based on original order
            similarity_scores = [1.0 - (i / len(passages)) for i in range(len(passages))]
        
        # Create scored passages
        scored_passages = []
        for i, (passage, url, title, site_name, published_at, score) in enumerate(
            zip(passages, urls, titles, site_names, published_dates, similarity_scores)
        ):
            # Create RankedEvidence with 1-based ID
            evidence = RankedEvidence(
                id=i + 1,  # 1-based ID for citations
                url=url,
                title=title,
                site_name=site_name,
                passage=passage,
                score=score,
                published_at=published_at
            )
            scored_passages.append(evidence)
        
        # Sort by score (highest first)
        scored_passages.sort(key=lambda p: p.score, reverse=True)
        
        # Select top passages within context budget
        selected_passages = []
        current_budget = 0
        
        for passage in scored_passages:
            passage_length = len(passage.passage)
            if current_budget + passage_length <= max_context_chars:
                selected_passages.append(passage)
                current_budget += passage_length
            else:
                # If we can't fit the full passage, check if we can fit a truncated version
                remaining_budget = max_context_chars - current_budget
                if remaining_budget >= 200:  # Only truncate if we can fit a meaningful chunk
                    truncated_passage = passage.passage[:remaining_budget - 3] + "..."
                    passage.passage = truncated_passage
                    selected_passages.append(passage)
                break
        
        # Re-assign IDs to ensure they're sequential
        for i, passage in enumerate(selected_passages):
            passage.id = i + 1
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Ranked {len(all_passages)} passages in {duration_ms}ms, selected {len(selected_passages)} within budget")
        
        return selected_passages