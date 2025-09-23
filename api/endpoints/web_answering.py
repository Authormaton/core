"""
Web search and answering endpoint for Perplexity-style answers with citations.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
import os
import secrets
from pydantic import BaseModel

from config.settings import settings
from services.web_search_service import WebSearchService
from services.web_fetch_service import WebFetchService
from services.ranking_service import RankingService
from services.synthesis_service import SynthesisService
from pydantic import field_validator

logger = logging.getLogger(__name__)

# Reuse internal API key authentication
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY")
api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

# Create router
router = APIRouter(tags=["websearch"])

# Request model
class WebSearchAnswerRequest(BaseModel):
    query: str
    top_k_results: int = 8
    max_context_chars: int = 20000
    region: str = "auto"
    language: str = "en"
    style_profile_id: Optional[str] = None
    answer_tokens: int = 800
    include_snippets: bool = True
    timeout_seconds: int = 25

    # Validators
    @field_validator("top_k_results")
    @classmethod
    def validate_top_k_results(cls, v):
        return max(3, min(15, v))  # Clamp between 3 and 15

    @field_validator("max_context_chars")
    @classmethod
    def validate_max_context_chars(cls, v):
        return max(1000, min(50000, v))  # Clamp between 1000 and 50000

    @field_validator("answer_tokens")
    @classmethod
    def validate_answer_tokens(cls, v):
        return max(100, min(2000, v))  # Clamp between 100 and 2000

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, v):
        return max(5, min(60, v))  # Clamp between 5 and 60 seconds

# Citation model
class Citation(BaseModel):
    id: int
    url: str
    title: Optional[str] = None
    site_name: Optional[str] = None
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    score: float

# Timings model
class Timings(BaseModel):
    search: int = 0
    fetch: int = 0
    rank: int = 0
    generate: int = 0
    total: int = 0

# Metadata model
class Meta(BaseModel):
    engine: str
    region: str
    language: str
    style_profile_id: Optional[str] = None

# Response model
class WebSearchAnswerResponse(BaseModel):
    query: str
    answer_markdown: str
    citations: List[Citation]
    used_sources_count: int
    timings_ms: Timings
    meta: Meta

# Dependency for internal authentication
def verify_internal_api_key(api_key: str = Depends(api_key_header)):
    # Do not log or expose the secret
    if not api_key or not INTERNAL_API_KEY or not secrets.compare_digest(api_key, INTERNAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key."
        )

@router.post("/websearch/answer", response_model=WebSearchAnswerResponse, status_code=200)
async def web_search_answer(
    request: WebSearchAnswerRequest,
    req: Request,
    api_key: str = Depends(verify_internal_api_key)
):
    start_time = time.time()
    """
    Perform web search and generate an answer with citations.
    """
    # Generate request ID
    request_id = req.headers.get("X-Request-Id", str(uuid.uuid4()))
    logger_ctx = {"request_id": request_id}
    
    # Log request
    logger.info(
        f"Web search answer request: query='{request.query}', "
        f"top_k={request.top_k_results}, timeout={request.timeout_seconds}s",
        extra=logger_ctx
    )
    
    # Initialize timings
    timings = {
        "search": 0,
        "fetch": 0,
        "rank": 0,
        "generate": 0,
        "total": 0
    }
    # Initialize services
    try:
        search_service = WebSearchService()
        fetch_service = WebFetchService()
        ranking_service = RankingService()
        synthesis_service = SynthesisService()
    except Exception as e:
        logger.error("Failed to initialize services: %s", e, extra=logger_ctx)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service initialization error."
        )

    # 1. Search phase
    search_start = time.time()
    try:
        search_results = await asyncio.wait_for(
            search_service.search(
                query=request.query,
                k=request.top_k_results,
                region=request.region,
                language=request.language,
                timeout_seconds=min(request.timeout_seconds * 0.4, 10)  # Allocate 40% of timeout
            ),
            timeout=request.timeout_seconds * 0.4
        )
    except asyncio.TimeoutError:
        logger.warning(f"Search timed out after {request.timeout_seconds * 0.4}s", extra=logger_ctx)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Search phase timed out."
        )
    timings["search"] = int((time.time() - search_start) * 1000)
    if not search_results:
        logger.warning("No search results found", extra=logger_ctx)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No search results found for the query."
        )
    logger.info(f"Search completed with {len(search_results)} results", extra=logger_ctx)

    # 2. Fetch phase
    fetch_start = time.time()
    try:
        fetched_docs = await asyncio.wait_for(
            fetch_service.fetch_search_results(
                search_results=search_results,
                timeout_seconds=min(request.timeout_seconds * 0.3, 8),  # Allocate 30% of timeout
                preserve_snippets=request.include_snippets
            ),
            timeout=request.timeout_seconds * 0.3
        )
    except asyncio.TimeoutError:
        logger.warning(f"Fetch timed out after {request.timeout_seconds * 0.3}s", extra=logger_ctx)
        # Continue with whatever we got
        fetched_docs = []
    timings["fetch"] = int((time.time() - fetch_start) * 1000)
    if not fetched_docs and not request.include_snippets:
        logger.warning("No documents fetched successfully", extra=logger_ctx)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to fetch content from search results."
        )
    logger.info(f"Fetch completed with {len(fetched_docs)} documents", extra=logger_ctx)

    # 3. Ranking phase
    rank_start = time.time()
    try:
        ranked_evidence = await asyncio.wait_for(
            ranking_service.rank_documents(
                query=request.query,
                docs=fetched_docs,
                max_context_chars=request.max_context_chars
            ),
            timeout=request.timeout_seconds * 0.15  # Allocate 15% of timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Ranking timed out after {request.timeout_seconds * 0.15}s", extra=logger_ctx)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Ranking phase timed out."
        )
    timings["rank"] = int((time.time() - rank_start) * 1000)
    if not ranked_evidence:
        logger.warning("No evidence ranked for the query", extra=logger_ctx)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No relevant evidence found for the query."
        )
    logger.info(f"Ranking completed with {len(ranked_evidence)} evidence passages", extra=logger_ctx)

    # 4. Synthesis phase
    generate_start = time.time()
    try:
        synthesis_result = await asyncio.wait_for(
            synthesis_service.generate_answer(
                query=request.query,
                evidence_list=ranked_evidence,
                answer_tokens=request.answer_tokens,
                style_profile_id=request.style_profile_id
            ),
            timeout=request.timeout_seconds * 0.15  # Allocate remaining time
        )
    except asyncio.TimeoutError:
        logger.warning(f"Synthesis timed out after {request.timeout_seconds * 0.15}s", extra=logger_ctx)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Synthesis phase timed out."
        )
    timings["generate"] = int((time.time() - generate_start) * 1000)

    # Calculate total time
    timings["total"] = int((time.time() - start_time) * 1000)

    # Check if any citations were used
    if not synthesis_result.used_citation_ids:
        logger.warning("No citations used in the answer", extra=logger_ctx)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No evidence-based answer could be produced within constraints."
        )

    # Create citations list, including only those actually used in the answer
    citations = []
    for evidence in ranked_evidence:
        if evidence.id in synthesis_result.used_citation_ids:
            citations.append(Citation(
                id=evidence.id,
                url=evidence.url,
                title=evidence.title,
                site_name=evidence.site_name,
                published_at=evidence.published_at,
                snippet=evidence.passage[:200] + "..." if len(evidence.passage) > 200 else evidence.passage,
                score=evidence.score
            ))

    # Sort citations by ID for consistency
    citations.sort(key=lambda c: c.id)

    # Build response
    response = WebSearchAnswerResponse(
        query=request.query,
        answer_markdown=synthesis_result.answer_markdown,
        citations=citations,
        used_sources_count=len(citations),
        timings_ms=Timings(**timings),
        meta=Meta(
            engine=settings.web_search_engine,
            region=request.region,
            language=request.language,
            style_profile_id=request.style_profile_id
        )
    )

    logger.info(
        f"Answer generated successfully in {timings['total']}ms with {len(citations)} citations",
        extra=logger_ctx
    )

    return response