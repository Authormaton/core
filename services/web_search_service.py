"""
Service for performing web searches via different providers.
Currently supports Tavily, with provider-agnostic wrapper.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Type, ClassVar, Mapping

import httpx
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result from any provider."""
    url: str
    title: Optional[str] = None
    site_name: Optional[str] = None
    snippet: Optional[str] = None
    published_at: Optional[str] = None  # ISO format date string
    score: Optional[float] = None
    provider_meta: Dict = field(default_factory=dict)

class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    async def search(self, query: str, k: int, region: str, language: str, timeout_seconds: int) -> List[SearchResult]:
        """
        Perform a search with the given parameters.
        
        Args:
            query: The search query
            k: Number of results to return
            region: Region code (e.g., "us", "eu", "auto")
            language: Language code (e.g., "en", "fr", "de")
            timeout_seconds: Timeout in seconds
            
        Returns:
            List of SearchResult objects
        """
        pass

class DummySearchProvider(SearchProvider):
    """Dummy search provider for testing or when no API keys are available."""
    
    async def search(self, query: str, k: int, region: str, language: str, timeout_seconds: int) -> List[SearchResult]:
        """
        Return dummy search results for testing.
        
        Args:
            query: The search query
            k: Number of results to return
            region: Region code (ignored)
            language: Language code (ignored)
            timeout_seconds: Timeout in seconds (ignored)
            
        Returns:
            List of dummy SearchResult objects
        """
        # Create k dummy results
        results = []
        
        for i in range(min(k, 5)):  # Cap at 5 results
            result = SearchResult(
                url=f"https://example.com/result-{i+1}",
                title=f"Dummy Result {i+1} for '{query}'",
                site_name="Example.com",
                snippet=f"This is a dummy search result #{i+1} for the query: '{query}'. "
                       f"This is placeholder text and does not contain real information.",
                published_at="2025-09-01T12:00:00Z",
                score=1.0 - (i * 0.1),
                provider_meta={"provider": "dummy", "result_id": i+1}
            )
            results.append(result)
        
        # Simulate network delay
        await asyncio.sleep(0.5)
        
        logger.warning(f"Using DummySearchProvider for query: {query}. Configure a real search provider for production.")
        return results

class TavilySearchProvider(SearchProvider):
    """Tavily search provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with Tavily API key.
        
        Args:
            api_key: Tavily API key (defaults to settings.tavily_api_key)
        """
        self.api_key = api_key or settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
        if not self.api_key:
            raise ValueError("Tavily API key is required but not provided in settings or constructor")
        
        # Tavily API configuration
        self.api_url = "https://api.tavily.com/search"
    
    async def search(self, query: str, k: int, region: str, language: str, timeout_seconds: int) -> List[SearchResult]:
        """
        Perform a search using Tavily API.
        
        Args:
            query: The search query
            k: Number of results to return (max 15)
            region: Region hint (Tavily handles this internally)
            language: Language code
            timeout_seconds: Timeout in seconds
            
        Returns:
            List of SearchResult objects
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Clamp k between 3 and 15
        k = max(3, min(15, k))
        
        params = {
            "query": query,
            "search_depth": "advanced",
            "include_domains": [],
            "exclude_domains": [],
            "max_results": k,
            "include_answer": False,
            "include_raw_content": False,  # We'll fetch the content separately
        }
        
        # Add language if not auto
        if language and language.lower() != "auto":
            params["language"] = language
            
        # Create HTTP client with timeout and retry logic
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            max_retries = 3
            
            # Set headers with API key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            for attempt in range(max_retries):
                try:
                    response = await client.post(self.api_url, json=params, headers=headers)
                    response.raise_for_status()
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) + random.random()
                            logger.warning(f"Rate limited by Tavily. Retrying in {wait_time:.2f}s")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Tavily search failed after {max_retries} attempts: rate limited")
                            raise
                    else:
                        logger.error(f"Tavily search failed with status {e.response.status_code}: {e.response.text}")
                        raise
                except httpx.RequestError as e:
                    logger.error(f"Tavily request failed: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.random()
                        logger.warning(f"Request error. Retrying in {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            data = response.json()
            results = data.get("results", [])
            
            search_results = []
            for idx, result in enumerate(results):
                # Extract site_name from URL when domain is missing
                site = result.get("domain")
                if not site:
                    try:
                        from urllib.parse import urlparse  # Local import to avoid module load failures
                        site = urlparse(result.get("url", "")).netloc or None
                    except Exception:
                        site = None
                
                search_result = SearchResult(
                    url=result.get("url", ""),
                    title=result.get("title"),
                    site_name=site,
                    snippet=result.get("description") or result.get("content"),
                    published_at=result.get("published_date"),
                    # Prefer Tavily's score; fallback to position-based score
                    score=(result.get("score") if isinstance(result.get("score"), (int, float)) else None) 
                         or (1.0 - (idx / (len(results) or 1))),
                    provider_meta={"tavily_id": result.get("id")} if "id" in result else {}
                )
                search_results.append(search_result)
            
            return search_results

class WebSearchService:
    """
    Service for performing web searches across different providers.
    Uses the provider specified in settings.web_search_engine.
    """
    
    # Registry of available providers
    _providers: ClassVar[Dict[str, Type[SearchProvider]]] = {
        "tavily": TavilySearchProvider,
        "dummy": DummySearchProvider,
    }
    
    # LRU cache for search results (module-level, simple implementation)
    _cache: ClassVar[Dict[str, tuple[float, List[SearchResult]]]] = {}
    _cache_ttl: ClassVar[int] = 300  # 5 minutes in seconds
    _cache_max_size: ClassVar[int] = 100
    
    def __init__(self, provider_name: Optional[str] = None):
        """
        Initialize the web search service.
        
        Args:
            provider_name: Name of the provider to use (defaults to settings.web_search_engine)
        """
        self.provider_name = provider_name or settings.web_search_engine
        
        # If no provider name is set, use the dummy provider
        if not self.provider_name or self.provider_name == "none":
            logger.warning("No search provider specified. Using DummySearchProvider for development.")
            self.provider_name = "dummy"
            self.provider = DummySearchProvider()
            return
        
        if self.provider_name not in self._providers:
            raise ValueError(f"Unsupported search provider: {self.provider_name}")
        
        try:
            self.provider = self._providers[self.provider_name]()
            logger.info(f"Initialized WebSearchService with provider: {self.provider_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider_name} search provider: {str(e)}")
            logger.warning("Falling back to DummySearchProvider due to initialization failure.")
            self.provider_name = "dummy"
            self.provider = DummySearchProvider()
    
    async def search(self, query: str, k: int = None, region: str = "auto", 
                    language: str = "en", timeout_seconds: int = 15,
                    use_cache: bool = True) -> List[SearchResult]:
        """
        Perform a web search using the configured provider.
        
        Args:
            query: The search query
            k: Number of results to return (defaults to settings.default_top_k_results)
            region: Region code or "auto"
            language: Language code (default "en")
            timeout_seconds: Timeout in seconds
            use_cache: Whether to use the cache
            
        Returns:
            List of SearchResult objects
        """
        if not query:
            return []
        
        # Use default from settings if not specified
        k = k or settings.default_top_k_results
        
        # Check cache first if enabled
        cache_key = f"{self.provider_name}:{query}:{k}:{region}:{language}"
        if use_cache and cache_key in self._cache:
            timestamp, results = self._cache[cache_key]
            if time.time() - timestamp <= self._cache_ttl:
                logger.debug(f"Cache hit for query: {query}")
                return results
        
        # Cache miss or disabled, perform actual search
        try:
            start_time = time.time()
            results = await self.provider.search(
                query=query, 
                k=k, 
                region=region, 
                language=language, 
                timeout_seconds=timeout_seconds
            )
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Search completed in {duration_ms:.0f}ms. Query: {query}, Provider: {self.provider_name}, Results: {len(results)}")
            
            # Cache results if caching is enabled
            if use_cache:
                # If cache is full, remove the oldest entry
                if len(self._cache) >= self._cache_max_size:
                    oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                    self._cache.pop(oldest_key)
                
                self._cache[cache_key] = (time.time(), results)
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise