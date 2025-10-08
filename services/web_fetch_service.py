"""
Service for fetching and extracting content from web pages.
Concurrently fetches pages with semaphore-based rate limiting.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin

import httpx
from config.settings import settings

logger = logging.getLogger(__name__)

# Import trafilatura if available, otherwise provide a fallback
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logger.warning("trafilatura package not found, using simple HTML extraction fallback")

@dataclass
class FetchedDoc:
    """Represents a fetched web document."""
    url: str
    title: Optional[str] = None
    site_name: Optional[str] = None
    text: str = ""
    published_at: Optional[str] = None  # ISO format date string
    fetch_ms: int = 0  # Time taken to fetch in milliseconds

class WebFetchService:
    """
    Service for fetching and extracting content from web pages.
    Uses asyncio for concurrent fetching with rate limiting via semaphore.
    """
    MAX_REDIRECTS = 5  # Maximum number of redirects to follow
        """
        Initialize the web fetch service.
        
        Args:
            max_concurrency: Maximum number of concurrent requests
                (defaults to settings.max_fetch_concurrency)
        """
        self.max_concurrency = max_concurrency or settings.max_fetch_concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None
        
        # Common HTML headers for the fetch requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
            # Let httpx handle Accept-Encoding and compression automatically
        }
        
        # Initialize httpx.AsyncClient with security and performance settings
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=self.max_concurrency, max_keepalive_connections=20),
            trust_env=False,  # Do not inherit proxy environment variables
            follow_redirects=False,  # Disable automatic redirects
            headers=self.headers # Set default headers for the client
        )

    async def close(self):
        """Close the httpx client session."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _extract_site_name(self, url: str) -> str:
        """Extract site name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix if present
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""
    
    def _is_url_allowed(self, url: str) -> bool:
        """
        Check if a URL is safe to fetch (SSRF protection).
        
        Args:
            url: The URL to validate
            
        Returns:
            True if URL is safe to fetch, False otherwise
        """
        try:
            p = urlparse(url)
            if p.scheme not in ("http", "https"):
                return False
            if not p.hostname or p.username or p.password:
                return False
            try:
                ip = ipaddress.ip_address(p.hostname)
                if not ip.is_global:
                    return False
            except ValueError:
                # Hostname; DNS resolution checks can be added later if needed.
                pass
            return True
        except Exception:
            return False
    
    def _sanitize_html_fallback(self, html: str) -> str:
        """
        Simple HTML sanitizer fallback when trafilatura is not available.
        Removes HTML tags, scripts, styles, and excessive whitespace.
        """
        # Remove scripts and styles
        html = re.sub(r'<script.*?</script>', ' ', html, flags=re.DOTALL)
        html = re.sub(r'<style.*?</style>', ' ', html, flags=re.DOTALL)
        
        # Remove all tags
        html = re.sub(r'<[^>]+>', ' ', html)
        
        # Replace entities
        html = re.sub(r'&nbsp;', ' ', html)
        html = re.sub(r'&amp;', '&', html)
        html = re.sub(r'&lt;', '<', html)
        html = re.sub(r'&gt;', '>', html)
        html = re.sub(r'&quot;', '"', html)
        html = re.sub(r'&#\d+;', ' ', html)
        
        # Normalize whitespace
        html = re.sub(r'\s+', ' ', html)
        
        return html.strip()
    
    def _extract_title(self, html: str) -> Optional[str]:
        """Extract title from HTML."""
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up title
            title = re.sub(r'\s+', ' ', title)
            return title
        return None
    
    def _extract_text_from_html(self, html: str) -> str:
        """
        Extract readable text from HTML using trafilatura if available,
        otherwise fallback to simple regex-based extraction.
        """
        if not html:
            return ""
            
        if TRAFILATURA_AVAILABLE:
            try:
                text = trafilatura.extract(html, include_comments=False, include_tables=False, 
                                          favor_precision=True, include_formatting=False)
                if text:
                    return text
                # If trafilatura returns None, fall back to simple extraction
                logger.warning("trafilatura extraction failed, falling back to simple extraction")
            except Exception as e:
                logger.warning(f"trafilatura extraction error: {e}, falling back to simple extraction")
                
        # Fallback: simple HTML tag removal
        return self._sanitize_html_fallback(html)
    
    async def _fetch_url(self, url: str, timeout_seconds: int = 10, redirect_count: int = 0) -> FetchedDoc:
        """
        Fetch a single URL and extract its content, handling redirects manually.
        
        Args:
            url: The URL to fetch
            timeout_seconds: Timeout in seconds
            redirect_count: Current number of redirects followed
            
        Returns:
            FetchedDoc object with extracted content
        """
        async with self.semaphore:
            start_time = time.time()
            site_name = self._extract_site_name(url)
            
            # Initialize with empty/default values
            fetched_doc = FetchedDoc(
                url=url,
                site_name=site_name,
                fetch_ms=0
            )
            
            # SSRF guard for initial URL and redirects
            if not self._is_url_allowed(url):
                fetched_doc.fetch_ms = int((time.time() - start_time) * 1000)
                logger.warning("Blocked potentially unsafe URL: %s", url)
                return fetched_doc
            
            if redirect_count > self.MAX_REDIRECTS:
                fetched_doc.fetch_ms = int((time.time() - start_time) * 1000)
                logger.warning("Exceeded maximum redirect limit for URL: %s", url)
                return fetched_doc
            
            try:
                response = await self._client.get(url, timeout=timeout_seconds)
                response.raise_for_status()
                
                # Manual redirect handling
                if 300 <= response.status_code < 400:
                    location = response.headers.get("Location")
                    if location:
                        new_url = urljoin(url, location)
                        logger.info(f"Following redirect from {url} to {new_url}")
                        return await self._fetch_url(new_url, timeout_seconds, redirect_count + 1)
                    else:
                        logger.warning(f"Redirect status {response.status_code} but no Location header for {url}")
                        
                html = response.text
                
                # Extract title if not already provided
                title = self._extract_title(html)
                
                # Extract text content
                text = self._extract_text_from_html(html)
                
                # Update the fetched document
                fetched_doc.title = title
                fetched_doc.text = text
                
                # Calculate fetch time
                fetch_ms = int((time.time() - start_time) * 1000)
                fetched_doc.fetch_ms = fetch_ms
                
                logger.info(f"Fetched {url} in {fetch_ms}ms, extracted {len(text)} chars")
                return fetched_doc
                    
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                logger.warning(f"HTTP error {status} fetching {url}: {str(e)}")
            except httpx.RequestError as e:
                logger.warning(f"Request error fetching {url}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error fetching {url}: {str(e)}")
                
            # If we got here, there was an error
            fetch_ms = int((time.time() - start_time) * 1000)
            fetched_doc.fetch_ms = fetch_ms
            logger.warning(f"Failed to fetch {url} after {fetch_ms}ms")
            return fetched_doc
    
    async def fetch_urls(self, urls: List[str], timeout_seconds: int = 10) -> List[FetchedDoc]:
        """
        Fetch multiple URLs concurrently.
        
        Args:
            urls: List of URLs to fetch
            timeout_seconds: Timeout in seconds per request
            
        Returns:
            List of FetchedDoc objects
        """
        if not urls:
            return []
        
        # Create fetch tasks for all URLs
        tasks = [self._fetch_url(url, timeout_seconds) for url in urls]
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in fetch_urls: {str(e)}")
            return []
        
        # Filter out exceptions and empty results
        fetched_docs = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Exception during fetch: {str(result)}")
            elif isinstance(result, FetchedDoc) and result.text:
                fetched_docs.append(result)
        
        logger.info(f"Fetched {len(fetched_docs)}/{len(urls)} URLs successfully")
        return fetched_docs
    
    async def fetch_search_results(self, search_results: List, timeout_seconds: int = 10,
                                 preserve_snippets: bool = True) -> List[FetchedDoc]:
        """
        Fetch content for search results.
        
        Args:
            search_results: List of SearchResult objects
            timeout_seconds: Timeout in seconds per request
            preserve_snippets: Whether to use snippets as fallback when fetch fails
            
        Returns:
            List of FetchedDoc objects
        """
        # Extract URLs from search results
        urls = [result.url for result in search_results if result.url]
        
        # Create a mapping of URL to search result for later use
        url_to_result = {result.url: result for result in search_results if result.url}
        
        # Fetch all URLs
        fetched_docs = await self.fetch_urls(urls, timeout_seconds)
        
        # If preserve_snippets is True, create FetchedDocs for failed fetches using snippets
        if preserve_snippets:
            fetched_urls = {doc.url for doc in fetched_docs}
            for url in urls:
                if url not in fetched_urls and url in url_to_result:
                    result = url_to_result[url]
                    if result.snippet:
                        logger.info(f"Using snippet as fallback for {url}")
                        fetched_docs.append(FetchedDoc(
                            url=url,
                            title=result.title,
                            site_name=result.site_name or self._extract_site_name(url),
                            text=result.snippet,
                            published_at=result.published_at,
                            fetch_ms=0  # Indicate that this wasn't actually fetched
                        ))
        
        return fetched_docs