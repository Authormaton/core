"""
Service for gathering information from the internet and verifying sources.
"""

import requests
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Constants (define if missing)
DEFAULT_HTTP_TIMEOUT = 10
DEFAULT_USER_AGENT = "CoreWebResearch/1.0"

class WebResearchService:
    def __init__(self, timeout: int = DEFAULT_HTTP_TIMEOUT, user_agent: str = DEFAULT_USER_AGENT):
        self.timeout = timeout
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def search_and_fetch(self, query: str, num_results: int = 3):
        # Example: Use Bing Web Search API or similar (stubbed here)
        # Replace with real API integration
        # For demo, use DuckDuckGo HTML scraping (not recommended for production)
        import logging
        from urllib.parse import urljoin, urlparse
        results = []
        try:
            resp = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=DEFAULT_HTTP_TIMEOUT,
                headers={"User-Agent": DEFAULT_USER_AGENT}
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logging.warning(f"Search request failed: {e}")
            return []

        for a in soup.select("a.result__a")[:num_results]:
            href = a.get("href")
            url = urljoin("https://duckduckgo.com/", href) if href else ""
            parsed = urlparse(url)
            if not url or parsed.scheme not in ("http", "https"):
                continue
            snippet = self.fetch_and_extract(url)
            verified = self.verify_source(url)
            results.append({"url": url, "snippet": snippet, "verified": verified})
        return results

    def fetch_and_extract(self, url: str) -> str:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                logging.warning(f"Fetch failed for {url}: status {resp.status_code}")
                return ""
            ctype = resp.headers.get("Content-Type", "").lower()
            if "html" not in ctype and "text" not in ctype:
                logging.warning(f"Non-HTML/text content for {url}: {ctype}")
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join([p.get_text(" ", strip=True) for p in paragraphs])
            return text[:1000]  # Limit for demo
        except requests.RequestException as e:
            logging.warning(f"Fetch failed for {url}: {e}", exc_info=True)
            return ""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                logging.warning(f"Fetch failed for {url}: status {resp.status_code}")
                return ""
            ctype = resp.headers.get("Content-Type", "").lower()
            if "html" not in ctype and "text" not in ctype:
                logging.warning(f"Non-HTML/text content for {url}: {ctype}")
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join([p.get_text(" ", strip=True) for p in paragraphs])
            return text[:1000]  # Limit for demo
        except requests.RequestException as e:
            logging.warning(f"Fetch failed for {url}: {e}", exc_info=True)
            return ""

    def verify_source(self, url: str) -> bool:
        netloc = urlparse(url).netloc.lower()
        # Remove port if present
        if ':' in netloc:
            netloc = netloc.split(':')[0]
        if not netloc:
            return False
        for trusted in TRUSTED_BASE_DOMAINS:
            trusted = trusted.lower()
            if netloc == trusted or netloc.endswith('.' + trusted):
                return True
        for suffix in TRUSTED_DOMAIN_SUFFIXES:
            if netloc.endswith(suffix):
                return True
        return False
