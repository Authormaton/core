"""
Service for gathering information from the internet and verifying sources.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

TRUSTED_DOMAINS = [".edu", ".gov", "wikipedia.org", "ieee.org", "acm.org", "nature.com", "sciencedirect.com"]

class WebResearchService:
    def __init__(self):
        pass

    def search_and_fetch(self, query: str, num_results: int = 3):
        # Example: Use Bing Web Search API or similar (stubbed here)
        # Replace with real API integration
        # For demo, use DuckDuckGo HTML scraping (not recommended for production)
        search_url = f"https://duckduckgo.com/html/?q={query}"
        resp = requests.get(search_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for a in soup.select("a.result__a")[:num_results]:
            url = a.get("href")
            snippet = self.fetch_and_extract(url)
            verified = self.verify_source(url)
            results.append({"url": url, "snippet": snippet, "verified": verified})
        return results

    def fetch_and_extract(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            # Extract main text (very basic)
            paragraphs = soup.find_all("p")
            text = "\n".join([p.get_text() for p in paragraphs])
            return text[:1000]  # Limit for demo
        except Exception:
            return ""

    def verify_source(self, url: str) -> bool:
        domain = urlparse(url).netloc
        return any(trusted in domain for trusted in TRUSTED_DOMAINS)
