"""
WebResearchService: Gather information from the internet and verify sources.
"""

from services.logging_config import get_logger

logger = get_logger(__name__)
from typing import Iterable
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Default config
DEFAULT_HTTP_TIMEOUT = 10
DEFAULT_USER_AGENT = "CoreWebResearch/1.0"

# Trust domain constants (define here or import from config)
TRUSTED_BASE_DOMAINS = {
    "wikipedia.org",
    "bbc.com",
    "nytimes.com",
    "nature.com",
    "sciencedirect.com",
    "reuters.com",
    "theguardian.com",
    # Add more as needed
}
TRUSTED_DOMAIN_SUFFIXES = {
    ".gov",
    ".edu",
    ".ac.uk",
    # Add more as needed
}

class WebResearchService:
    def __init__(
        self,
        timeout: int = DEFAULT_HTTP_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        trusted_base_domains: Iterable[str] | None = None,
        trusted_domain_suffixes: Iterable[str] | None = None,
        ca_bundle_path: str | None = None,
        trust_roots: Iterable[str] | None = None,
    ):
        """Initialize WebResearchService with session, trust lists, and SSL config."""
        self.timeout = timeout
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        # Trust lists (normalized)
        self.trusted_base_domains = {d.lower().lstrip(".") for d in (trusted_base_domains or TRUSTED_BASE_DOMAINS)}
        self.trusted_domain_suffixes = {"." + s.lower().lstrip(".") for s in (trusted_domain_suffixes or TRUSTED_DOMAIN_SUFFIXES)}
        self.ca_bundle_path = ca_bundle_path
        self.trust_roots = set(trust_roots) if trust_roots else None
        # Retries/backoff
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods={"GET", "HEAD"},
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        # SSL verification (trust anchors)
        if ca_bundle_path:
            self.session.verify = ca_bundle_path
        elif self.trust_roots:
            import ssl
            context = ssl.create_default_context()
            for anchor in self.trust_roots:
                context.load_verify_locations(anchor)
            self.session.verify = context

    def search_and_fetch(self, query: str, num_results: int = 3):
        """Search DuckDuckGo and fetch snippets for top results. Returns title, snippet, verified, and error if any."""
        results = []
        try:
            resp = self.session.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.warning(f"Search request failed: {e}")
            return [{"url": None, "title": None, "snippet": None, "verified": False, "error": str(e)}]

        for a in soup.select("a.result__a")[:num_results]:
            href = a.get("href", "")
            url = self._extract_target_from_ddg(href)
            if not url:
                continue
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                continue
            verified = self.verify_source(url)
            snippet = self.fetch_and_extract(url) if verified else ""
            title = None
            error = None
            if verified:
                try:
                    resp = self.session.get(url, timeout=self.timeout)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")
                    title_tag = soup.find("title")
                    title = title_tag.get_text(strip=True) if title_tag else None
                except Exception as e:
                    error = str(e)
            results.append({"url": url, "title": title, "snippet": snippet, "verified": verified, "error": error})
        return results

    def _extract_target_from_ddg(self, href: str) -> str:
        """Extract real target URL from DuckDuckGo result link."""
        from urllib.parse import parse_qs, unquote, urljoin
        if not href:
            return ""
        p = urlparse(href)
        if p.scheme in ("http", "https"):
            return href
        if href.startswith("/"):
            qs = parse_qs(p.query)
            uddg = qs.get("uddg", [""])[0]
            return unquote(uddg) if uddg else urljoin("https://duckduckgo.com/", href)
        return ""

    def fetch_and_extract(self, url: str) -> str:
        """Fetch a URL and extract up to N paragraphs of text, fallback to meta/og:description, then visible text if no <p> found."""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()
            if "html" not in ctype and "text" not in ctype:
                logging.warning(f"Non-HTML/text content for {url}: {ctype}")
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = soup.find_all("p")
            N = 20  # Cap paragraphs scanned
            if paragraphs:
                text = "\n".join([p.get_text(" ", strip=True) for p in paragraphs[:N]])
            else:
                # Fallback: try meta description or og:description
                meta = soup.find("meta", attrs={"name": "description"})
                og = soup.find("meta", attrs={"property": "og:description"})
                text = ""
                if meta and meta.get("content"):
                    text = meta["content"]
                elif og and og.get("content"):
                    text = og["content"]
                else:
                    # Final fallback: extract all visible text
                    texts = soup.stripped_strings
                    text = " ".join(list(texts))
            return text[:1000]  # Cap output length
        except requests.RequestException as e:
            logging.warning(f"Fetch failed for {url}: {e}", exc_info=True)
            return ""

    def verify_source(self, url: str) -> bool:
        """Check if a URL's domain is in trusted lists."""
        netloc = urlparse(url).netloc.lower()
        # Remove port if present
        if ':' in netloc:
            netloc = netloc.split(':')[0]
        if not netloc:
            return False
        for trusted in self.trusted_base_domains:
            if netloc == trusted or netloc.endswith('.' + trusted):
                return True
        for suffix in self.trusted_domain_suffixes:
            if netloc.endswith(suffix):
                return True
        return False

# --- Script test block ---
if __name__ == "__main__":
    print("Testing WebResearchService...")
    service = WebResearchService()
    results = service.search_and_fetch("Wikipedia", num_results=1)
    for r in results:
        print(f"URL: {r['url']}")
        print(f"Title: {r['title']}")
        print(f"Verified: {r['verified']}")
        print(f"Snippet: {r['snippet']}")
        if r['error']:
            print(f"Error: {r['error']}")
        print('-' * 40)
