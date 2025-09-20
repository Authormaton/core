import os
import pytest
from services.web_search_service import WebSearchService

@pytest.mark.asyncio
async def test_tavily_search_integration():
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY not set; skipping Tavily integration test")
    search_service = WebSearchService(provider_name="tavily")
    # Stable, non-time-sensitive query
    results = await search_service.search(
        query="Tavily API documentation",
        k=3,
        region="auto",
        language="en",
        timeout_seconds=15,
    )
    assert results and len(results) > 0