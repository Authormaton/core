import os
import asyncio
import pytest
from services.web_search_service import WebSearchService

@pytest.mark.asyncio
@pytest.mark.integration
async def test_tavily_search_returns_results():
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY not set")
    search_service = WebSearchService(provider_name="tavily")
    results = await search_service.search(
        query="OpenAI API documentation",
        k=3,
        region="auto",
        language="en",
        timeout_seconds=15,
    )
    assert isinstance(results, list)
    assert len(results) > 0

# No __main__ runner in test files