import sys
import asyncio
import os
from pathlib import Path

# Import the WebSearchService
from services.web_search_service import WebSearchService

async def test_tavily_search():
    try:
        # Initialize the search service with Tavily (not dummy)
        search_service = WebSearchService(provider_name="tavily")
        
        # Perform a search
        query = "WHat is the result of the asia cup match between ban vs sri today?"
        print(f"Searching for: {query}")
        
        results = await search_service.search(
            query=query,
            k=5,
            region="auto",
            language="en",
            timeout_seconds=15
        )
        
        print(f"\nFound {len(results)} results:")
        for idx, result in enumerate(results, 1):
            print(f"\n--- Result {idx} ---")
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            print(f"Site: {result.site_name}")
            if result.snippet:
                snippet = result.snippet[:100] + "..." if len(result.snippet) > 100 else result.snippet
                print(f"Snippet: {snippet}")
            print(f"Score: {result.score}")
            
        assert len(results) > 0, "No search results returned"
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tavily_search())
    print(f"\nTest {'succeeded' if success else 'failed'}")