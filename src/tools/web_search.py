import os
import sys
import asyncio
from typing import List, Dict, Optional

from langchain_community.tools import DuckDuckGoSearchResults
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from dotenv import load_dotenv

load_dotenv()

class AdvancedWebCrawler:
    def __init__(self, 
                 max_search_results: int = 5, 
                 word_count_threshold: int = 50,
                 content_filter_type: str = 'pruning',
                 filter_threshold: float = 0.48):
        """
        Initialize the Advanced Web Crawler
        
        Args:
            max_search_results (int): Maximum number of search results to process
            word_count_threshold (int): Minimum word count for crawled content
            content_filter_type (str): Type of content filter ('pruning' or 'bm25')
            filter_threshold (float): Threshold for content filtering
        """
        self.max_search_results = max_search_results
        self.word_count_threshold = word_count_threshold
        self.content_filter_type = content_filter_type
        self.filter_threshold = filter_threshold

    def _create_web_search_tool(self):
        """
        Create a web search tool using DuckDuckGo
        
        Returns:
            DuckDuckGoSearchResults: Web search tool
        """
        return DuckDuckGoSearchResults(max_results=self.max_search_results, output_format="list")

    def _create_content_filter(self, user_query: Optional[str] = None):
        """
        Create content filter based on specified type
        
        Args:
            user_query (Optional[str]): Query to use for BM25 filtering
        
        Returns:
            Content filter strategy
        """
        if self.content_filter_type == 'bm25' and user_query:
            return BM25ContentFilter(
                user_query=user_query, 
                bm25_threshold=self.filter_threshold
            )
        else:
            return PruningContentFilter(
                threshold=self.filter_threshold, 
                threshold_type="fixed", 
                min_word_threshold=self.word_count_threshold
            )

    async def crawl_urls(self, urls: List[str], user_query: Optional[str] = None):
        """
        Crawl multiple URLs with content filtering
        
        Args:
            urls (List[str]): List of URLs to crawl
            user_query (Optional[str]): Query used for BM25 content filtering
        
        Returns:
            List of crawl results
        """
        async with AsyncWebCrawler(
            browser_type="chromium", 
            headless=True, 
            verbose=True
        ) as crawler:
            # Create appropriate content filter
            content_filter = self._create_content_filter(user_query)
            
            # Run crawling for multiple URLs
            results = await crawler.arun_many(
                urls=urls,
                word_count_threshold=self.word_count_threshold,
                bypass_cache=True,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=content_filter
                ),
                cache_mode=CacheMode.DISABLED,
                exclude_external_links=True,
                remove_overlay_elements=True,
                simulate_user=True,
                magic=True
            )
            
            # Process and return crawl results
            processed_results = []
            for result in results:
                crawl_result = {
                    "url": result.url,
                    "success": result.success,
                    "title": result.metadata.get('title', 'N/A'),
                    "content": result.markdown_v2.raw_markdown if result.success else result.error_message,
                    "word_count": len(result.markdown_v2.raw_markdown.split()) if result.success else 0,
                    "links": {
                        "internal": len(result.links.get('internal', [])),
                        "external": len(result.links.get('external', []))
                    },
                    "images": len(result.media.get('images', []))
                }
                processed_results.append(crawl_result)
            
            return processed_results

    async def search_and_crawl(self, query: str) -> List[Dict]:
        """
        Perform web search and crawl the results
        
        Args:
            query (str): Search query
        
        Returns:
            List of crawled content results
        """
        # Perform web search
        search_tool = self._create_web_search_tool()
        try:
            search_results = search_tool.invoke({"query": query})
            
            # Extract URLs from search results
            urls = [result['link'] for result in search_results]
            print(f"Found {len(urls)} URLs for query: {query}")
            
            # Crawl URLs
            crawl_results = await self.crawl_urls(urls, user_query=query)
            
            return crawl_results
        
        except Exception as e:
            print(f"Web search and crawl error: {e}")
            return []

def main():
    # Example usage
    crawler = AdvancedWebCrawler(
        max_search_results=5,
        word_count_threshold=50,
        content_filter_type='f',
        filter_threshold=0.48
    )
    
    test_queries = [
        "Latest developments in AI agents",
        "Today's weather forecast in Kolkata",
    ]
    
    for query in test_queries:
        # Run search and crawl asynchronously
        results = asyncio.run(crawler.search_and_crawl(query))
        
        print(f"\nResults for query: {query}")
        for result in results:
            print(f"URL: {result['url']}")
            print(f"Success: {result['success']}")
            print(f"Title: {result['title']}")
            print(f"Word Count: {result['word_count']}")
            print(f"Content Preview: {result['content'][:500]}...\n")

if __name__ == "__main__":
    main()