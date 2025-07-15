# web_search_service.py
import asyncio
import logging
import re
import time
from typing import List, Dict, Optional, Any
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: Optional[str] = None  # Full scraped content
    scraped_successfully: bool = False
    
class WebSearchService:
    def __init__(self):
        self.session_timeout = 30.0
        self.scrape_timeout = 15.0
        self.max_content_length = 10000  # Limit content size
        self.rate_limit_delay = 1.0  # Delay between requests
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
        # Domains to avoid scraping (add more as needed)
        self.blocked_domains = {
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'pinterest.com', 'tiktok.com', 'snapchat.com'
        }
    
    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search DuckDuckGo and return results."""
        try:
            logger.info(f"🔍 Searching DuckDuckGo for: '{query}'")
            
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                # DuckDuckGo instant answer API
                params = {
                    'q': query,
                    'format': 'json',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                response = await client.get(
                    'https://api.duckduckgo.com/',
                    params=params,
                    headers={'User-Agent': self.user_agent}
                )
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                # Get related topics (these often have good URLs)
                related_topics = data.get('RelatedTopics', [])
                for topic in related_topics[:max_results]:
                    if isinstance(topic, dict) and 'FirstURL' in topic:
                        result = SearchResult(
                            title=topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                            url=topic.get('FirstURL', ''),
                            snippet=topic.get('Text', '')
                        )
                        if result.url and self._is_scrapeable_url(result.url):
                            results.append(result)
                
                # If we don't have enough results, try the HTML search
                if len(results) < max_results:
                    html_results = await self._search_duckduckgo_html(client, query, max_results - len(results))
                    results.extend(html_results)
                
                logger.info(f"🔍 Found {len(results)} search results")
                return results[:max_results]
                
        except Exception as e:
            logger.error(f"❌ DuckDuckGo search error: {e}")
            return []
    
    async def _search_duckduckgo_html(self, client: httpx.AsyncClient, query: str, max_results: int) -> List[SearchResult]:
        """Fallback HTML search for DuckDuckGo."""
        try:
            # Wait for rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            params = {'q': query}
            response = await client.get(
                'https://html.duckduckgo.com/html/',
                params=params,
                headers={'User-Agent': self.user_agent}
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse search results
            for result_div in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result_div.find('a', class_='result__a')
                snippet_elem = result_div.find('a', class_='result__snippet')
                
                if title_elem and title_elem.get('href'):
                    url = title_elem.get('href')
                    # DuckDuckGo sometimes uses redirect URLs
                    if url.startswith('//duckduckgo.com/l/?uddg='):
                        continue  # Skip redirect URLs for now
                    
                    result = SearchResult(
                        title=title_elem.get_text(strip=True),
                        url=url,
                        snippet=snippet_elem.get_text(strip=True) if snippet_elem else ''
                    )
                    
                    if self._is_scrapeable_url(result.url):
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ DuckDuckGo HTML search error: {e}")
            return []
    
    def _is_scrapeable_url(self, url: str) -> bool:
        """Check if URL is safe to scrape."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check against blocked domains
            if domain in self.blocked_domains:
                return False
            
            # Only HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Avoid certain file types
            path = parsed.path.lower()
            blocked_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar']
            if any(path.endswith(ext) for ext in blocked_extensions):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def scrape_content(self, results: List[SearchResult]) -> List[SearchResult]:
        """Scrape content from search result URLs."""
        logger.info(f"🕷️ Scraping content from {len(results)} URLs")
        
        async with httpx.AsyncClient(timeout=self.scrape_timeout) as client:
            for result in results:
                try:
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    logger.debug(f"Scraping: {result.url}")
                    response = await client.get(
                        result.url,
                        headers={'User-Agent': self.user_agent},
                        follow_redirects=True
                    )
                    response.raise_for_status()
                    
                    # Parse content
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove unwanted elements
                    for elem in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        elem.decompose()
                    
                    # Extract main content
                    content = self._extract_main_content(soup)
                    
                    if content and len(content.strip()) > 50:  # Minimum content threshold
                        result.content = content[:self.max_content_length]
                        result.scraped_successfully = True
                        logger.debug(f"✅ Scraped {len(result.content)} chars from {result.url}")
                    else:
                        logger.debug(f"⚠️ Insufficient content from {result.url}")
                        
                except Exception as e:
                    logger.debug(f"❌ Scraping failed for {result.url}: {e}")
                    result.scraped_successfully = False
        
        successful_scrapes = sum(1 for r in results if r.scraped_successfully)
        logger.info(f"🕷️ Successfully scraped {successful_scrapes}/{len(results)} URLs")
        
        return results
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from parsed HTML."""
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.post-content', 
            '.entry-content', '.article-content', '.main-content'
        ]
        
        # Try to find main content area
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return ""
        
        # Extract text while preserving some structure
        text_parts = []
        for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = elem.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short text
                text_parts.append(text)
        
        return '\n\n'.join(text_parts)
    
    def format_search_context(self, query: str, results: List[SearchResult]) -> str:
        """Format search results for LLM context."""
        if not results:
            return f"WEB SEARCH RESULTS for '{query}':\nNo relevant results found."
        
        context_parts = [f"WEB SEARCH RESULTS for '{query}':"]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n[{i}] {result.title}")
            context_parts.append(f"URL: {result.url}")
            
            if result.scraped_successfully and result.content:
                # Use scraped content if available
                content_preview = result.content[:800] + "..." if len(result.content) > 800 else result.content
                context_parts.append(f"Content: {content_preview}")
            elif result.snippet:
                # Fallback to snippet
                context_parts.append(f"Snippet: {result.snippet}")
            
            context_parts.append("")  # Empty line between results
        
        return '\n'.join(context_parts)

# Global service instance
web_search_service = WebSearchService()

async def perform_web_search(query: str, max_results: int = 5) -> str:
    """Main function to perform web search and return formatted context."""
    try:
        logger.info(f"🌐 Starting web search for: '{query}'")
        
        # Search
        results = await web_search_service.search_duckduckgo(query, max_results)
        
        if not results:
            return f"WEB SEARCH RESULTS for '{query}':\nNo results found or search service unavailable."
        
        # Scrape content
        results_with_content = await web_search_service.scrape_content(results)
        
        # Format for LLM
        formatted_context = web_search_service.format_search_context(query, results_with_content)
        
        logger.info(f"🌐 Web search completed for: '{query}'")
        return formatted_context
        
    except Exception as e:
        logger.error(f"❌ Web search error: {e}")
        return f"WEB SEARCH RESULTS for '{query}':\nSearch failed due to technical error: {str(e)}"