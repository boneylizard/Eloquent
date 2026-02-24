# web_search_service.py
import asyncio
import logging
import re
import time
from typing import List, Dict, Optional, Any, Callable
from urllib.parse import urljoin, urlparse, unquote, parse_qs
import httpx
from bs4 import BeautifulSoup
import json
from dataclasses import dataclass, asdict
from html import unescape

logger = logging.getLogger(__name__)

# ============================================================================
# TOOL DEFINITION - For models that support function/tool calling
# ============================================================================

WEB_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use this when you need up-to-date information, facts you're unsure about, recent events, or to verify claims. The search will return relevant web pages with their content.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more optimized search queries. Break complex questions into multiple targeted searches. Use specific keywords, not full sentences. Example: for 'What's the weather like in Paris and what should I pack?', use ['Paris weather forecast', 'Paris travel packing list']",
                    "minItems": 1,
                    "maxItems": 3
                },
                "search_intent": {
                    "type": "string",
                    "description": "Brief description of what information you're looking for and why",
                }
            },
            "required": ["search_queries", "search_intent"]
        }
    }
}

# Simpler single-query version for basic tool calling
WEB_SEARCH_TOOL_SIMPLE = {
    "type": "function", 
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use when you need up-to-date facts, recent events, or to verify claims.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "An optimized search query. Use specific keywords, not full sentences. Example: instead of 'What is the current president of France?', use 'France president 2024'"
                }
            },
            "required": ["query"]
        }
    }
}

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: Optional[str] = None  # Full scraped content
    scraped_successfully: bool = False
    publisher: Optional[str] = None  # Publisher/source name (e.g. from RSS <source>), for news display

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass  
class SmartSearchResult:
    """Result from smart search including query optimization info."""
    original_prompt: str
    optimized_queries: List[str]
    search_intent: str
    results: List[SearchResult]
    formatted_context: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "optimized_queries": self.optimized_queries,
            "search_intent": self.search_intent,
            "results": [r.to_dict() for r in self.results],
            "formatted_context": self.formatted_context
        }
    
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
        
        # LLM function for query optimization (set via set_llm_function)
        self._llm_function: Optional[Callable] = None
        
        # Prompt for query optimization
        self.QUERY_OPTIMIZATION_PROMPT = """You are a search query optimizer. Given a user's question or request, generate optimized search queries that will find the most relevant information.

RULES:
1. Convert natural language questions into keyword-focused search queries
2. Remove filler words, keep essential terms
3. Add context terms that help narrow results (e.g., year, location, specific domain)
4. For complex questions, break into 1-3 separate targeted queries
5. Use quotation marks for exact phrases when needed
6. Include alternative phrasings if the topic could be searched differently

USER INPUT: {user_prompt}

Respond in this exact JSON format only, no other text:
{{"queries": ["query1", "query2"], "intent": "brief description of what user wants to find"}}"""

        logger.info("🔎 Web search configured: DuckDuckGo")

    def set_llm_function(self, llm_func: Callable):
        """Set the LLM function used for query optimization.
        
        The function should accept (prompt: str) and return the generated text.
        Can be sync or async.
        """
        self._llm_function = llm_func
        logger.info("🔍 LLM function set for smart query optimization")
    
    async def optimize_query(self, user_prompt: str) -> tuple[List[str], str]:
        """Use LLM to convert user prompt into optimized search queries.
        
        Returns: (list of optimized queries, search intent description)
        """
        if not self._llm_function:
            # Fallback: basic query cleaning if no LLM available
            logger.warning("⚠️ No LLM function set, using basic query optimization")
            return self._basic_query_optimization(user_prompt)
        
        try:
            prompt = self.QUERY_OPTIMIZATION_PROMPT.format(user_prompt=user_prompt)
            
            # Call LLM (handle both sync and async)
            if asyncio.iscoroutinefunction(self._llm_function):
                response = await self._llm_function(prompt)
            else:
                response = self._llm_function(prompt)
            
            # Parse JSON response
            # Try to extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^{}]*"queries"[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                queries = parsed.get("queries", [user_prompt])
                intent = parsed.get("intent", "general search")
                
                # Validate queries
                if not queries or not isinstance(queries, list):
                    queries = [user_prompt]
                
                # Limit to 3 queries max
                queries = queries[:3]
                
                logger.info(f"🧠 Optimized '{user_prompt[:50]}...' → {queries}")
                return queries, intent
            else:
                logger.warning(f"⚠️ Could not parse LLM response, using original query")
                return [user_prompt], "general search"
                
        except Exception as e:
            logger.error(f"❌ Query optimization error: {e}")
            return self._basic_query_optimization(user_prompt)
    
    def _basic_query_optimization(self, user_prompt: str) -> tuple[List[str], str]:
        """Parse the user's true meaning into search-friendly query/queries (no LLM)."""
        text = user_prompt.strip()
        if not text:
            return [user_prompt], "general search"
        lower = text.lower()

        # Intent patterns: map conversational phrasing -> search intent (query suffix or full query)
        # Order matters: more specific first
        intent_rewrites = [
            (r"^(?:what'?s?|what is) the latest (?:on|about|with)\s+(.+)$", r"\1 latest news"),
            (r"^(?:any )?latest (?:on|about|news? about)\s+(.+)$", r"\1 latest"),
            (r"^how (?:do i|can i|to)\s+(.+)$", r"how to \1"),
            (r"^who is\s+(.+)$", r"\1"),
            (r"^what (?:do you think about|is your take on)\s+(.+)$", r"\1"),
            (r"^(?:tell me |find |search for |look up |google )(.+)$", r"\1"),
            (r"^(?:can you |could you |please )(?:tell me |find |search |look up )?(.+)$", r"\1"),
            (r"^(?:what is|what are|why is|why are|where is|when is)\s+(.+)$", r"\1"),
            (r"^anything (?:about|on)\s+(.+)$", r"\1"),
            (r"^(?:explain|describe)\s+(.+)$", r"\1"),
            (r"^news (?:about|on)\s+(.+)$", r"\1 news"),
        ]
        for pattern, repl in intent_rewrites:
            m = re.match(pattern, lower, re.IGNORECASE)
            if m:
                cleaned = re.sub(pattern, repl, lower, count=1, flags=re.IGNORECASE)
                cleaned = cleaned.strip().rstrip("?")
                if cleaned:
                    return [cleaned], "intent-based search"
                break

        # Generic cleanup: remove leading filler and trailing "?"
        removals = [
            r"^(?:can you |could you |please |i want to know |tell me |what is |what are |how do |how does |how can |why is |why are |where is |where are |when is |when did )",
            r"\?+$",
            r"^(?:search for |look up |find |google )",
        ]
        cleaned = lower
        for pattern in removals:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = text.strip().rstrip("?")
        return [cleaned], "general search"
    
    def _decode_duckduckgo_redirect(self, url: str) -> str:
        """Extract real URL from DuckDuckGo redirect link (l/?uddg=...)."""
        if not url:
            return url
        try:
            if "uddg=" in url:
                parsed = urlparse(url)
                qs = parse_qs(parsed.query)
                uddg = (qs.get("uddg") or [None])[0]
                if uddg:
                    return unquote(uddg)
            if url.startswith("//"):
                return "https:" + url
        except Exception:
            pass
        return url

    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search DuckDuckGo and return results. Prefer HTML search (organic results); Instant Answer API often empty."""
        try:
            logger.info(f"🔍 Searching DuckDuckGo for: '{query}'")
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                # Try HTML search first (organic results; Instant Answer API often returns empty for news)
                results = await self._search_duckduckgo_html(client, query, max_results, headers)
                if len(results) < max_results:
                    params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
                    response = await client.get("https://api.duckduckgo.com/", params=params, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    for topic in data.get("RelatedTopics", [])[: max_results - len(results)]:
                        if isinstance(topic, dict) and topic.get("FirstURL"):
                            url = topic["FirstURL"]
                            if self._is_scrapeable_url(url) and url not in {r.url for r in results}:
                                text = topic.get("Text", "")
                                results.append(SearchResult(title=text.split(" - ")[0] if " - " in text else text, url=url, snippet=text))
                logger.info(f"🔍 Found {len(results)} search results")
                return results[:max_results]
        except Exception as e:
            logger.error(f"❌ DuckDuckGo search error: {e}")
            return []

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search the web using DuckDuckGo."""
        return await self.search_duckduckgo(query, max_results)
    
    async def search_news(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """News-focused search with RSS fallbacks before DuckDuckGo."""
        results = await self._search_google_news_rss(query, max_results)
        if not results:
            results = await self._search_bing_news_rss(query, max_results)
        if not results:
            results = await self.search_duckduckgo(query, max_results)
        if not results:
            broadened = self._broaden_news_query(query)
            if broadened and broadened != query:
                results = await self._search_google_news_rss(broadened, max_results)
                if not results:
                    results = await self._search_bing_news_rss(broadened, max_results)
                if not results:
                    results = await self.search_duckduckgo(broadened, max_results)
        return results

    def _broaden_news_query(self, query: str) -> str:
        q = query or ""
        q = re.sub(r"\b(news|headlines?)\b", "", q, flags=re.IGNORECASE)
        q = re.sub(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            "",
            q,
            flags=re.IGNORECASE
        )
        q = re.sub(r"\b(19|20)\d{2}\b", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q or query

    async def _search_google_news_rss(self, query: str, max_results: int = 10) -> List[SearchResult]:
        url = "https://news.google.com/rss/search"
        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en"
        }
        return await self._fetch_rss_results(url, params, max_results)

    async def _search_bing_news_rss(self, query: str, max_results: int = 10) -> List[SearchResult]:
        url = "https://www.bing.com/news/search"
        params = {
            "q": query,
            "format": "rss",
            "mkt": "en-US"
        }
        return await self._fetch_rss_results(url, params, max_results)

    async def _fetch_rss_results(self, url: str, params: Dict[str, Any], max_results: int) -> List[SearchResult]:
        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code not in (200, 202):
                    return []
                soup = BeautifulSoup(response.text, "xml")
                items = soup.find_all("item")
                results: List[SearchResult] = []

                for item in items:
                    title_node = item.find("title")
                    link_node = item.find("link")
                    desc_node = item.find("description") or item.find("summary")

                    title = unescape(title_node.get_text(strip=True)) if title_node else "No title"
                    link = link_node.get_text(strip=True) if link_node else ""
                    if not link:
                        continue

                    snippet = ""
                    if desc_node and desc_node.get_text(strip=True):
                        snippet_html = unescape(desc_node.get_text())
                        snippet = BeautifulSoup(snippet_html, "html.parser").get_text(" ", strip=True)[:500]

                    publisher = None
                    source_node = item.find("source")
                    if source_node and source_node.get_text(strip=True):
                        publisher = unescape(source_node.get_text(strip=True))

                    results.append(SearchResult(title=title or "No title", url=link, snippet=snippet, publisher=publisher))
                    if len(results) >= max_results:
                        break

                return results
        except Exception:
            return []

    async def _search_duckduckgo_html(self, client: httpx.AsyncClient, query: str, max_results: int, headers: Optional[Dict[str, str]] = None) -> List[SearchResult]:
        """HTML search for DuckDuckGo. Decode uddg redirect URLs to get real links."""
        try:
            await asyncio.sleep(self.rate_limit_delay)
            req_headers = headers or {"User-Agent": self.user_agent}
            params = {"q": query}
            search_urls = [
                "https://html.duckduckgo.com/html/",
                "https://duckduckgo.com/html/"
            ]

            for search_url in search_urls:
                response = await client.get(
                    search_url,
                    params=params,
                    headers=req_headers,
                )
                if response.status_code not in (200, 202):
                    continue
                results = self._parse_duckduckgo_html_results(response.text, max_results)
                if results:
                    return results

            # Fallback: DDG lite often succeeds when HTML endpoint returns empty/202
            return await self._search_duckduckgo_lite(client, query, max_results, req_headers)
        except Exception as e:
            logger.error(f"DuckDuckGo HTML search error: {e}")
            return []

    def _parse_duckduckgo_html_results(self, html: str, max_results: int) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[SearchResult] = []
        seen_urls = set()

        for result_div in soup.find_all("div", class_="result")[: max_results * 3]:
            title_elem = result_div.find("a", class_="result__a")
            snippet_elem = result_div.find("a", class_="result__snippet") or result_div.find("div", class_="result__snippet")
            href = title_elem.get("href") if title_elem else None
            if not href or not title_elem:
                continue
            real_url = self._decode_duckduckgo_redirect(href)
            if not real_url or real_url in seen_urls or not self._is_scrapeable_url(real_url):
                continue
            seen_urls.add(real_url)
            title = title_elem.get_text(strip=True)
            snippet = (snippet_elem.get_text(strip=True) if snippet_elem else "")[:500]
            results.append(SearchResult(title=title or "No title", url=real_url, snippet=snippet))
            if len(results) >= max_results:
                return results

        return results

    async def _search_duckduckgo_lite(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
        headers: Optional[Dict[str, str]] = None
    ) -> List[SearchResult]:
        try:
            await asyncio.sleep(self.rate_limit_delay)
            params = {"q": query}
            response = await client.get(
                "https://lite.duckduckgo.com/lite/",
                params=params,
                headers=headers or {"User-Agent": self.user_agent},
            )
            if response.status_code not in (200, 202):
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            results: List[SearchResult] = []
            seen_urls = set()

            for link in soup.find_all("a", class_="result-link"):
                href = link.get("href")
                if not href:
                    continue
                real_url = self._decode_duckduckgo_redirect(href)
                if not real_url or real_url in seen_urls or not self._is_scrapeable_url(real_url):
                    continue
                seen_urls.add(real_url)
                title = link.get_text(strip=True)
                snippet = ""

                parent_row = link.find_parent("tr")
                if parent_row:
                    snippet_row = parent_row.find_next_sibling("tr")
                    if snippet_row:
                        snippet_cell = snippet_row.find("td", class_="result-snippet")
                        if snippet_cell:
                            snippet = snippet_cell.get_text(strip=True)[:500]

                results.append(SearchResult(title=title or "No title", url=real_url, snippet=snippet))
                if len(results) >= max_results:
                    return results

            for link in soup.find_all("a", class_="result__a"):
                href = link.get("href")
                if not href:
                    continue
                real_url = self._decode_duckduckgo_redirect(href)
                if not real_url or real_url in seen_urls or not self._is_scrapeable_url(real_url):
                    continue
                seen_urls.add(real_url)
                title = link.get_text(strip=True)
                results.append(SearchResult(title=title or "No title", url=real_url, snippet=""))
                if len(results) >= max_results:
                    return results

            return results
        except Exception:
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
    
    def format_smart_search_context(
        self, 
        original_prompt: str, 
        optimized_queries: List[str], 
        intent: str, 
        results: List[SearchResult]
    ) -> str:
        """Format smart search results with query optimization info."""
        if not results:
            return f"WEB SEARCH for '{original_prompt}':\nNo relevant results found."
        
        context_parts = [
            f"WEB SEARCH RESULTS",
            f"Original query: {original_prompt}",
            f"Search intent: {intent}",
            f"Optimized searches: {', '.join(optimized_queries)}",
            f"---"
        ]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n[{i}] {result.title}")
            context_parts.append(f"URL: {result.url}")
            
            if result.scraped_successfully and result.content:
                content_preview = result.content[:800] + "..." if len(result.content) > 800 else result.content
                context_parts.append(f"Content: {content_preview}")
            elif result.snippet:
                context_parts.append(f"Snippet: {result.snippet}")
            
            context_parts.append("")
        
        return '\n'.join(context_parts)
    
    def format_tool_response(
        self, 
        queries: List[str], 
        intent: str, 
        results: List[SearchResult]
    ) -> str:
        """Format search results for tool call response."""
        if not results:
            return f"No results found for: {', '.join(queries)}"
        
        context_parts = [
            f"Search completed for: {', '.join(queries)}",
            f"Intent: {intent}",
            f"Found {len(results)} relevant results:",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result.title}")
            context_parts.append(f"URL: {result.url}")
            if getattr(result, "publisher", None):
                context_parts.append(f"Publisher: {result.publisher}")

            if result.scraped_successfully and result.content:
                # For tool responses, include more content
                content_preview = result.content[:1200] + "..." if len(result.content) > 1200 else result.content
                context_parts.append(f"{content_preview}")
            elif result.snippet:
                context_parts.append(f"{result.snippet}")
            
            context_parts.append("")
        
        return '\n'.join(context_parts)

# Global service instance
web_search_service = WebSearchService()

async def perform_web_search(query: str, max_results: int = 5) -> str:
    """Main function to perform web search and return formatted context."""
    try:
        logger.info(f"🌐 Starting web search for: '{query}'")
        
        # Search
        results = await web_search_service.search(query, max_results)
        
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


async def perform_smart_web_search(
    user_prompt: str, 
    max_results: int = 5,
    use_optimization: bool = True
) -> SmartSearchResult:
    """
    Smart web search that optimizes the user's query before searching.
    
    Args:
        user_prompt: The user's natural language question/request
        max_results: Maximum results per query
        use_optimization: Whether to use LLM query optimization
        
    Returns:
        SmartSearchResult with optimized queries and search results
    """
    try:
        logger.info(f"🧠 Starting smart web search for: '{user_prompt[:100]}...'")
        
        # Step 1: Optimize the query using LLM
        if use_optimization:
            optimized_queries, search_intent = await web_search_service.optimize_query(user_prompt)
        else:
            optimized_queries = [user_prompt]
            search_intent = "direct search"
        
        logger.info(f"🔍 Optimized queries: {optimized_queries}")
        
        # Step 2: Search for each optimized query
        all_results: List[SearchResult] = []
        seen_urls = set()
        
        for query in optimized_queries:
            results = await web_search_service.search(query, max_results)
            
            # Deduplicate by URL
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
        
        if not all_results:
            return SmartSearchResult(
                original_prompt=user_prompt,
                optimized_queries=optimized_queries,
                search_intent=search_intent,
                results=[],
                formatted_context=f"WEB SEARCH for '{user_prompt}':\nNo results found."
            )
        
        # Step 3: Scrape content from top results
        # Limit scraping to avoid timeout
        results_to_scrape = all_results[:max_results]
        results_with_content = await web_search_service.scrape_content(results_to_scrape)
        
        # Step 4: Format context
        formatted_context = web_search_service.format_smart_search_context(
            user_prompt, optimized_queries, search_intent, results_with_content
        )
        
        logger.info(f"🧠 Smart search completed: {len(results_with_content)} results from {len(optimized_queries)} queries")
        
        return SmartSearchResult(
            original_prompt=user_prompt,
            optimized_queries=optimized_queries,
            search_intent=search_intent,
            results=results_with_content,
            formatted_context=formatted_context
        )
        
    except Exception as e:
        logger.error(f"❌ Smart web search error: {e}")
        return SmartSearchResult(
            original_prompt=user_prompt,
            optimized_queries=[user_prompt],
            search_intent="error",
            results=[],
            formatted_context=f"WEB SEARCH for '{user_prompt}':\nSearch failed: {str(e)}"
        )


async def handle_web_search_tool_call(arguments: Dict[str, Any], max_results: int = 5, news: bool = False) -> str:
    """
    Handle a web_search tool call from a model.
    
    Accepts arguments in either format:
    - {"query": "single query"} (simple format)
    - {"search_queries": ["q1", "q2"], "search_intent": "..."} (advanced format)
    
    Returns formatted search results as a string.
    """
    try:
        if isinstance(arguments, str):
            arguments = {"query": arguments}
        if isinstance(arguments, dict) and "query" in arguments and isinstance(arguments["query"], str):
            raw = arguments["query"]
            if "{" in raw and "query" in raw:
                queries = re.findall(r'"query"\s*:\s*"([^"]+)"', raw)
                if queries:
                    arguments = {"search_queries": queries}

        # Handle both tool definition formats
        if "search_queries" in arguments:
            queries = arguments["search_queries"]
            intent = arguments.get("search_intent", "general search")
            if isinstance(queries, str):
                queries = [queries]
        elif "query" in arguments:
            queries = [arguments["query"]]
            intent = "direct search"
        else:
            return "Error: No search query provided in tool call arguments"
        
        logger.info(f"🔧 Handling web_search tool call ({'news' if news else 'web'}): {queries}")
        
        # Search for each query
        all_results: List[SearchResult] = []
        seen_urls = set()
        
        search_fn = web_search_service.search_news if news else web_search_service.search
        for query in queries[:3]:  # Limit to 3 queries
            results = await search_fn(query, max_results)
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
        
        if not all_results:
            return f"Web search found no results for: {queries}"

        if news:
            # For news, rely on RSS titles/snippets; skip scraping to avoid Google News blocks.
            top_titles = [r.title for r in all_results[:5] if r.title]
            logger.info(f"🗞️ News search returned {len(all_results)} results. Top: {top_titles}")
            return web_search_service.format_tool_response(queries, intent, all_results[:max_results])
        
        # Scrape top results
        results_to_scrape = all_results[:max_results]
        results_with_content = await web_search_service.scrape_content(results_to_scrape)
        
        # Format for tool response
        formatted = web_search_service.format_tool_response(queries, intent, results_with_content)
        
        return formatted
        
    except Exception as e:
        logger.error(f"❌ Web search tool call error: {e}")
        return f"Web search failed: {str(e)}"


def get_web_search_tool_definition(simple: bool = False) -> Dict[str, Any]:
    """Get the tool definition for web search.
    
    Args:
        simple: If True, returns the simpler single-query version
        
    Returns:
        Tool definition dict compatible with OpenAI/Anthropic tool calling format
    """
    return WEB_SEARCH_TOOL_SIMPLE if simple else WEB_SEARCH_TOOL_DEFINITION


def set_web_search_llm(llm_function: Callable):
    """Configure the LLM function for smart query optimization.
    
    Example:
        async def my_llm(prompt):
            response = await my_model.generate(prompt)
            return response.text
            
        set_web_search_llm(my_llm)
    """
    web_search_service.set_llm_function(llm_function)
