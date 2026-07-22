# web_search_service.py
#
# Multi-engine web search with:
#   - DuckDuckGo, Brave Search, Google Custom Search, Bing, Serper, Tavily
#   - LLM-powered query optimization (required, no regex fallback)
#   - Multi-strategy content extraction (readability, schema.org, JSON-LD, OpenGraph)
#   - Result ranking, source credibility, dedup, caching
#   - Jina Reader + Wayback Machine fallbacks for blocked pages

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, unquote, parse_qs, urlencode

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

_dotenv_loaded = False
def _ensure_dotenv():
    global _dotenv_loaded
    if not _dotenv_loaded:
        loaded = load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
        if not loaded:
            loaded = load_dotenv()
        _dotenv_loaded = True
        if loaded:
            logger = logging.getLogger(__name__)
            logger.info("Loaded .env file for web search API keys")
_ensure_dotenv()

logger = logging.getLogger(__name__)

# ============================================================================
# TOOL DEFINITIONS
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

WEB_SEARCH_TOOL_UNIFIED = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use specific keywords. "
            "For breaking news or politics use web_search_news instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Primary search keywords (preferred for GLM/DeepSeek)",
                },
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: 1-3 keyword queries for broader research",
                    "maxItems": 3,
                },
                "search_intent": {
                    "type": "string",
                    "description": "Brief note on what you are trying to find",
                },
            },
            "required": ["query"],
        },
    },
}

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
                    "description": "An optimized search query. Use specific keywords, not full sentences."
                }
            },
            "required": ["query"]
        }
    }
}

WEB_FETCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch full content from a specific URL. Use when you need to read the complete article or page content from a search result.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from"
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 25000)",
                    "default": 25000
                }
            },
            "required": ["url"]
        }
    }
}

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    scraped_successfully: bool = False
    publisher: Optional[str] = None
    engine: str = "duckduckgo"
    published_date: Optional[str] = None
    relevance_score: float = 1.0
    credibility_score: float = 0.5
    content_type: str = "article"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SmartSearchResult:
    original_prompt: str
    optimized_queries: List[str]
    search_intent: str
    results: List[SearchResult]
    formatted_context: str
    cache_hit: bool = False
    engines_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "optimized_queries": self.optimized_queries,
            "search_intent": self.search_intent,
            "results": [r.to_dict() for r in self.results],
            "formatted_context": self.formatted_context,
            "cache_hit": self.cache_hit,
            "engines_used": self.engines_used,
        }

# ============================================================================
# CONSTANTS
# ============================================================================

_ERROR_PAGE_MARKERS = (
    "403 forbidden", "access denied", "access to this page has been denied",
    "just a moment", "cloudflare", "captcha", "enable javascript",
    "bot detection", "request blocked", "unusual traffic", "verify you are human",
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]

CREDIBLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "wsj.com", "washingtonpost.com", "economist.com", "nature.com",
    "science.org", "sciencedaily.com", "nasa.gov", "nih.gov", "who.int",
    "un.org", "worldbank.org", "imf.org", "arxiv.org", "ieee.org",
    "acm.org", "springer.com", "elsevier.com", "cambridge.org",
    "oup.com", "sagepub.com", "tandfonline.com", "wiley.com",
}

LOW_CREDIBILITY_DOMAINS = {
    "wikipedia.org",  # better than nothing but not definitive
    "medium.com", "quora.com", "reddit.com", "tumblr.com",
    "wordpress.com", "blogspot.com", "wixsite.com", "weebly.com",
}

# ============================================================================
# CACHE
# ============================================================================

class SearchCache:
    def __init__(self, ttl_seconds: int = 300, max_entries: int = 500):
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self.ttl = ttl_seconds
        self.max_entries = max_entries

    def _key(self, *args, **kwargs) -> str:
        raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, *args, **kwargs) -> Optional[Any]:
        key = self._key(*args, **kwargs)
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        if time.time() - ts > self.ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, value: Any, *args, **kwargs) -> None:
        key = self._key(*args, **kwargs)
        self._cache[key] = (time.time(), value)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def invalidate(self, *args, **kwargs) -> None:
        key = self._key(*args, **kwargs)
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

# ============================================================================
# CORE SERVICE
# ============================================================================

class WebSearchService:
    def __init__(self):
        self.session_timeout = 30.0
        self.scrape_timeout = 20.0
        self.max_content_length = 25000
        self.deep_max_content_length = 50000
        self.rate_limit_delay = 0.8

        # API keys from environment (optional — each adds a search engine)
        self.brave_api_key = (os.environ.get("BRAVE_API_KEY") or "").strip()
        self.google_api_key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
        self.google_cse_id = (os.environ.get("GOOGLE_CSE_ID") or "").strip()
        self.bing_api_key = (os.environ.get("BING_API_KEY") or "").strip()
        self.serper_api_key = (os.environ.get("SERPER_API_KEY") or "").strip()
        self.tavily_api_key = (os.environ.get("TAVILY_API_KEY") or "").strip()

        self.jina_api_key = (
            os.environ.get("ELOQUENT_JINA_API_KEY") or os.environ.get("JINA_API_KEY") or ""
        ).strip()
        self.use_jina_fallback = os.environ.get("ELOQUENT_WEB_FETCH_JINA", "1").lower() not in (
            "0", "false", "no",
        )
        self.use_wayback_fallback = os.environ.get("ELOQUENT_WEB_FETCH_WAYBACK", "1").lower() not in (
            "0", "false", "no",
        )

        self.blocked_domains = {
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'pinterest.com', 'tiktok.com', 'snapchat.com', 'x.com',
        }

        self._llm_function: Optional[Callable] = None
        self._search_cache = SearchCache(ttl_seconds=120, max_entries=300)
        self._scrape_cache = SearchCache(ttl_seconds=600, max_entries=500)

        self.QUERY_OPTIMIZATION_PROMPT = """You are a world-class search query optimizer. Given a user's question or request, generate optimized search queries that will find the most relevant, high-quality information.

RULES:
1. Convert natural language into keyword-focused search queries (remove filler words)
2. For complex questions, generate multiple targeted queries covering different aspects
3. Include context terms: year, location, domain-specific terminology
4. Use site: operators for high-quality sources when appropriate (e.g., site:reuters.com, site:.gov)
5. Use quotation marks for exact phrases, proper names, and technical terms
6. Include alternative phrasings and synonyms
7. For news/timely topics, always include the current year (2026)
8. Prefer terms that appear in authoritative sources

USER INPUT: {user_prompt}

Respond in this exact JSON format only, no other text:
{{"queries": ["query1", "query2", "query3"], "intent": "detailed description of what information is being sought and why"}}"""

        logger.info(
            "WebSearchService: DDG + %s %s %s %s %s",
            "Brave" if self.brave_api_key else "",
            "Google" if self.google_api_key else "",
            "Bing" if self.bing_api_key else "",
            "Serper" if self.serper_api_key else "",
            "Tavily" if self.tavily_api_key else "",
        )

    # ========================================================================
    # LLM SETUP
    # ========================================================================

    def set_llm_function(self, llm_func: Callable):
        self._llm_function = llm_func
        logger.info("LLM function set for query optimization")

    # ========================================================================
    # QUERY OPTIMIZATION (LLM-required)
    # ========================================================================

    async def optimize_query(self, user_prompt: str, max_queries: int = 4) -> tuple[List[str], str]:
        if not self._llm_function:
            logger.warning("No LLM function set — returning raw prompt as query")
            return [user_prompt], "general search"

        try:
            prompt = self.QUERY_OPTIMIZATION_PROMPT.format(user_prompt=user_prompt)
            if asyncio.iscoroutinefunction(self._llm_function):
                response = await self._llm_function(prompt)
            else:
                response = self._llm_function(prompt)

            if not response or not isinstance(response, str):
                logger.warning("LLM returned empty response")
                return [user_prompt], "general search"

            json_match = re.search(r'\{[^{}]*"queries"[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                queries = parsed.get("queries", [user_prompt])
                intent = parsed.get("intent", "general search")
                if not queries or not isinstance(queries, list):
                    queries = [user_prompt]
                if max_queries and max_queries > 0:
                    queries = queries[:max_queries]
                logger.info(f"Optimized: '{user_prompt[:60]}...' -> {queries}")
                return queries, intent
            else:
                logger.warning("Could not parse LLM JSON response")
                return [user_prompt], "general search"

        except Exception as e:
            logger.error(f"Query optimization error: {e}")
            return [user_prompt], "general search"

    # ========================================================================
    # MODEL-DRIVEN SEARCH DECISION — The chat model itself decides what to search
    # ========================================================================

    SEARCH_DECISION_PROMPT = """You are an AI assistant. Decide if you need to search the web for current information to respond to the user.

USER MESSAGE: {user_prompt}

Ask yourself:
- Does this require information that might have changed recently (news, prices, events, elections, etc.)?
- Am I unsure about specific facts I should verify?
- Does the user explicitly ask me to search, look up, or find something?
- Is this about a topic where I might have outdated knowledge?

If you DO need to search: respond with EXACTLY (no other text):
{{"search": true, "query": "your optimized search query", "intent": "brief reason"}}

If you DON'T need to search: respond with EXACTLY (no other text):
{{"search": false, "intent": "brief reason"}}"""

    async def decide_search_query(self, user_prompt: str) -> Tuple[bool, str, str]:
        """Ask the model itself if it needs web search. Returns (needs_search, query, intent)."""
        if not self._llm_function:
            return False, "", "no LLM function available"

        try:
            prompt = self.SEARCH_DECISION_PROMPT.format(user_prompt=user_prompt[:2000])
            if asyncio.iscoroutinefunction(self._llm_function):
                response = await self._llm_function(prompt)
            else:
                response = self._llm_function(prompt)

            if not response or not isinstance(response, str):
                return False, "", "empty model response"

            response_clean = response.strip()

            # Try to extract JSON block — handle markdown fences and loose JSON
            json_str = None
            for pattern in [
                r'```(?:json)?\s*(\{.*?"search".*?\})\s*```',
                r'(\{[\s\n]*"search"[\s\n]*:[\s\n]*(?:true|false).*?\})',
            ]:
                m = re.search(pattern, response_clean, re.DOTALL)
                if m:
                    json_str = m.group(1)
                    break

            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if parsed.get("search") is True:
                        query = (parsed.get("query") or user_prompt).strip()
                        intent = parsed.get("intent", "model decided to search")
                        logger.info(f"Model decided to search: '{query[:80]}...'")
                        return True, query, intent
                    else:
                        logger.info(f"Model decided NOT to search: {parsed.get('intent', 'no reason')[:100]}")
                        return False, "", parsed.get("intent", "model said no")
                except json.JSONDecodeError:
                    pass

            logger.warning("Model response had no parseable JSON: %.150s", response_clean)
            return False, "", "unparseable response"

            if not lower or len(lower) < 10:
                logger.info("Empty/short model response — defaulting to no search")
                return False, "", "empty response"

            logger.warning("Could not parse model search decision from: %.200s", response_clean)
            return False, "", "unparseable response"

        except Exception as e:
            logger.error(f"Search decision error: {e}")
            return False, "", f"error: {e}"

    # ========================================================================
    # PAGE FETCHING — Multi-strategy with anti-bot resilience
    # ========================================================================

    def _default_headers(self, url: Optional[str] = None, ua_index: int = 0) -> Dict[str, str]:
        headers = {
            "User-Agent": USER_AGENTS[ua_index % len(USER_AGENTS)],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }
        if url:
            try:
                parsed = urlparse(url)
                headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
            except Exception:
                pass
        return headers

    @staticmethod
    def _is_error_page_content(text: str, status_code: int) -> bool:
        if status_code in (401, 403, 429, 503):
            return True
        if not text:
            return True
        low = text[:3000].lower()
        if any(m in low for m in _ERROR_PAGE_MARKERS):
            if len(text.strip()) < 15000:
                return True
        return False

    async def _fetch_via_jina(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        if not self.use_jina_fallback:
            return None
        reader_url = f"https://r.jina.ai/{url}"
        headers: Dict[str, str] = {
            "Accept": "text/plain",
            "X-Return-Format": "text",
            "X-With-Generated-Alt": "true",
            "X-With-Images-Summary": "true",
        }
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"
        try:
            response = await client.get(reader_url, headers=headers, timeout=30.0)
            if response.status_code not in (200, 201):
                logger.debug("Jina reader HTTP %s for %s", response.status_code, url)
                return None
            text = (response.text or "").strip()
            if not text or self._is_error_page_content(text, 200):
                return None
            if len(text) < 80:
                return None
            logger.info("Jina reader fetched %s (%d chars)", url, len(text))
            return text[:self.max_content_length]
        except Exception as e:
            logger.debug("Jina reader failed for %s: %s", url, e)
            return None

    async def _fetch_via_wayback(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        if not self.use_wayback_fallback:
            return None
        try:
            avail = await client.get(
                "https://archive.org/wayback/available",
                params={"url": url},
                timeout=15.0,
            )
            if avail.status_code != 200:
                return None
            data = avail.json()
            closest = (data.get("archived_snapshots") or {}).get("closest") or {}
            if not closest.get("available"):
                return None
            snap_url = closest.get("url")
            if not snap_url:
                return None
            response = await client.get(
                snap_url,
                headers=self._default_headers(url),
                follow_redirects=True,
                timeout=self.scrape_timeout,
            )
            if response.status_code in (401, 403, 429):
                return None
            if self._is_error_page_content(response.text, response.status_code):
                return None
            content = self._extract_readable_content(response.text)
            if content and len(content.strip()) > 80:
                logger.info("Wayback snapshot fetched %s (%d chars)", url, len(content))
                return content[:self.max_content_length]
        except Exception as e:
            logger.debug("Wayback failed for %s: %s", url, e)
        return None

    async def _fetch_via_textise(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        """Textise dot iitty — removes JS/CSS and returns clean text."""
        try:
            textise_url = f"https://r.jina.ai/{url}" if not self.jina_api_key else None
            if textise_url:
                response = await client.get(
                    textise_url,
                    headers={"User-Agent": USER_AGENTS[0]},
                    timeout=20.0,
                )
                if response.status_code == 200:
                    text = (response.text or "").strip()
                    if len(text) > 200:
                        return text[:self.max_content_length]
            return None
        except Exception:
            return None

    async def _fetch_page_text(
        self, client: httpx.AsyncClient, url: str, ua_index: int = 0
    ) -> Tuple[bool, str, str]:
        cache_key = f"scrape:{url}"
        cached = self._scrape_cache.get(cache_key)
        if cached:
            return cached

        try:
            await asyncio.sleep(self.rate_limit_delay * 0.5)
            response = await client.get(
                url,
                headers=self._default_headers(url, ua_index),
                follow_redirects=True,
                timeout=self.scrape_timeout,
            )
            status = response.status_code
            if status not in (401, 403, 429):
                if status < 500:
                    if not self._is_error_page_content(response.text, status):
                        content = self._extract_readable_content(response.text)
                        if content and len(content.strip()) > 80:
                            result = (True, content[:self.max_content_length], "direct")
                            self._scrape_cache.set(result, cache_key)
                            return result
        except httpx.HTTPStatusError as e:
            code = e.response.status_code if e.response is not None else 0
            if code not in (401, 403, 429):
                logger.debug("Direct fetch %s: HTTP %s", url, code)
        except Exception as e:
            logger.debug("Direct fetch %s: %s", url, e)

        jina_text = await self._fetch_via_jina(client, url)
        if jina_text:
            result = (True, jina_text, "jina")
            self._scrape_cache.set(result, cache_key)
            return result

        wayback_text = await self._fetch_via_wayback(client, url)
        if wayback_text:
            result = (True, wayback_text, "wayback")
            self._scrape_cache.set(result, cache_key)
            return result

        result = (False, "", "none")
        self._scrape_cache.set(result, cache_key)
        return result

    HUMAN_FETCH_HINT = (
        "HUMAN WORKAROUND (no captcha solver): open the URL in your browser, "
        "copy the article text, paste it into chat, or upload it under Documents. "
        "Optional: set JINA_API_KEY in env for higher Jina Reader limits."
    )

    # ========================================================================
    # CONTENT EXTRACTION — Readability-style + structured data
    # ========================================================================

    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                parsed = json.loads(script.string) if script.string else {}
                if isinstance(parsed, list):
                    for item in parsed:
                        data.update(item)
                else:
                    data.update(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        og = {}
        for meta in soup.find_all("meta"):
            prop = meta.get("property", "") or meta.get("name", "")
            content = meta.get("content", "")
            if prop.startswith("og:") and content:
                og[prop] = content
        data["og"] = og

        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            data["canonical"] = canonical["href"]

        author = soup.find("meta", attrs={"name": "author"})
        if author and author.get("content"):
            data["author"] = author["content"]

        return data

    def _extract_readable_content(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for elem in soup(["script", "style", "nav", "footer", "header", "aside",
                          "noscript", "iframe", "svg", "form", "button",
                          ".sidebar", ".advertisement", ".ad", ".social-share",
                          ".comments", ".comment", ".related-posts", ".recommended"]):
            elem.decompose()

        structured = self._extract_structured_data(soup)

        main_content = None
        for selector in [
            'article', 'main', '[role="main"]',
            '.post-content', '.entry-content', '.article-content',
            '.content', '.main-content', '.post-body',
            '#content', '#main-content', '#article',
            '[itemprop="articleBody"]',
        ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body')

        if not main_content:
            return ""

        text_parts = []
        headings = []
        for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = elem.get_text(strip=True)
            if text and len(text) > 5:
                headings.append(text)

        for elem in main_content.find_all(['p', 'li', 'blockquote', 'td', 'th']):
            text = elem.get_text(strip=True)
            if text and len(text) > 15:
                text_parts.append(text)

        content = '\n\n'.join(text_parts)

        if structured:
            enrichment = []
            og = structured.get("og", {})
            if og.get("og:title"):
                enrichment.append(f"Title: {og['og:title']}")
            if og.get("og:description"):
                enrichment.append(f"Description: {og['og:description']}")
            if og.get("og:site_name"):
                enrichment.append(f"Site: {og['og:site_name']}")
            if structured.get("author"):
                enrichment.append(f"Author: {structured['author']}")
            if structured.get("datePublished"):
                enrichment.append(f"Published: {structured['datePublished']}")
            if enrichment:
                content = '\n'.join(enrichment) + '\n\n' + content

        return content

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        return self._extract_readable_content(str(soup))

    # ========================================================================
    # URL UTILITIES
    # ========================================================================

    def _decode_duckduckgo_redirect(self, url: str) -> str:
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

    def _is_scrapeable_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            if domain in self.blocked_domains:
                return False
            if parsed.scheme not in ['http', 'https']:
                return False
            path = parsed.path.lower()
            blocked_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx',
                                  '.ppt', '.pptx', '.zip', '.rar', '.gz', '.tar']
            if any(path.endswith(ext) for ext in blocked_extensions):
                return False
            return True
        except Exception:
            return False

    def _compute_credibility(self, url: str, publisher: Optional[str] = None) -> float:
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            if domain in CREDIBLE_DOMAINS:
                return 0.95
            if domain in LOW_CREDIBILITY_DOMAINS:
                return 0.35
            if publisher:
                pub_lower = publisher.lower()
                for cd in CREDIBLE_DOMAINS:
                    if cd in pub_lower:
                        return 0.9
            if any(ext in domain for ext in ['.edu', '.gov', '.mil']):
                return 0.85
            if '.org' in domain:
                return 0.6
            return 0.5
        except Exception:
            return 0.5

    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        if not results:
            return results
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for r in results:
            score = 1.0
            title_lower = r.title.lower()
            snippet_lower = r.snippet.lower()
            content_lower = (r.content or "").lower()

            title_match = sum(1 for t in query_terms if t in title_lower)
            snippet_match = sum(1 for t in query_terms if t in snippet_lower)
            content_match = sum(1 for t in query_terms if t in content_lower) if content_lower else 0

            score += title_match * 0.3
            score += snippet_match * 0.15
            score += content_match * 0.05

            if r.published_date:
                score += 0.1
            if r.scraped_successfully and r.content:
                score += 0.2
                content_len = len(r.content)
                if 500 < content_len < 50000:
                    score += 0.1

            credibility = self._compute_credibility(r.url, r.publisher)
            score += credibility * 0.2
            r.credibility_score = credibility

            r.relevance_score = score

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    # ========================================================================
    # SEARCH ENGINES
    # ========================================================================

    # --- DuckDuckGo ---

    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            logger.info(f"Searching DuckDuckGo for: '{query}'")
            headers = self._default_headers()
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                results = await self._search_duckduckgo_html(client, query, max_results, headers)
                if len(results) < max_results:
                    params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
                    response = await client.get("https://api.duckduckgo.com/", params=params, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        for topic in data.get("RelatedTopics", [])[: max_results - len(results)]:
                            if isinstance(topic, dict) and topic.get("FirstURL"):
                                url = topic["FirstURL"]
                                if self._is_scrapeable_url(url) and url not in {r.url for r in results}:
                                    text = topic.get("Text", "")
                                    results.append(SearchResult(
                                        title=text.split(" - ")[0] if " - " in text else text,
                                        url=url, snippet=text, engine="duckduckgo"
                                    ))
                logger.info(f"DuckDuckGo found {len(results)} results")
                for r in results:
                    r.engine = "duckduckgo"
                return results[:max_results]
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def _search_duckduckgo_html(
        self, client: httpx.AsyncClient, query: str, max_results: int,
        headers: Optional[Dict[str, str]] = None
    ) -> List[SearchResult]:
        try:
            await asyncio.sleep(self.rate_limit_delay)
            req_headers = headers or {"User-Agent": USER_AGENTS[0]}
            params = {"q": query}
            search_urls = [
                "https://html.duckduckgo.com/html/",
                "https://duckduckgo.com/html/"
            ]
            for search_url in search_urls:
                response = await client.get(search_url, params=params, headers=req_headers)
                if response.status_code not in (200, 202):
                    continue
                results = self._parse_duckduckgo_html_results(response.text, max_results)
                if results:
                    return results
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
            results.append(SearchResult(title=title or "No title", url=real_url, snippet=snippet, engine="duckduckgo"))
            if len(results) >= max_results:
                return results
        return results

    async def _search_duckduckgo_lite(
        self, client: httpx.AsyncClient, query: str, max_results: int,
        headers: Optional[Dict[str, str]] = None
    ) -> List[SearchResult]:
        try:
            await asyncio.sleep(self.rate_limit_delay)
            params = {"q": query}
            response = await client.get(
                "https://lite.duckduckgo.com/lite/",
                params=params,
                headers=headers or {"User-Agent": USER_AGENTS[0]},
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
                results.append(SearchResult(title=title or "No title", url=real_url, snippet=snippet, engine="duckduckgo"))
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
                results.append(SearchResult(title=title or "No title", url=real_url, snippet="", engine="duckduckgo"))
                if len(results) >= max_results:
                    return results
            return results
        except Exception:
            return []

    # --- Brave Search ---

    async def search_brave(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.brave_api_key:
            return []
        try:
            logger.info(f"Searching Brave for: '{query}'")
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": min(max_results, 20), "safesearch": "off"},
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self.brave_api_key,
                    },
                )
                if response.status_code != 200:
                    logger.warning(f"Brave API returned {response.status_code}")
                    return []
                data = response.json()
                results = []
                for item in (data.get("web", {}) or {}).get("results", [])[:max_results]:
                    url = (item.get("url") or "").strip()
                    if not url or not self._is_scrapeable_url(url):
                        continue
                    results.append(SearchResult(
                        title=(item.get("title") or "").strip(),
                        url=url,
                        snippet=(item.get("description") or "").strip(),
                        engine="brave",
                        publisher=item.get("source") or item.get("domain"),
                        published_date=item.get("age") or item.get("published_date"),
                    ))
                logger.info(f"Brave found {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []

    # --- Serper (Google via Serper.dev) ---

    async def search_serper(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.serper_api_key:
            return []
        try:
            logger.info(f"Searching Serper for: '{query}'")
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": min(max_results, 20)},
                    headers={
                        "X-API-KEY": self.serper_api_key,
                        "Content-Type": "application/json",
                    },
                )
                if response.status_code != 200:
                    logger.warning(f"Serper API returned {response.status_code}")
                    return []
                data = response.json()
                results = []
                for item in (data.get("organic") or [])[:max_results]:
                    url = (item.get("link") or "").strip()
                    if not url or not self._is_scrapeable_url(url):
                        continue
                    results.append(SearchResult(
                        title=(item.get("title") or "").strip(),
                        url=url,
                        snippet=(item.get("snippet") or "").strip(),
                        engine="serper",
                        publisher=item.get("source"),
                        published_date=item.get("date"),
                    ))
                logger.info(f"Serper found {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []

    # --- Tavily ---

    async def search_tavily(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.tavily_api_key:
            return []
        try:
            logger.info(f"Searching Tavily for: '{query}'")
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "max_results": min(max_results, 20),
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
                if response.status_code != 200:
                    logger.warning(f"Tavily API returned {response.status_code}")
                    return []
                data = response.json()
                results = []
                for item in (data.get("results") or [])[:max_results]:
                    url = (item.get("url") or "").strip()
                    if not url or not self._is_scrapeable_url(url):
                        continue
                    results.append(SearchResult(
                        title=(item.get("title") or "").strip(),
                        url=url,
                        snippet=(item.get("content") or "").strip(),
                        engine="tavily",
                        publisher=item.get("source"),
                        published_date=item.get("published_date"),
                    ))
                logger.info(f"Tavily found {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    # --- Bing ---

    async def search_bing(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.bing_api_key:
            return []
        try:
            logger.info(f"Searching Bing for: '{query}'")
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    params={"q": query, "count": min(max_results, 20), "mkt": "en-US"},
                    headers={
                        "Ocp-Apim-Subscription-Key": self.bing_api_key,
                    },
                )
                if response.status_code != 200:
                    logger.warning(f"Bing API returned {response.status_code}")
                    return []
                data = response.json()
                results = []
                for item in (data.get("webPages") or {}).get("value", [])[:max_results]:
                    url = (item.get("url") or "").strip()
                    if not url or not self._is_scrapeable_url(url):
                        continue
                    results.append(SearchResult(
                        title=(item.get("name") or "").strip(),
                        url=url,
                        snippet=(item.get("snippet") or "").strip(),
                        engine="bing",
                        publisher=item.get("provider", [{}])[0].get("name") if item.get("provider") else None,
                        published_date=item.get("datePublished"),
                    ))
                logger.info(f"Bing found {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return []

    # --- Google Custom Search ---

    async def search_google(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.google_api_key or not self.google_cse_id:
            return []
        try:
            logger.info(f"Searching Google for: '{query}'")
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": self.google_api_key,
                        "cx": self.google_cse_id,
                        "q": query,
                        "num": min(max_results, 10),
                    },
                )
                if response.status_code != 200:
                    logger.warning(f"Google API returned {response.status_code}")
                    return []
                data = response.json()
                results = []
                for item in (data.get("items") or [])[:max_results]:
                    url = (item.get("link") or "").strip()
                    if not url or not self._is_scrapeable_url(url):
                        continue
                    results.append(SearchResult(
                        title=(item.get("title") or "").strip(),
                        url=url,
                        snippet=(item.get("snippet") or "").strip(),
                        engine="google",
                        publisher=item.get("displayLink"),
                    ))
                logger.info(f"Google found {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []

    # ========================================================================
    # NEWS SEARCH
    # ========================================================================

    async def search_news(self, query: str, max_results: int = 10) -> List[SearchResult]:
        engines = [
            self._search_google_news_rss,
            self._search_bing_news_rss,
        ]
        for engine in engines:
            results = await engine(query, max_results)
            if results:
                for r in results:
                    r.engine = "news_rss"
                return results

        results = await self.search_duckduckgo(query, max_results)
        if results:
            for r in results:
                r.engine = "duckduckgo_news"
            return results

        broadened = self._broaden_news_query(query)
        if broadened and broadened != query:
            for engine in engines:
                results = await engine(broadened, max_results)
                if results:
                    for r in results:
                        r.engine = "news_rss"
                    return results

        return await self.search(query, max_results)

    def _broaden_news_query(self, query: str) -> str:
        q = query or ""
        q = re.sub(r"\b(news|headlines?)\b", "", q, flags=re.IGNORECASE)
        q = re.sub(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            "", q, flags=re.IGNORECASE
        )
        q = re.sub(r"\b(19|20)\d{2}\b", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q or query

    async def _search_google_news_rss(self, query: str, max_results: int = 10) -> List[SearchResult]:
        url = "https://news.google.com/rss/search"
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        return await self._fetch_rss_results(url, params, max_results)

    async def _search_bing_news_rss(self, query: str, max_results: int = 10) -> List[SearchResult]:
        url = "https://www.bing.com/news/search"
        params = {"q": query, "format": "rss", "mkt": "en-US"}
        return await self._fetch_rss_results(url, params, max_results)

    async def _fetch_rss_results(self, url: str, params: Dict[str, Any], max_results: int) -> List[SearchResult]:
        try:
            headers = {
                "User-Agent": USER_AGENTS[0],
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
                    pub_date = None
                    pub_date_node = item.find("pubDate")
                    if pub_date_node and pub_date_node.get_text(strip=True):
                        pub_date = pub_date_node.get_text(strip=True)
                    results.append(SearchResult(
                        title=title or "No title", url=link, snippet=snippet,
                        publisher=publisher, published_date=pub_date, engine="news_rss"
                    ))
                    if len(results) >= max_results:
                        break
                return results
        except Exception:
            return []

    # ========================================================================
    # AGGREGATED SEARCH (multi-engine fallback chain)
    # ========================================================================

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        engines: List[Tuple[str, Callable[[str, int], Awaitable[List[SearchResult]]]]] = []

        if self.serper_api_key:
            engines.append(("serper", self.search_serper))
        if self.tavily_api_key:
            engines.append(("tavily", self.search_tavily))
        if self.brave_api_key:
            engines.append(("brave", self.search_brave))
        if self.bing_api_key:
            engines.append(("bing", self.search_bing))
        if self.google_api_key and self.google_cse_id:
            engines.append(("google", self.search_google))
        engines.append(("duckduckgo", self.search_duckduckgo))

        all_results: List[SearchResult] = []
        seen_urls: set[str] = set()
        engines_used: List[str] = []

        for engine_name, engine_fn in engines:
            try:
                results = await engine_fn(query, max_results)
                if results:
                    engines_used.append(engine_name)
                    for r in results:
                        if r.url not in seen_urls:
                            seen_urls.add(r.url)
                            all_results.append(r)
                    if len(all_results) >= max_results * 3:
                        break
            except Exception as e:
                logger.debug(f"Engine {engine_name} failed: {e}")
                continue

        if not all_results:
            return []

        all_results = self._rank_results(all_results, query)
        for r in all_results:
            if not r.engine:
                r.engine = engines_used[0] if engines_used else "duckduckgo"

        final = all_results[:max_results]
        return final

    # ========================================================================
    # CONTENT SCRAPING
    # ========================================================================

    async def scrape_content(
        self,
        results: List[SearchResult],
        *,
        max_to_scrape: Optional[int] = None,
        deep: bool = False,
    ) -> List[SearchResult]:
        if not results:
            return results
        cap = max_to_scrape if max_to_scrape is not None else len(results)
        to_scrape = results[:max(0, cap)]
        logger.info(f"Scraping up to {len(to_scrape)}/{len(results)} URLs")

        content_limit = self.deep_max_content_length if deep else self.max_content_length
        blocked: List[str] = []
        async with httpx.AsyncClient(timeout=self.scrape_timeout) as client:
            for i, result in enumerate(to_scrape):
                logger.debug(f"Scraping: {result.url}")
                ok, text, method = await self._fetch_page_text(client, result.url, ua_index=i)
                if ok and text:
                    result.content = text[:content_limit]
                    result.scraped_successfully = True
                    if method != "direct":
                        result.snippet = (result.snippet or "") + f" [via {method}]"
                    logger.debug("Fetched %d chars from %s (%s)", len(result.content), result.url, method)
                else:
                    blocked.append(result.url)
                    result.scraped_successfully = False
                    logger.info("Blocked %s — tried direct, Jina, Wayback", result.url)

        successful = sum(1 for r in to_scrape if r.scraped_successfully)
        logger.info(f"Successfully fetched {successful}/{len(to_scrape)} URLs")
        if blocked and successful < len(to_scrape):
            logger.info("%d URL(s) still blocked. %s", len(blocked), self.HUMAN_FETCH_HINT[:120])
        return results

    # ========================================================================
    # FORMATTING
    # ========================================================================

    def format_search_context(self, query: str, results: List[SearchResult]) -> str:
        if not results:
            return f"WEB SEARCH RESULTS for '{query}':\nNo relevant results found."
        context_parts = [f"WEB SEARCH RESULTS for '{query}':"]
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n[{i}] {result.title}")
            context_parts.append(f"URL: {result.url}")
            if result.engine:
                context_parts.append(f"Engine: {result.engine}")
            if result.scraped_successfully and result.content:
                content_preview = result.content[:1200] + "..." if len(result.content) > 1200 else result.content
                context_parts.append(f"Content: {content_preview}")
            elif result.snippet:
                context_parts.append(f"Snippet: {result.snippet}")
            context_parts.append("")
        return '\n'.join(context_parts)

    def format_smart_search_context(
        self,
        original_prompt: str,
        optimized_queries: List[str],
        intent: str,
        results: List[SearchResult],
        engines_used: Optional[List[str]] = None,
    ) -> str:
        if not results:
            return f"WEB SEARCH for '{original_prompt}':\nNo relevant results found."
        context_parts = [
            f"WEB SEARCH RESULTS",
            f"Original query: {original_prompt}",
            f"Search intent: {intent}",
            f"Optimized searches: {', '.join(optimized_queries)}",
        ]
        if engines_used:
            context_parts.append(f"Engines: {', '.join(engines_used)}")
        context_parts.append(f"---")
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n[{i}] {result.title}")
            context_parts.append(f"URL: {result.url}")
            if result.publisher:
                context_parts.append(f"Source: {result.publisher}")
            if result.scraped_successfully and result.content:
                content_preview = result.content[:1200] + "..." if len(result.content) > 1200 else result.content
                context_parts.append(f"Content: {content_preview}")
            elif result.snippet:
                context_parts.append(f"Snippet: {result.snippet}")
            context_parts.append("")
        return '\n'.join(context_parts)

    def format_tool_response(
        self,
        queries: List[str],
        intent: str,
        results: List[SearchResult],
        *,
        max_chars_per_result: int = 2000,
    ) -> str:
        if not results:
            return f"No results found for: {', '.join(queries)}"
        limit = max(400, min(max_chars_per_result, self.max_content_length))
        context_parts = [
            f"Search completed for: {', '.join(queries)}",
            f"Intent: {intent}",
            f"Found {len(results)} relevant results:",
            "",
        ]
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result.title}")
            context_parts.append(f"URL: {result.url}")
            if getattr(result, "publisher", None):
                context_parts.append(f"Publisher: {result.publisher}")
            if result.scraped_successfully and result.content:
                text = result.content
                if len(text) > limit:
                    text = text[:limit] + f"\n… [truncated at {limit} chars]"
                context_parts.append(text)
            elif result.snippet:
                context_parts.append(result.snippet)
            context_parts.append("")
        return "\n".join(context_parts)

    # ========================================================================
    # URL FETCHER
    # ========================================================================

    async def fetch_urls(
        self,
        urls: List[str],
        *,
        max_urls: int = 20,
        max_chars_per_url: int = 10000,
    ) -> List[SearchResult]:
        seen: set[str] = set()
        unique: List[str] = []
        for raw in urls:
            u = (raw or "").strip()
            if not u or u in seen:
                continue
            if not u.startswith("http"):
                u = "https://" + u.lstrip("/")
            if self._is_scrapeable_url(u):
                seen.add(u)
                unique.append(u)
            if len(unique) >= max_urls:
                break
        if not unique:
            return []
        stubs = [SearchResult(title=u, url=u, snippet=f"Fetching {u}") for u in unique]
        scraped = await self.scrape_content(stubs)
        for r in scraped:
            if r.content and len(r.content) > max_chars_per_url:
                r.content = r.content[:max_chars_per_url]
            if r.scraped_successfully and r.content and not r.title:
                r.title = r.url
        return scraped


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

web_search_service = WebSearchService()


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

async def perform_web_search(query: str, max_results: int = 5, *, scrape_full: bool = False) -> str:
    try:
        logger.info(f"Starting web search for: '{query}'")
        results = await web_search_service.search(query, max_results)
        if not results:
            return f"WEB SEARCH RESULTS for '{query}':\nNo results found or search service unavailable."
        if scrape_full:
            results_with_content = await web_search_service.scrape_content(
                results, max_to_scrape=min(4, max_results)
            )
        else:
            results_with_content = results
        formatted_context = web_search_service.format_search_context(query, results_with_content)
        return formatted_context
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"WEB SEARCH RESULTS for '{query}':\nSearch failed due to technical error: {str(e)}"


async def perform_smart_web_search(
    user_prompt: str,
    max_results: int = 5,
    use_optimization: bool = True,
    max_queries: int = 4,
    *,
    scrape_full: bool = False,
) -> SmartSearchResult:
    try:
        logger.info(f"Starting smart web search for: '{user_prompt[:100]}...'")

        if use_optimization:
            optimized_queries, search_intent = await web_search_service.optimize_query(
                user_prompt, max_queries=max_queries
            )
        else:
            optimized_queries = [user_prompt]
            search_intent = "direct search"

        logger.info(f"Optimized queries: {optimized_queries}")

        all_results: List[SearchResult] = []
        seen_urls: set[str] = set()
        engines_used: set[str] = set()

        for query in optimized_queries:
            results = await web_search_service.search(query, max_results)
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
                    if result.engine:
                        engines_used.add(result.engine)

        if not all_results:
            return SmartSearchResult(
                original_prompt=user_prompt,
                optimized_queries=optimized_queries,
                search_intent=search_intent,
                results=[],
                formatted_context=f"WEB SEARCH for '{user_prompt}':\nNo results found.",
                engines_used=list(engines_used),
            )

        results_to_scrape = all_results[:max_results]
        if scrape_full:
            results_with_content = await web_search_service.scrape_content(
                results_to_scrape, max_to_scrape=min(6, max_results)
            )
        else:
            results_with_content = results_to_scrape

        formatted_context = web_search_service.format_smart_search_context(
            user_prompt, optimized_queries, search_intent, results_with_content,
            engines_used=list(engines_used),
        )

        logger.info(
            "Smart search: %d results from %d queries using %s",
            len(results_with_content), len(optimized_queries),
            ", ".join(sorted(engines_used)) or "duckduckgo",
        )

        return SmartSearchResult(
            original_prompt=user_prompt,
            optimized_queries=optimized_queries,
            search_intent=search_intent,
            results=results_with_content,
            formatted_context=formatted_context,
            engines_used=list(engines_used),
        )

    except Exception as e:
        logger.error(f"Smart web search error: {e}")
        return SmartSearchResult(
            original_prompt=user_prompt,
            optimized_queries=[user_prompt],
            search_intent="error",
            results=[],
            formatted_context=f"WEB SEARCH for '{user_prompt}':\nSearch failed: {str(e)}",
        )


async def handle_fetch_urls_tool_call(
    arguments: Dict[str, Any],
    *,
    max_urls: int = 20,
    max_chars_per_url: int = 10000,
) -> str:
    try:
        if isinstance(arguments, str):
            arguments = {"urls": [arguments]}
        urls = arguments.get("urls") or arguments.get("url") or []
        if isinstance(urls, str):
            urls = [u.strip() for u in re.split(r"[\n,]+", urls) if u.strip()]
        if not urls:
            return "Error: provide urls as a list of article links."
        results = await web_search_service.fetch_urls(
            urls, max_urls=max_urls, max_chars_per_url=max_chars_per_url,
        )
        if not results:
            return "Could not fetch any of the provided URLs."
        ok = [r for r in results if r.scraped_successfully and r.content]
        parts = [f"Fetched {len(ok)}/{len(urls)} URLs:", ""]
        for i, r in enumerate(ok, 1):
            parts.append(f"=== Article {i}: {r.title or r.url} ===")
            parts.append(f"URL: {r.url}")
            parts.append(r.content or "")
            parts.append("")
        if len(ok) < len(urls):
            failed = [r.url for r in results if not r.scraped_successfully]
            parts.append(f"Failed to fetch: {', '.join(failed[:10])}")
            parts.append(web_search_service.HUMAN_FETCH_HINT)
        return "\n".join(parts)
    except Exception as e:
        logger.error("fetch_urls tool error: %s", e)
        return f"fetch_urls failed: {e}"


async def handle_web_search_tool_call(
    arguments: Dict[str, Any],
    max_results: int = 5,
    news: bool = False,
    *,
    max_chars_per_result: int = 2000,
    scrape_full: bool = True,
) -> str:
    try:
        if isinstance(arguments, str):
            arguments = {"query": arguments}
        if isinstance(arguments, dict) and "arguments" in arguments and isinstance(arguments["arguments"], dict):
            arguments = arguments["arguments"]
        if isinstance(arguments, dict) and "parameters" in arguments and isinstance(arguments["parameters"], dict):
            arguments = arguments["parameters"]
        if isinstance(arguments, dict) and "query" in arguments and isinstance(arguments["query"], str):
            raw = arguments["query"]
            if "{" in raw and "query" in raw:
                queries = re.findall(r'"query"\s*:\s*"([^"]+)"', raw)
                if queries:
                    arguments = {"search_queries": queries}
        if "search_queries" in arguments and "query" not in arguments:
            sq = arguments.get("search_queries")
            if isinstance(sq, list) and sq:
                arguments = {**arguments, "query": sq[0]}
            elif isinstance(sq, str) and sq.strip():
                arguments = {**arguments, "query": sq.strip()}
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

        logger.info(f"Handling web_search tool call ({'news' if news else 'web'}): {queries}")

        all_results: List[SearchResult] = []
        seen_urls: set[str] = set()

        search_fn = web_search_service.search_news if news else web_search_service.search
        for query in queries[:3]:
            results = await search_fn(query, max_results)
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)

        if not all_results:
            return f"Web search found no results for: {queries}"

        if news:
            top_titles = [r.title for r in all_results[:5] if r.title]
            logger.info(f"News search returned {len(all_results)} results. Top: {top_titles}")
            return web_search_service.format_tool_response(
                queries, intent, all_results[:max_results],
                max_chars_per_result=max_chars_per_result,
            )

        results_to_scrape = all_results[:max_results]
        if scrape_full:
            results_with_content = await web_search_service.scrape_content(
                results_to_scrape, max_to_scrape=min(6, max_results)
            )
        else:
            results_with_content = results_to_scrape

        formatted = web_search_service.format_tool_response(
            queries, intent, results_with_content,
            max_chars_per_result=max_chars_per_result,
        )
        return formatted

    except Exception as e:
        logger.error(f"Web search tool call error: {e}")
        return f"Web search failed: {str(e)}"


async def handle_web_fetch_tool_call(
    arguments: Dict[str, Any],
    max_chars: int = 25000,
) -> str:
    try:
        if isinstance(arguments, str):
            arguments = {"url": arguments}
        if isinstance(arguments, dict) and "arguments" in arguments and isinstance(arguments["arguments"], dict):
            arguments = arguments["arguments"]
        if isinstance(arguments, dict) and "parameters" in arguments and isinstance(arguments["parameters"], dict):
            arguments = arguments["parameters"]
        
        url = arguments.get("url") or arguments.get("URL")
        if not url:
            return "Error: No URL provided in tool call arguments"
        
        max_chars = arguments.get("max_chars", max_chars)
        if not isinstance(max_chars, int) or max_chars < 1000:
            max_chars = 25000
        
        logger.info(f"Handling web_fetch tool call: {url}")
        
        result = await web_search_service.search(url, max_results=1)
        if not result:
            return f"Failed to fetch: {url}"
        
        scraped = await web_search_service.scrape_content(
            result[:1], max_to_scrape=1
        )
        
        if scraped and scraped[0].scraped_successfully:
            content = scraped[0].content or ""
            if len(content) > max_chars:
                content = content[:max_chars] + "... [truncated]"
            return json.dumps({
                "url": scraped[0].url,
                "title": scraped[0].title,
                "extracted_text": content,
                "scraped_successfully": True
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "url": url,
                "title": "",
                "extracted_text": "Failed to extract content from page",
                "scraped_successfully": False
            }, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Web fetch tool call error: {e}")
        return f"Web fetch failed: {str(e)}"


def get_web_search_tool_definition(simple: bool = False, unified: bool = False) -> Dict[str, Any]:
    if unified:
        return WEB_SEARCH_TOOL_UNIFIED
    return WEB_SEARCH_TOOL_SIMPLE if simple else WEB_SEARCH_TOOL_DEFINITION


def get_web_fetch_tool_definition() -> Dict[str, Any]:
    return WEB_FETCH_TOOL_DEFINITION


def set_web_search_llm(llm_function: Callable):
    web_search_service.set_llm_function(llm_function)
