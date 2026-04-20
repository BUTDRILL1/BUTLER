from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext, ToolError

# Cache and rate limit state
_web_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_domain_last_call: dict[str, float] = {}
_query_last_call: dict[str, float] = {}
CACHE_TTL_SECONDS = 600

class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=100)

def _web_search(ctx: ToolContext, args: WebSearchArgs) -> dict[str, Any]:
    query = args.query.strip()
    if not query:
        raise ToolError("Empty query")
        
    cache_key = query.lower()
    if cache_key in _web_cache:
        cached_results, timestamp = _web_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return {"query": query, "results": cached_results, "cached": True}

    now = time.time()
    last_query_call = _query_last_call.get(cache_key, 0)
    if now - last_query_call < 2.0:
        time.sleep(1.0)
    _query_last_call[cache_key] = time.time()

    try:
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=15))
    except Exception as e:
        raise ToolError(f"Search failed: {e}")

    results = []
    seen_urls = set()
    seen_domains = set()
    
    for r in raw_results:
        href = r.get("href")
        title = r.get("title")
        snippet = r.get("body", "")
        
        if not href or not title:
            continue
            
        url_len = len(href)
        if url_len > 500:
            continue
            
        domain = urlparse(href).netloc.lower()
        if any(domain.endswith(blocked) for blocked in ctx.config.blocked_web_domains):
            continue
            
        if href in seen_urls:
            continue
            
        if domain in seen_domains:
            continue
            
        seen_urls.add(href)
        seen_domains.add(domain)
        
        parsed = urlparse(href)
        clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        
        results.append({
            "rank": len(results) + 1,
            "title": title[:200],
            "snippet": snippet[:300].strip(),
            "url": clean_url
        })
        
        if len(results) >= 5:
            break

    _web_cache[cache_key] = (results, now)
    return {"query": query, "results": results, "cached": False}

class WebReadArgs(BaseModel):
    url: str

def _web_read(ctx: ToolContext, args: WebReadArgs) -> dict[str, Any]:
    url = args.url.strip()
    if not url.startswith(("http://", "https://")):
        raise ToolError("Invalid URL")
        
    domain = urlparse(url).netloc.lower()
    if any(domain.endswith(blocked) for blocked in ctx.config.blocked_web_domains):
        raise ToolError(f"Domain {domain} is blocked.")
        
    now = time.time()
    last_call = _domain_last_call.get(domain, 0)
    if now - last_call < 1.0:
        time.sleep(1.0 - (now - last_call))
    _domain_last_call[domain] = time.time()
    
    headers = {"User-Agent": "Mozilla/5.0"}
    res = None
    
    for _ in range(2):
        try:
            res = requests.get(url, headers=headers, timeout=(3, 10), allow_redirects=True, stream=True)
            break
        except requests.exceptions.RequestException:
            time.sleep(1)
            continue
            
    if res is None:
        raise ToolError("Timeout or connection failed")
        
    if res.status_code != 200:
        raise ToolError(f"Failed to fetch page: HTTP {res.status_code}")
        
    if len(res.history) > 5:
        raise ToolError("Too many redirects")
        
    content_type = res.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ToolError(f"Unsupported content type: {content_type}")
        
    raw_content = b""
    # Iterate safely up to 2MB to prevent memory bloat
    for chunk in res.iter_content(chunk_size=8192):
        raw_content += chunk
        if len(raw_content) > 2_000_000:
            res.close()
            raise ToolError("Page too large")
            
    # Mock behavior of requests.get bypassing raw content
    # In reality apparent_encoding needs chardet, let's just decode cleanly:
    # Actually requests handles encoding on res.content -> res.text via apparent_encoding
    res._content = raw_content
    res.encoding = res.apparent_encoding
    text = res.text
    
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img"]):
        tag.decompose()
        
    title = soup.title.string if soup.title else "No title"
    content = soup.get_text(separator=" ", strip=True)
    content = " ".join(content.split())
    
    if len(content) < 200:
        raise ToolError("Content too small or not useful")
        
    content = content[:3000].strip()
    return {"url": url, "title": title.strip(), "content": content}

class WebNewsArgs(BaseModel):
    query: str = Field(max_length=200)

def _web_news(ctx: ToolContext, args: WebNewsArgs) -> dict[str, Any]:
    with DDGS() as ddgs:
        results = list(ddgs.news(args.query, max_results=5))
    return {"results": results[:3]}

def build() -> list[Tool]:
    return [
        Tool(
            name="web.search",
            description="Search the web securely (DuckDuckGo). Returns top hits.",
            input_model=WebSearchArgs,
            handler=_web_search,
            side_effect=False,
        ),
        Tool(
            name="web.read",
            description="Read text content from a specific URL safely.",
            input_model=WebReadArgs,
            handler=_web_read,
            side_effect=False,
        ),
        Tool(
            name="web.news",
            description="Search the web specifically for latest news and headlines.",
            input_model=WebNewsArgs,
            handler=_web_news,
            side_effect=False,
        ),
    ]
