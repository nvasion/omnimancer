"""Web client for HTTP requests, API interactions, and web scraping with security features."""

import asyncio
import hashlib
import json
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse

import aiohttp
import html2text
from bs4 import BeautifulSoup
from readability import Document

from ..security import SecurityManager
from ..security.permission_controller import PermissionOperation


class RequestMethod(Enum):
    """HTTP request methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    PATCH = "PATCH"


@dataclass
class WebResponse:
    """Response from web request."""

    url: str
    status: int
    headers: Dict[str, str]
    content: Union[str, bytes]
    content_type: str
    encoding: str
    cookies: Dict[str, str] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    from_cache: bool = False

    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status < 300

    @property
    def is_text(self) -> bool:
        """Check if response is text content."""
        return isinstance(self.content, str)

    @property
    def text(self) -> str:
        """Get text content."""
        if isinstance(self.content, str):
            return self.content

        # Handle binary encoding specially
        if self.encoding == "binary":
            return self.content.decode("utf-8", errors="ignore")

        return self.content.decode(self.encoding or "utf-8", errors="ignore")

    @property
    def json(self) -> Any:
        """Parse JSON content."""
        return json.loads(self.text)


@dataclass
class CacheEntry:
    """Cache entry for responses."""

    response: WebResponse
    timestamp: datetime
    expires: datetime
    etag: Optional[str] = None
    last_modified: Optional[str] = None


class RateLimiter:
    """Rate limiter for HTTP requests."""

    def __init__(
        self,
        requests_per_second: float = 10.0,
        requests_per_minute: float = 100.0,
        requests_per_hour: float = 1000.0,
    ):
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

        self.request_times: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()

    async def wait_if_needed(self, domain: str) -> None:
        """Wait if rate limit would be exceeded."""
        async with self.lock:
            now = time.time()

            if domain not in self.request_times:
                self.request_times[domain] = []

            times = self.request_times[domain]

            # Clean old entries
            times[:] = [t for t in times if now - t < 3600]  # Keep last hour

            # Check limits
            recent_second = [t for t in times if now - t < 1.0]
            recent_minute = [t for t in times if now - t < 60.0]
            recent_hour = [t for t in times if now - t < 3600.0]

            wait_time = 0.0

            # Check per-second limit
            if len(recent_second) >= self.requests_per_second:
                wait_time = max(wait_time, 1.0 - (now - recent_second[0]))

            # Check per-minute limit
            if len(recent_minute) >= self.requests_per_minute:
                wait_time = max(wait_time, 60.0 - (now - recent_minute[0]))

            # Check per-hour limit
            if len(recent_hour) >= self.requests_per_hour:
                wait_time = max(wait_time, 3600.0 - (now - recent_hour[0]))

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Record this request
            times.append(time.time())


class ResponseCache:
    """Cache for HTTP responses."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_mb: int = 100,
        default_ttl: int = 3600,
    ):

        self.cache_dir = cache_dir or Path.home() / ".omnimancer" / "web_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.memory_cache: Dict[str, CacheEntry] = {}

    def _get_cache_key(self, url: str, method: str, headers: Dict[str, str]) -> str:
        """Generate cache key for request."""
        key_data = f"{method}:{url}:{json.dumps(sorted(headers.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"{cache_key}.cache"

    async def get(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[WebResponse]:
        """Get cached response if available and valid."""

        headers = headers or {}
        cache_key = self._get_cache_key(url, method, headers)

        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if datetime.now() < entry.expires:
                entry.response.from_cache = True
                return entry.response
            else:
                del self.memory_cache[cache_key]

        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    entry = pickle.load(f)

                if datetime.now() < entry.expires:
                    # Load into memory cache
                    self.memory_cache[cache_key] = entry
                    entry.response.from_cache = True
                    return entry.response
                else:
                    # Remove expired cache
                    cache_path.unlink()

            except (pickle.PickleError, EOFError, FileNotFoundError):
                # Remove corrupted cache
                if cache_path.exists():
                    cache_path.unlink()

        return None

    async def set(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        response: WebResponse,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache response."""

        ttl = ttl if ttl is not None else self.default_ttl
        cache_key = self._get_cache_key(url, method, headers)

        expires = datetime.now() + timedelta(seconds=ttl)
        entry = CacheEntry(
            response=response,
            timestamp=datetime.now(),
            expires=expires,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )

        # Store in memory
        self.memory_cache[cache_key] = entry

        # Store on disk
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(entry, f)
        except (pickle.PickleError, OSError):
            pass  # Continue without disk cache

        self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Clean up cache to stay within size limits."""
        # Remove expired entries from memory
        now = datetime.now()
        expired_keys = [k for k, v in self.memory_cache.items() if now >= v.expires]
        for key in expired_keys:
            del self.memory_cache[key]

        # Check disk cache size
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
            max_size_bytes = self.max_size_mb * 1024 * 1024

            if total_size > max_size_bytes:
                # Remove oldest files
                cache_files = sorted(
                    self.cache_dir.glob("*.cache"),
                    key=lambda f: f.stat().st_mtime,
                )

                for cache_file in cache_files:
                    cache_file.unlink()
                    total_size -= cache_file.stat().st_size
                    if total_size <= max_size_bytes * 0.8:  # Keep 20% buffer
                        break
        except OSError:
            pass


class WebClient:
    """Advanced web client with rate limiting, caching, and content extraction."""

    def __init__(
        self,
        security_manager: Optional[SecurityManager] = None,
        user_agent: str = "Omnimancer-Agent/1.0",
        timeout: int = 30,
        max_redirects: int = 10,
        enable_cache: bool = True,
        enable_rate_limiting: bool = True,
    ):

        self.security = security_manager or SecurityManager()
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_redirects = max_redirects

        # Initialize components
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.cache = ResponseCache() if enable_cache else None

        # Domain blacklist (security)
        self.blacklisted_domains = self._get_default_blacklist()
        self.whitelisted_domains: Set[str] = set()

        # Content extraction
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True

        # Session will be created when needed
        self._session: Optional[aiohttp.ClientSession] = None

        # Request statistics
        self.stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limited": 0,
            "blocked_requests": 0,
            "failed_requests": 0,
        }

    def _get_default_blacklist(self) -> Set[str]:
        """Get default domain blacklist."""
        return {
            # Localhost and private networks
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
            # Private IP ranges (partial)
            "192.168.",
            "10.",
            "172.16.",
            "172.17.",
            "172.18.",
            "172.19.",
            "172.20.",
            "172.21.",
            "172.22.",
            "172.23.",
            "172.24.",
            "172.25.",
            "172.26.",
            "172.27.",
            "172.28.",
            "172.29.",
            "172.30.",
            "172.31.",
            # Potentially harmful
            "malware-test.com",
            "eicar.org",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"User-Agent": self.user_agent}

            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(
                    limit=100, limit_per_host=20, enable_cleanup_closed=True
                ),
            )

        return self._session

    def _is_url_allowed(self, url: str) -> bool:
        """Check if URL is allowed by security policy."""
        try:
            parsed = urlparse(url)
            domain = parsed.hostname

            if not domain:
                return False

            # Check whitelist first
            if self.whitelisted_domains and domain in self.whitelisted_domains:
                return True

            # Check blacklist
            if domain in self.blacklisted_domains:
                return False

            # Check for private IP patterns
            for blocked_pattern in self.blacklisted_domains:
                if blocked_pattern.endswith(".") and domain.startswith(blocked_pattern):
                    return False

            # Additional security checks
            if parsed.scheme not in ["http", "https"]:
                return False

            return True

        except Exception:
            return False

    async def request(
        self,
        url: str,
        method: RequestMethod = RequestMethod.GET,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        cookies: Optional[Dict[str, str]] = None,
        allow_redirects: bool = True,
        timeout: Optional[int] = None,
        cache_ttl: Optional[int] = None,
    ) -> WebResponse:
        """Make HTTP request with security, rate limiting, and caching."""

        url = url.strip()
        headers = headers or {}
        method_str = (
            method.value if isinstance(method, RequestMethod) else str(method).upper()
        )

        # Security validation
        if not self._is_url_allowed(url):
            self.stats["blocked_requests"] += 1
            raise ValueError(f"URL blocked by security policy: {url}")

        # Validate with security manager
        operation = PermissionOperation(
            operation_type="network_request",
            url=url,
            method=method_str,
            headers=headers,
        )

        validation = await self.security.validate_operation(operation)
        if not validation["allowed"]:
            self.stats["blocked_requests"] += 1
            raise ValueError(f"Request blocked: {', '.join(validation['reasons'])}")

        # Check cache for GET requests
        cached_response = None
        if method == RequestMethod.GET and self.cache:
            cached_response = await self.cache.get(url, method_str, headers)
            if cached_response:
                self.stats["cache_hits"] += 1
                return cached_response
            self.stats["cache_misses"] += 1

        # Rate limiting
        if self.rate_limiter:
            domain = urlparse(url).hostname
            if domain:
                start_time = time.time()
                await self.rate_limiter.wait_if_needed(domain)
                if time.time() - start_time > 0.1:  # If we waited more than 100ms
                    self.stats["rate_limited"] += 1

        # Make request with retries
        response = await self._make_request_with_retry(
            url,
            method_str,
            headers,
            params,
            data,
            json_data,
            cookies,
            allow_redirects,
            timeout or self.timeout,
        )

        # Cache successful GET responses
        if (
            response.is_success
            and method == RequestMethod.GET
            and self.cache
            and response.status != 204
        ):
            await self.cache.set(url, method_str, headers, response, cache_ttl)

        self.stats["requests_made"] += 1
        return response

    async def _make_request_with_retry(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]],
        data: Optional[Union[str, bytes, Dict[str, Any]]],
        json_data: Optional[Dict[str, Any]],
        cookies: Optional[Dict[str, str]],
        allow_redirects: bool,
        timeout: int,
        max_retries: int = 3,
    ) -> WebResponse:
        """Make request with exponential backoff retry."""

        session = await self._get_session()
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                # Prepare request kwargs
                kwargs = {}
                if params:
                    kwargs["params"] = params
                if data:
                    kwargs["data"] = data
                if json_data:
                    kwargs["json"] = json_data
                if cookies:
                    kwargs["cookies"] = cookies
                if not allow_redirects:
                    kwargs["allow_redirects"] = False

                async with session.request(
                    method, url, headers=headers, **kwargs
                ) as resp:
                    elapsed = time.time() - start_time

                    # Read content
                    content_bytes = await resp.read()

                    # Determine if content is text
                    content_type = resp.headers.get("content-type", "").lower()
                    is_text = any(
                        t in content_type
                        for t in [
                            "text/",
                            "application/json",
                            "application/xml",
                        ]
                    )

                    # Convert to text if appropriate
                    if is_text:
                        encoding = resp.charset or "utf-8"
                        content = content_bytes.decode(encoding, errors="ignore")
                    else:
                        content = content_bytes
                        encoding = "binary"

                    # Build response
                    response = WebResponse(
                        url=str(resp.url),
                        status=resp.status,
                        headers=dict(resp.headers),
                        content=content,
                        content_type=content_type,
                        encoding=encoding,
                        cookies=dict(resp.cookies),
                        history=[str(h.url) for h in resp.history],
                        elapsed=elapsed,
                    )

                    return response

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e

                # Don't retry on client errors (4xx) except specific cases
                if hasattr(e, "status") and 400 <= e.status < 500:
                    if e.status not in [429, 408]:  # Rate limit, timeout
                        break

                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = (2**attempt) + (0.1 * attempt)
                    await asyncio.sleep(wait_time)

        # All retries failed
        self.stats["failed_requests"] += 1
        raise Exception(
            f"Request failed after {max_retries + 1} attempts: {last_exception}"
        )

    async def get(self, url: str, **kwargs) -> WebResponse:
        """Make GET request."""
        return await self.request(url, RequestMethod.GET, **kwargs)

    async def post(self, url: str, **kwargs) -> WebResponse:
        """Make POST request."""
        return await self.request(url, RequestMethod.POST, **kwargs)

    async def put(self, url: str, **kwargs) -> WebResponse:
        """Make PUT request."""
        return await self.request(url, RequestMethod.PUT, **kwargs)

    async def delete(self, url: str, **kwargs) -> WebResponse:
        """Make DELETE request."""
        return await self.request(url, RequestMethod.DELETE, **kwargs)

    async def head(self, url: str, **kwargs) -> WebResponse:
        """Make HEAD request."""
        return await self.request(url, RequestMethod.HEAD, **kwargs)

    async def scrape_content(
        self,
        url: str,
        extract_main_content: bool = True,
        convert_to_markdown: bool = True,
        remove_scripts: bool = True,
    ) -> Dict[str, Any]:
        """Scrape and extract content from web page."""

        response = await self.get(url)

        if not response.is_success:
            raise ValueError(f"Failed to fetch content: HTTP {response.status}")

        if not response.is_text:
            raise ValueError("Response is not text content")

        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted elements
        if remove_scripts:
            for element in soup(
                ["script", "style", "nav", "header", "footer", "aside"]
            ):
                element.decompose()

        # Extract metadata
        title = soup.find("title")
        title_text = title.get_text().strip() if title else ""

        meta_description = soup.find("meta", attrs={"name": "description"})
        description = (
            meta_description.get("content", "").strip() if meta_description else ""
        )

        # Extract main content
        if extract_main_content:
            # Try readability first (use cleaned HTML)
            try:
                cleaned_html = str(soup)
                doc = Document(cleaned_html)
                main_content = doc.summary()
                main_soup = BeautifulSoup(main_content, "html.parser")
            except:
                # Fallback to common content selectors
                main_soup = (
                    soup.find("main")
                    or soup.find("article")
                    or soup.find("div", class_=re.compile(r"content|main|article"))
                )
                if not main_soup:
                    main_soup = soup
        else:
            main_soup = soup

        # Convert to text
        if convert_to_markdown:
            # Convert to markdown
            main_text = self.html_converter.handle(str(main_soup))
        else:
            # Plain text
            main_text = main_soup.get_text(separator="\n", strip=True)

        # Extract links
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text().strip()
            if href and text:
                absolute_url = urljoin(url, href)
                links.append({"url": absolute_url, "text": text})

        # Extract images
        images = []
        for img in soup.find_all("img", src=True):
            src = img["src"]
            alt = img.get("alt", "").strip()
            absolute_url = urljoin(url, src)
            images.append({"url": absolute_url, "alt": alt})

        return {
            "url": response.url,
            "title": title_text,
            "description": description,
            "content": main_text,
            "links": links[:50],  # Limit links
            "images": images[:20],  # Limit images
            "status": response.status,
            "content_type": response.content_type,
            "word_count": len(main_text.split()),
            "char_count": len(main_text),
        }

    def add_to_blacklist(self, domain: str) -> None:
        """Add domain to blacklist."""
        self.blacklisted_domains.add(domain)

    def add_to_whitelist(self, domain: str) -> None:
        """Add domain to whitelist."""
        self.whitelisted_domains.add(domain)

    def remove_from_blacklist(self, domain: str) -> None:
        """Remove domain from blacklist."""
        self.blacklisted_domains.discard(domain)

    def remove_from_whitelist(self, domain: str) -> None:
        """Remove domain from whitelist."""
        self.whitelisted_domains.discard(domain)

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self.stats,
            "cache_size": len(self.cache.memory_cache) if self.cache else 0,
            "blacklisted_domains": len(self.blacklisted_domains),
            "whitelisted_domains": len(self.whitelisted_domains),
        }

    def clear_cache(self) -> None:
        """Clear response cache."""
        if self.cache:
            self.cache.memory_cache.clear()
            # Clear disk cache
            for cache_file in self.cache.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass

    async def close(self) -> None:
        """Close the web client and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

        # Clear cache on close
        if self.cache:
            self.cache.memory_cache.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
