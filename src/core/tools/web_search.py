"""Web Search & Scraping Module for O.L.I.V.I.A.
Ad filtering, multi-source comparison, and content extraction.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

try:
    from src.utils.logger import get_logger

    log = get_logger("web")
except ImportError:
    import logging

    log = logging.getLogger("web")

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False


class SearchConfig:
    """Central configuration for web search behavior."""

    # Number of results to fetch and scrape
    MAX_SEARCH_RESULTS = 10  # How many results to get from search
    MAX_SCRAPE_RESULTS = 5  # How many to actually scrape (top N after filtering)

    # Content limits
    MAX_CONTENT_LENGTH = 8000  # Max chars per page (increased from 4000)
    MAX_TOTAL_CONTEXT = 30000  # Max total chars to send to LLM

    # Timeouts
    SEARCH_TIMEOUT = 15  # Seconds for search request
    SCRAPE_TIMEOUT = 12  # Seconds per page scrape

    # Rate limiting
    REQUEST_DELAY = 0.5  # Seconds between requests (be nice to servers)

    # Ad/Spam filtering - pre-compiled patterns for performance
    _AD_URL_PATTERN_STRINGS = [
        r"ad_domain=",
        r"ad_provider=",
        r"ad_type=",
        r"/aclick\?",
        r"doubleclick\.net",
        r"googleadservices\.com",
        r"googlesyndication\.com",
        r"amazon-adsystem\.com",
        r"facebook\.com/tr",
        r"click\.linksynergy",
        r"shareasale\.com",
        r"commission-junction",
        r"affiliate",
        r"sponsored",
    ]

    _AD_TITLE_PATTERN_STRINGS = [
        r"^ad\s*[-–—:]\s*",
        r"\s*[-–—]\s*ad$",
        r"sponsored",
        r"advertisement",
        r"^shop\s",
        r"buy now",
        r"limited time offer",
        r"free shipping",
    ]

    # Pre-compiled regex patterns (compiled once at class definition time)
    AD_URL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _AD_URL_PATTERN_STRINGS]
    AD_TITLE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _AD_TITLE_PATTERN_STRINGS]

    # Optimization: Pre-compile combined regex for single-pass matching
    # Combined pattern matches any ad URL indicator in one pass instead of N pattern loops
    AD_URL_COMBINED = re.compile("|".join(_AD_URL_PATTERN_STRINGS), re.IGNORECASE)
    AD_TITLE_COMBINED = re.compile("|".join(_AD_TITLE_PATTERN_STRINGS), re.IGNORECASE)

    # Low-quality domains to deprioritize
    # Optimization: Use frozenset for O(1) lookup and immutability
    LOW_QUALITY_DOMAINS = frozenset({
        "pinterest.com",
        "facebook.com",
        "twitter.com",
        "instagram.com",
        "tiktok.com",
        "reddit.com",  # Often just discussions, not facts
        "quora.com",  # User-generated, unreliable
        "yahoo.com/answers",
    })

    # High-quality domains to prioritize
    # Optimization: Split into exact match set and suffix tuple for O(1) + O(k) lookup
    # Exact domains use frozenset O(1), suffixes use tuple for endswith() check
    HIGH_QUALITY_DOMAINS_EXACT = frozenset({
        "wikipedia.org",
        "britannica.com",
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "npr.org",
        "nature.com",
        "sciencedirect.com",
    })
    # Suffix-based domains (.gov, .edu) - tuple for str.endswith()
    HIGH_QUALITY_SUFFIXES = (".gov", ".edu")

    # Legacy attribute for backward compatibility
    HIGH_QUALITY_DOMAINS = HIGH_QUALITY_DOMAINS_EXACT | {".gov", ".edu"}


@dataclass
class SearchResult:
    """A single search result with quality scoring."""

    title: str
    url: str
    snippet: str
    is_ad: bool = False
    quality_score: float = 0.5
    domain: str = ""

    def __post_init__(self):
        # Extract domain
        try:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc.lower().replace("www.", "")
        except (ValueError, AttributeError):
            self.domain = ""


@dataclass
class WebContent:
    """Extracted web page content."""

    url: str
    title: str
    text: str
    success: bool
    word_count: int = 0
    error: Optional[str] = None
    scrape_time: float = 0.0

    def __post_init__(self):
        if self.text:
            self.word_count = len(self.text.split())


@dataclass
class SearchReport:
    """Comprehensive search report with multiple sources."""

    query: str
    results: List[SearchResult] = field(default_factory=list)
    scraped_content: List[WebContent] = field(default_factory=list)
    total_sources: int = 0
    successful_scrapes: int = 0
    filtered_ads: int = 0
    search_time: float = 0.0

    def get_summary(self) -> str:
        """Get a brief summary of the search."""
        return (
            f"Query: {self.query} | "
            f"Sources: {self.successful_scrapes}/{self.total_sources} | "
            f"Ads filtered: {self.filtered_ads} | "
            f"Time: {self.search_time:.1f}s"
        )


class WebSearchEngine:
    """Enhanced web search with ad filtering and quality scoring."""

    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def search(self, query: str, max_results: int = None) -> Tuple[List[SearchResult], int]:
        """Search DuckDuckGo and return filtered results.

        Returns: (results, num_ads_filtered)
        """
        if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
            log.error("Missing dependencies for web search")
            return [], 0

        max_results = max_results or self.config.MAX_SEARCH_RESULTS

        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

            log.info(f"🔍 Searching: {query}")
            response = requests.get(url, headers=self.headers, timeout=self.config.SEARCH_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            all_results: List[SearchResult] = []
            ads_filtered = 0

            for result_elem in soup.select(".result"):
                title_elem = result_elem.select_one(".result__title")
                link_elem = result_elem.select_one(".result__url")
                snippet_elem = result_elem.select_one(".result__snippet")

                if not title_elem or not link_elem:
                    continue

                # Extract URL
                href = title_elem.select_one("a")
                if href and href.get("href"):
                    raw_url = href.get("href", "")
                    actual_url = self._extract_url(raw_url)
                else:
                    actual_url = link_elem.get_text(strip=True)

                if not actual_url.startswith("http"):
                    actual_url = f"https://{actual_url}"

                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                # Create result
                result = SearchResult(title=title, url=actual_url, snippet=snippet)

                # Check if it's an ad
                result.is_ad = self._is_ad(result, raw_url if href else "")

                if result.is_ad:
                    ads_filtered += 1
                    log.debug(f"🚫 Filtered ad: {title[:50]}")
                    continue

                # Calculate quality score
                result.quality_score = self._calculate_quality(result)

                all_results.append(result)

            # Sort by quality score (highest first)
            all_results.sort(key=lambda r: r.quality_score, reverse=True)

            # Limit results
            final_results = all_results[:max_results]

            log.info(f"✅ Found {len(final_results)} results (filtered {ads_filtered} ads)")
            return final_results, ads_filtered

        except requests.RequestException as e:
            log.error(f"Search failed: {e}")
            return [], 0
        except Exception as e:
            log.error(f"Search error: {e}")
            return [], 0

    def _extract_url(self, ddg_url: str) -> str:
        """Extract real URL from DuckDuckGo redirect."""
        if "uddg=" in ddg_url:
            try:
                parsed = urlparse(ddg_url)
                params = parse_qs(parsed.query)
                if "uddg" in params:
                    return unquote(params["uddg"][0])
            except Exception:
                pass
        return ddg_url

    def _is_ad(self, result: SearchResult, raw_url: str) -> bool:
        """Determine if a result is an advertisement."""
        # Optimization: Use combined regex for single-pass matching instead of N-pattern loop
        # Reduces from O(N) pattern matches to O(1) combined match
        url_to_check = raw_url + result.url
        if self.config.AD_URL_COMBINED.search(url_to_check):
            return True

        if self.config.AD_TITLE_COMBINED.search(result.title):
            return True

        # Check for "Ad" marker in snippet
        if (
            result.snippet.lower().startswith("ad ")
            or "Viewing ads is privacy protected" in result.snippet
        ):
            return True

        return False

    def _calculate_quality(self, result: SearchResult) -> float:
        """Calculate quality score for a result (0.0 to 1.0)."""
        score = 0.5  # Base score

        domain = result.domain

        # Optimization: Use frozenset O(1) lookup for exact matches + tuple endswith() for suffixes
        # Previous: O(N) any() iteration over all patterns
        # Now: O(1) set membership + O(k) suffix check where k = number of suffixes

        # Check suffix-based high-quality domains first (most common boost)
        if domain.endswith(self.config.HIGH_QUALITY_SUFFIXES):
            score += 0.5  # Combined boost for .gov/.edu (was 0.3 + 0.2 separately)
        elif domain in self.config.HIGH_QUALITY_DOMAINS_EXACT or any(
            domain.endswith("." + hq) for hq in self.config.HIGH_QUALITY_DOMAINS_EXACT
        ):
            score += 0.3

        # Penalize low-quality domains - O(1) frozenset lookup
        if domain in self.config.LOW_QUALITY_DOMAINS or any(
            domain.endswith("." + lq) for lq in self.config.LOW_QUALITY_DOMAINS
        ):
            score -= 0.2

        # Boost if snippet is substantial
        if len(result.snippet) > 100:
            score += 0.1

        return max(0.0, min(1.0, score))


class WebScraper:
    """Enhanced web scraper with deeper content extraction."""

    # Optimization: Pre-compile ad pattern regex at class level (compiled once, reused)
    # Previous: re.compile() called in loop for each pattern on each page scrape
    # Now: Compiled once at import time, O(1) lookup per element
    _AD_CLASS_PATTERN = re.compile(
        r"ad|ads|advert|sponsor|promo|banner|cookie|popup", re.IGNORECASE
    )
    # Pre-compile whitespace cleanup patterns
    _WHITESPACE_PATTERN = re.compile(r"[ \t]+")
    _NEWLINE_PATTERN = re.compile(r"\n{3,}")

    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def scrape(self, url: str) -> WebContent:
        """Fetch and extract main content from URL."""
        if not REQUESTS_AVAILABLE:
            return WebContent(
                url=url, title="", text="", success=False, error="requests not installed"
            )

        # Optimization: Use perf_counter for higher precision timing (nanosecond resolution)
        start_time = time.perf_counter()

        try:
            log.info(f"📄 Scraping: {url}")
            response = requests.get(url, headers=self.headers, timeout=self.config.SCRAPE_TIMEOUT)
            response.raise_for_status()
            html = response.text

            # Try trafilatura first (best extraction)
            if TRAFILATURA_AVAILABLE:
                text = trafilatura.extract(
                    html,
                    include_links=False,
                    include_images=False,
                    include_tables=True,
                    include_comments=False,
                    favor_recall=True,  # Get more content
                )
                if text:
                    title = self._extract_title(html)
                    return WebContent(
                        url=url,
                        title=title,
                        text=self._clean_and_truncate(text),
                        success=True,
                        scrape_time=time.perf_counter() - start_time,
                    )

            # Fallback to BeautifulSoup
            if BS4_AVAILABLE:
                text, title = self._bs4_extract(html)
                if text:
                    return WebContent(
                        url=url,
                        title=title,
                        text=self._clean_and_truncate(text),
                        success=True,
                        scrape_time=time.perf_counter() - start_time,
                    )

            return WebContent(
                url=url,
                title="",
                text="",
                success=False,
                error="Could not extract content",
                scrape_time=time.perf_counter() - start_time,
            )

        except requests.Timeout:
            log.warning(f"⏱️ Timeout scraping: {url}")
            return WebContent(url=url, title="", text="", success=False, error="Timeout")
        except requests.RequestException as e:
            log.error(f"Scrape failed: {e}")
            return WebContent(url=url, title="", text="", success=False, error=str(e))
        except Exception as e:
            log.error(f"Scrape error: {e}")
            return WebContent(url=url, title="", text="", success=False, error=str(e))

    def _bs4_extract(self, html: str) -> Tuple[str, str]:
        """Extract content using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "aside",
                "form",
                "button",
                "iframe",
                "noscript",
                "svg",
            ]
        ):
            tag.decompose()

        # Optimization: Use single pre-compiled pattern for ad element removal
        # Previous: 8 patterns x 2 (class + id) = 16 re.compile() calls per scrape
        # Now: 1 pre-compiled pattern, 2 find_all() calls total
        for elem in soup.find_all(class_=self._AD_CLASS_PATTERN):
            elem.decompose()
        for elem in soup.find_all(id=self._AD_CLASS_PATTERN):
            elem.decompose()

        title = soup.title.string if soup.title else ""

        # Try to find main content in order of preference
        main_content = None
        for selector in [
            "main",
            "article",
            '[role="main"]',
            ".content",
            ".post",
            ".entry",
            "#content",
            "body",
        ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content:
            # Get text with proper spacing
            text = main_content.get_text(separator="\n", strip=True)
            # Optimization: Use pre-compiled pattern instead of re.sub with string pattern
            text = self._NEWLINE_PATTERN.sub("\n\n", text)
            return text, title

        return "", title

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            if soup.title:
                return soup.title.string or ""
        match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        return match.group(1) if match else ""

    def _clean_and_truncate(self, text: str) -> str:
        """Clean up text and truncate to max length."""
        # Optimization: Use pre-compiled patterns instead of re.sub with string patterns
        # re.sub compiles the pattern on each call when given a string
        text = self._WHITESPACE_PATTERN.sub(" ", text)
        text = self._NEWLINE_PATTERN.sub("\n\n", text)
        text = text.strip()

        # Truncate if needed
        if len(text) <= self.config.MAX_CONTENT_LENGTH:
            return text

        # Try to truncate at a sentence boundary
        truncated = text[: self.config.MAX_CONTENT_LENGTH]
        last_period = truncated.rfind(". ")
        if last_period > self.config.MAX_CONTENT_LENGTH * 0.8:
            return truncated[: last_period + 1] + "\n[Content truncated...]"

        return truncated + "...\n[Content truncated...]"


class ResearchEngine:
    """Performs comprehensive research across multiple sources."""

    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.search_engine = WebSearchEngine(self.config)
        self.scraper = WebScraper(self.config)

    def research(self, query: str, max_sources: int = None) -> SearchReport:
        """Perform comprehensive research on a topic.
        Returns a SearchReport with all findings.
        """
        start_time = time.time()
        max_sources = max_sources or self.config.MAX_SCRAPE_RESULTS

        report = SearchReport(query=query)

        # Step 1: Search
        results, ads_filtered = self.search_engine.search(query)
        report.results = results
        report.filtered_ads = ads_filtered
        report.total_sources = len(results)

        if not results:
            report.search_time = time.time() - start_time
            return report

        # Step 2: Scrape top results
        sources_to_scrape = results[:max_sources]

        for i, result in enumerate(sources_to_scrape):
            log.info(f"📖 Reading source {i + 1}/{len(sources_to_scrape)}: {result.domain}")

            content = self.scraper.scrape(result.url)
            report.scraped_content.append(content)

            if content.success:
                report.successful_scrapes += 1

            # Rate limiting
            if i < len(sources_to_scrape) - 1:
                time.sleep(self.config.REQUEST_DELAY)

        report.search_time = time.time() - start_time
        log.info(f"✅ Research complete: {report.get_summary()}")

        return report

    def format_for_llm(self, report: SearchReport, include_comparison_prompt: bool = True) -> str:
        """Format research report for LLM consumption with comparison instructions."""
        if not report.scraped_content:
            return f"No information found for: {report.query}"

        sections = []

        # Header
        sections.append(f"═══ WEB RESEARCH: {report.query} ═══")
        sections.append(
            f"Sources consulted: {report.successful_scrapes} | Time: {report.search_time:.1f}s"
        )
        sections.append("")

        # Each source
        total_length = 0
        for i, content in enumerate(report.scraped_content):
            if not content.success:
                continue

            # Check if we're exceeding total context limit
            if total_length > self.config.MAX_TOTAL_CONTEXT:
                sections.append("\n[Additional sources truncated to fit context limit]")
                break

            result = report.results[i] if i < len(report.results) else None
            domain = result.domain if result else urlparse(content.url).netloc

            source_section = []
            source_section.append(f"─── SOURCE {i + 1}: {domain} ───")
            source_section.append(f"Title: {content.title}")
            source_section.append(f"URL: {content.url}")
            source_section.append(f"Words: {content.word_count}")
            source_section.append("")
            source_section.append(content.text)
            source_section.append("")

            source_text = "\n".join(source_section)
            total_length += len(source_text)
            sections.append(source_text)

        # Comparison instructions for LLM
        if include_comparison_prompt:
            sections.append("─── INSTRUCTIONS ───")
            sections.append(
                "You have been provided information from multiple sources. "
                "Please synthesize this information to provide a comprehensive answer. "
                "If sources disagree, mention the different perspectives. "
                "Cite sources naturally (e.g., 'According to [domain]...'). "
                "If information seems outdated or uncertain, note that."
            )

        sections.append("═══ END OF RESEARCH ═══")

        return "\n".join(sections)


class WebTools:
    """Main interface for web search capabilities."""

    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.research_engine = ResearchEngine(self.config)
        self.search_engine = WebSearchEngine(self.config)
        self.scraper = WebScraper(self.config)

    def quick_search(self, query: str, max_results: int = 5) -> str:
        """Quick search - just snippets, no scraping.
        Fast but less detailed.
        """
        results, _ = self.search_engine.search(query, max_results)

        if not results:
            return "No search results found."

        lines = [f"=== Quick Search: {query} ===\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.title}")
            lines.append(f"   Source: {r.domain}")
            lines.append(f"   {r.snippet}\n")

        return "\n".join(lines)

    def search(self, query: str, max_sources: int = 5) -> str:
        """Standard search - searches and scrapes multiple sources."""
        report = self.research_engine.research(query, max_sources)
        return self.research_engine.format_for_llm(report)

    def deep_search(self, query: str, max_sources: int = 10) -> str:
        """Deep search - maximum sources, comprehensive research."""
        report = self.research_engine.research(query, max_sources)
        return self.research_engine.format_for_llm(report, include_comparison_prompt=True)

    def read_url(self, url: str) -> str:
        """Scrape a specific URL."""
        content = self.scraper.scrape(url)

        if not content.success:
            return f"Failed to read {url}: {content.error}"

        return f"=== Content from: {content.title} ===\nURL: {url}\n\n{content.text}"


_web_tools: Optional[WebTools] = None
_config: Optional[SearchConfig] = None


def get_config() -> SearchConfig:
    """Get or create config singleton."""
    global _config
    if _config is None:
        _config = SearchConfig()
    return _config


def get_web_tools() -> WebTools:
    """Get or create WebTools singleton."""
    global _web_tools
    if _web_tools is None:
        _web_tools = WebTools(get_config())
    return _web_tools


# Convenience functions
def web_search(query: str, max_results: int = 5) -> str:
    """Standard search with scraping."""
    return get_web_tools().search(query, max_results)


def web_search_quick(query: str, max_results: int = 5) -> str:
    """Quick search - snippets only."""
    return get_web_tools().quick_search(query, max_results)


def web_search_deep(query: str, max_sources: int = 10) -> str:
    """Deep search - comprehensive research."""
    return get_web_tools().deep_search(query, max_sources)


def web_search_and_read(query: str, max_results: int = 5) -> str:
    """Alias for web_search (backward compatibility)."""
    return web_search(query, max_results)


def web_read(url: str) -> str:
    """Read a specific URL."""
    return get_web_tools().read_url(url)


if __name__ == "__main__":
    print("Testing Web Search Module")
    result = web_search("Python programming language", max_results=5)
    print(result[:2000] + "..." if len(result) > 2000 else result)
