"""
src/jina_client.py
Handles all Jina AI interactions — search via s.jina.ai and page fetch via r.jina.ai.
Applies content cleaning after every fetch.
"""

import urllib.request
import urllib.parse
import urllib.error
import re
import time
from dataclasses import dataclass, field

from .content_cleaner import clean_content

# Country code → (gl param, location city)
COUNTRY_PARAMS = {
    "AU": ("AU", "Melbourne"),
    "US": ("US", "New York"),
    "UK": ("GB", "London"),
    "NZ": ("NZ", "Auckland"),
    "CA": ("CA", "Toronto"),
    "DE": ("DE", "Berlin"),
    "FR": ("FR", "Paris"),
    "JP": ("JP", "Tokyo"),
    "SG": ("SG", "Singapore"),
    "IN": ("IN", "Mumbai"),
}

LOW_VALUE_DOMAINS = {
    "reddit.com", "quora.com", "pinterest.com", "instagram.com",
    "facebook.com", "twitter.com", "x.com", "youtube.com", "tiktok.com",
    "gumtree.com", "craigslist.org", "alibaba.com", "aliexpress.com",
    "wish.com", "ebay.com.au", "answers.yahoo.com",
}

HIGH_VALUE_SIGNALS = [
    "/product/", "/products/", "/item/", "/p/", "/catalogue/",
    "/shop/", "/detail/", "productdetail", "specifications", "specs",
    "datasheet", "/buy/",
]


@dataclass
class JinaConfig:
    api_key:          str   = ""
    country_code:     str   = "AU"
    urls_per_sku:     int   = 2
    max_chars:        int   = 4000
    timeout:          int   = 25
    no_cache:         bool  = True
    return_format:    str   = "markdown"
    search_only:      bool  = True
    delay_between:    float = 0.5
    retry_on_few:     bool  = True
    # DOM-level content targeting — applied by Jina before markdown conversion.
    # Far more reliable than post-hoc Python heuristics because they operate on
    # the real DOM nodes, not on the resulting markdown line patterns.
    target_selector:  str   = ""  # x-target-selector: only include matching elements
    remove_selector:  str   = ""  # x-remove-selector: strip matching elements
    use_readerlm:     bool  = False  # x-respond-with: readerlm-v2 (3x token cost, higher quality)


def _base_headers(api_key: str = "") -> dict:
    h = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/plain, */*",
    }
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def search(query: str, cfg: JinaConfig) -> list[dict]:
    gl, location  = COUNTRY_PARAMS.get(cfg.country_code.upper(), ("AU", "Melbourne"))
    encoded_query = urllib.parse.quote(query)
    encoded_loc   = urllib.parse.quote(location)
    url           = f"https://s.jina.ai/?q={encoded_query}&gl={gl}&location={encoded_loc}"

    headers = {**_base_headers(cfg.api_key), "Accept": "application/json"}
    if cfg.search_only:
        headers["X-Respond-With"] = "no-content"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            import json
            data    = json.loads(resp.read().decode("utf-8", errors="replace"))
            results = data.get("data", [])
            return [
                {"url": r.get("url",""), "title": r.get("title",""), "snippet": r.get("description","")}
                for r in results if r.get("url")
            ]
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimitError("Jina search rate limited (429). Add/check your Jina API key.")
        raise
    except Exception as e:
        raise JinaError(f"Search failed: {e}")


# Minimum confidence threshold for SKU search results.
# Results below this score are treated as a poor match and trigger a
# secondary-term retry if one is configured.
_LOW_CONFIDENCE_THRESHOLD = 0.4


def _results_look_relevant(results: list[dict], sku: str) -> bool:
    """
    Heuristic: decide whether search results are good enough to proceed.
    Returns False (triggering a secondary-term retry) when:
      - No results were returned at all, OR
      - The top result's URL/title doesn't contain the SKU and the average
        score across all results is below our confidence threshold.
    This catches the common case where a short or ambiguous SKU returns
    completely unrelated pages before we waste a fetch on them.
    """
    if not results:
        return False
    sku_lower = sku.lower()
    top = results[0]
    top_url   = (top.get("url")   or "").lower()
    top_title = (top.get("title") or "").lower()
    sku_in_top = sku_lower in top_url or sku_lower in top_title
    avg_score = sum(r.get("score", 0) for r in results) / len(results)
    return sku_in_top or avg_score >= _LOW_CONFIDENCE_THRESHOLD


def search_for_sku(sku: str, cfg: JinaConfig, secondary_term: str = "") -> list[dict]:
    """
    Search for a SKU using a two-stage fallback strategy.

    Stage 1: Search with the exact SKU quoted (e.g. "ABC123").
             If results look relevant, return them immediately.

    Stage 2: Only if Stage 1 results look off (wrong product, low score,
             no URL match), retry using the secondary_term as a standalone
             query — NOT combined with the SKU. This handles the common case
             where a short or manufacturer-internal SKU returns unrelated
             pages, but the secondary column (e.g. a product name or brand)
             produces much better results on its own.

    The two queries are deliberately kept separate. Combining them (e.g.
    "ABC123 Bosch Drill") typically yields no results because the SKU and
    the name rarely co-occur verbatim on the same page.
    """
    sku_results = search(f'"{sku}"', cfg)

    if _results_look_relevant(sku_results, sku):
        # SKU search looks good — apply the existing retry-on-few logic
        if len(sku_results) < 2 and cfg.retry_on_few:
            fallback = search(f'"{sku}" specifications', cfg)
            if len(fallback) > len(sku_results):
                sku_results = fallback
        return sku_results

    # SKU search returned poor results — try the secondary term standalone
    if secondary_term:
        secondary_results = search(secondary_term, cfg)
        if secondary_results:
            return secondary_results

    # No secondary term, or secondary also returned nothing — return whatever
    # the SKU search gave us (even if weak) so the pipeline can still attempt
    # extraction and flag the row for review.
    return sku_results


def score_url(result: dict) -> int:
    url    = result.get("url","").lower()
    title  = result.get("title","").lower()
    domain = url.split("/")[2] if url.count("/") >= 2 else url
    if url.endswith(".pdf"):
        return -1
    if any(bad in domain for bad in LOW_VALUE_DOMAINS):
        return -1
    score = 0
    for sig in HIGH_VALUE_SIGNALS:
        if sig in url:
            score += 2
    if "spec" in title or "product" in title:
        score += 1
    if domain.count(".") == 1:
        score += 3
    return score


def select_top_urls(results: list[dict], n: int) -> list[dict]:
    scored = [(score_url(r), r) for r in results]
    scored = [(s, r) for s, r in scored if s >= 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for _, r in scored[:n]]
    return top if top else results[:n]


def fetch_page(url: str, cfg: JinaConfig) -> str | None:
    """Fetch via Jina Reader and return cleaned content."""
    encoded  = urllib.parse.quote(url, safe="")
    jina_url = f"https://r.jina.ai/{encoded}"

    headers = {
        **_base_headers(cfg.api_key),
        "X-Return-Format": cfg.return_format,
    }
    if cfg.no_cache:
        headers["X-No-Cache"] = "true"
    # DOM-level selectors — strip nav/footer/ads before Jina converts to markdown.
    # x-target-selector tells Jina to return ONLY the matching elements.
    # x-remove-selector strips elements unconditionally.
    # Using both is valid: Jina applies remove first, then target.
    if cfg.target_selector:
        headers["X-Target-Selector"] = cfg.target_selector
    if cfg.remove_selector:
        headers["X-Remove-Selector"] = cfg.remove_selector
    # ReaderLM-v2 is a dedicated SLM for HTML→markdown conversion. It produces
    # significantly cleaner output on complex pages at 3x the token cost.
    if cfg.use_readerlm:
        headers["X-Respond-With"] = "readerlm-v2"

    req = urllib.request.Request(jina_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimitError("Jina fetch rate limited (429). Add/check your Jina API key.")
        raw = _fetch_direct(url, cfg)
    except Exception:
        raw = _fetch_direct(url, cfg)

    if not raw:
        return None

    # Apply content-aware cleaning
    return clean_content(raw, cfg.max_chars)


def _fetch_direct(url: str, cfg: JinaConfig) -> str | None:
    try:
        req = urllib.request.Request(url, headers=_base_headers())
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"\s{2,}", " ", raw).strip()
        return raw
    except Exception:
        return None


def fetch_pages_for_sku(sku: str, cfg: JinaConfig, secondary_term: str = "") -> tuple[list[dict], str]:
    """
    Full pipeline for one SKU: search → score → fetch → clean.
    Returns (pages, status) where pages = [{url, content}, ...].
    secondary_term: value from the secondary column — used as a standalone
    fallback query if the SKU search returns irrelevant results.
    """
    try:
        results = search_for_sku(sku, cfg, secondary_term=secondary_term)
    except RateLimitError:
        return [], "rate_limited"
    except Exception:
        return [], "not_found"

    if not results:
        return [], "not_found"

    top_urls = select_top_urls(results, cfg.urls_per_sku)
    pages    = []

    for result in top_urls:
        try:
            content = fetch_page(result["url"], cfg)
            if content and len(content) > 200:
                pages.append({"url": result["url"], "content": content})
        except RateLimitError:
            return pages, "rate_limited"
        except Exception:
            pass
        time.sleep(cfg.delay_between)

    if not pages:
        return [], "blocked"

    return pages, "ok"


class JinaError(Exception):
    pass

class RateLimitError(JinaError):
    pass
