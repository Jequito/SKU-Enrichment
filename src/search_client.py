"""
src/search_client.py
SerpAPI Google search + direct httpx fetch + trafilatura content extraction.

Strategy: exact-match Google search on a user-mapped Primary column. If that
returns zero results, retry on a user-mapped Fallback column. Both column
mappings are user-configured in the sidebar — no hardcoded priorities.

Per row: 1 SerpAPI query when the primary works, 2 when it doesn't.
"""

import time
import httpx
import trafilatura
from dataclasses import dataclass

from .content_cleaner import extract_jsonld


# Country code → SerpAPI gl param (lowercase ISO 3166-1 alpha-2,
# matching Google's country code conventions — UK is "gb")
SERPAPI_COUNTRIES = {
    "AU": "au", "US": "us", "UK": "gb", "NZ": "nz", "CA": "ca",
    "DE": "de", "FR": "fr", "JP": "jp", "SG": "sg", "IN": "in",
}

COUNTRY_LABELS = {
    "AU": "🇦🇺  Australia",       "US": "🇺🇸  United States",
    "UK": "🇬🇧  United Kingdom",  "NZ": "🇳🇿  New Zealand",
    "CA": "🇨🇦  Canada",           "DE": "🇩🇪  Germany",
    "FR": "🇫🇷  France",           "JP": "🇯🇵  Japan",
    "SG": "🇸🇬  Singapore",        "IN": "🇮🇳  India",
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

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


# ── Identifiers ───────────────────────────────────────────────────────────────

@dataclass
class IdentifierSet:
    """Per-row identifier values pulled from the user's mapped columns.

    primary  is searched first (exact-quoted Google).
    fallback is searched only if the primary search returns zero results.
    Both are also passed to the LLM as match anchors and used for content
    validation.
    """
    primary:  str = ""
    fallback: str = ""

    def codes(self) -> list[str]:
        """All non-empty identifier values, deduped."""
        out = []
        for v in (self.primary, self.fallback):
            v = (v or "").strip()
            if v and v not in out:
                out.append(v)
        return out

    def is_empty(self) -> bool:
        return not (self.primary.strip() or self.fallback.strip())

    def display_label(self) -> str:
        """Short label for progress UI."""
        return self.primary.strip() or self.fallback.strip() or "(unknown)"


@dataclass
class SearchConfig:
    serpapi_api_key: str   = ""
    country_code:    str   = "AU"
    urls_per_sku:    int   = 2
    max_chars:       int   = 4000
    timeout:         int   = 25
    max_results:     int   = 10
    delay_between:   float = 0.5


class RateLimitError(Exception):
    pass


class BackendConfigError(Exception):
    """SerpAPI is misconfigured (missing key, exhausted quota, account block)."""
    pass


# ── Search ────────────────────────────────────────────────────────────────────

def search(query: str, cfg: SearchConfig) -> list[dict]:
    """
    One Google search via SerpAPI. Returns [{url, title, snippet}, ...].

    SerpAPI billing note: only successful searches that return data count
    against quota. Empty results, errors, and rate-limited requests are not
    charged — so a primary-search miss followed by a fallback search costs
    only 1 quota credit, not 2.
    """
    if not query.strip():
        return []
    if not cfg.serpapi_api_key:
        raise BackendConfigError("SerpAPI key not configured")

    gl = SERPAPI_COUNTRIES.get(cfg.country_code.upper(), "au")
    params = {
        "api_key": cfg.serpapi_api_key,
        "engine":  "google",
        "q":       query,
        "gl":      gl,
        "hl":      "en",
        "num":     min(cfg.max_results, 20),
    }

    try:
        with httpx.Client(timeout=cfg.timeout) as client:
            r = client.get("https://serpapi.com/search.json", params=params)
            status = r.status_code
            if status == 429:
                raise RateLimitError("SerpAPI rate limited (429)")
            if status in (401, 403):
                raise BackendConfigError(
                    f"SerpAPI key rejected ({status}). "
                    "Check the key at serpapi.com/manage-api-key"
                )
            r.raise_for_status()
            data = r.json()
    except (RateLimitError, BackendConfigError):
        raise
    except Exception:
        # Transport/JSON parse errors — return empty so the caller can decide
        return []

    # SerpAPI surfaces application-level errors in its JSON body
    err_msg = data.get("error")
    if err_msg:
        msg_lower = str(err_msg).lower()
        if "invalid api key" in msg_lower or "expired" in msg_lower:
            raise BackendConfigError(f"SerpAPI: {err_msg}")
        if any(k in msg_lower for k in ("run out of searches", "quota", "exceed")):
            raise RateLimitError(f"SerpAPI quota exhausted: {err_msg}")
        raise BackendConfigError(f"SerpAPI error: {err_msg}")

    organic = data.get("organic_results", []) or []
    results = []
    for r in organic[:cfg.max_results]:
        url = r.get("link", "")
        if not url:
            continue
        results.append({
            "url":     url,
            "title":   r.get("title", "") or "",
            "snippet": r.get("snippet", "") or "",
        })
    return results


def search_for_product(ids: IdentifierSet, cfg: SearchConfig) -> tuple[list[dict], str]:
    """
    Two-stage exact-match search:
      1. `"primary_value"` — quoted Google search via SerpAPI
      2. `"fallback_value"` — only runs if stage 1 returned zero results

    Returns (results, query_used). Empty results means neither column found
    anything on Google — the row is genuinely not findable, no further
    fallbacks are tried.
    """
    primary_val  = (ids.primary or "").strip()
    fallback_val = (ids.fallback or "").strip()

    # Stage 1: primary
    if primary_val:
        query = f'"{primary_val}"'
        results = search(query, cfg)
        if results:
            return results, query

    # Stage 2: fallback (only if primary was empty or returned nothing)
    if fallback_val and fallback_val.lower() != primary_val.lower():
        query = f'"{fallback_val}"'
        results = search(query, cfg)
        if results:
            return results, query

    return [], ""


def score_url(result: dict) -> int:
    url    = (result.get("url") or "").lower()
    title  = (result.get("title") or "").lower()
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


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_page(url: str, cfg: SearchConfig) -> dict | None:
    """
    Direct httpx GET → trafilatura content extraction. JSON-LD is pulled
    from raw HTML before extraction so the structured-data hint survives.
    """
    try:
        with httpx.Client(headers=_BROWSER_HEADERS, follow_redirects=True,
                          timeout=cfg.timeout) as client:
            r = client.get(url)
            status = r.status_code
            if status >= 400:
                return None
            html = r.text
    except Exception:
        return None

    if not html or len(html) < 100:
        return None

    jsonld = extract_jsonld(html)

    extracted = trafilatura.extract(
        html,
        include_links=False,
        include_tables=True,
        include_comments=False,
        output_format="markdown",
        favor_precision=True,
    )

    if not extracted or len(extracted) < 200:
        return None

    if len(extracted) > cfg.max_chars:
        truncated = extracted[:cfg.max_chars]
        cut       = truncated.rfind("\n\n")
        if cut > int(cfg.max_chars * 0.7):
            truncated = truncated[:cut]
        extracted = truncated

    return {"content": extracted, "jsonld": jsonld, "status": status}


def fetch_pages_for_product(
    ids: IdentifierSet,
    cfg: SearchConfig,
) -> tuple[list[dict], str, str]:
    """
    Per-product pipeline: search → score top URLs → fetch.
    Returns (pages, status, query_used).
    """
    if ids.is_empty():
        return [], "not_found", ""

    try:
        results, query = search_for_product(ids, cfg)
    except RateLimitError:
        return [], "rate_limited", ""
    except BackendConfigError:
        # Misconfigured SerpAPI propagates so the pipeline shows a real error
        # instead of silently logging not_found for every row in the batch.
        raise
    except Exception:
        return [], "not_found", ""

    if not results:
        return [], "not_found", query

    top_urls = select_top_urls(results, cfg.urls_per_sku)
    pages    = []

    for result in top_urls:
        try:
            fetched = fetch_page(result["url"], cfg)
        except Exception:
            fetched = None

        if fetched and len(fetched["content"]) > 200:
            pages.append({
                "url":         result["url"],
                "content":     fetched["content"],
                "jsonld_hint": fetched.get("jsonld"),
            })
        time.sleep(cfg.delay_between)

    if not pages:
        return [], "blocked", query
    return pages, "ok", query


def content_validates_product(content: str, ids: IdentifierSet) -> bool:
    """Sanity check: does primary or fallback value appear in the fetched page?"""
    codes = ids.codes()
    if not codes:
        return True
    if not content:
        return False
    lower = content.lower()
    return any(c.lower() in lower for c in codes if c)
