"""
src/search_client.py
DuckDuckGo search + direct httpx fetch + trafilatura content extraction.
No external API keys required.

Cascade search strategy: tries multiple identifier combinations until one
returns results with at least one identifier code in URL/title/snippet.
"""

import time
import httpx
import trafilatura
from dataclasses import dataclass

from ddgs import DDGS

from .content_cleaner import extract_jsonld


# Country code → DDG region (lang-region format used by ddgs)
DDG_REGIONS = {
    "AU": "au-en",
    "US": "us-en",
    "UK": "uk-en",
    "NZ": "nz-en",
    "CA": "ca-en",
    "DE": "de-de",
    "FR": "fr-fr",
    "JP": "jp-jp",
    "SG": "sg-en",
    "IN": "in-en",
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


# ── Config and identifier types ───────────────────────────────────────────────

@dataclass
class IdentifierSet:
    """Per-row identifiers used for cascade search and validation.

    Empty fields are simply skipped — only populated identifiers contribute
    search stages and validation matches.
    """
    sku:               str = ""
    manufacturer_code: str = ""
    brand:             str = ""
    product_name:      str = ""

    def code_identifiers(self) -> list[str]:
        """Codes used for relevance matching in search results and content."""
        codes = []
        for c in (self.sku, self.manufacturer_code):
            c = (c or "").strip()
            if c and c not in codes:
                codes.append(c)
        return codes

    def is_empty(self) -> bool:
        return not any([self.sku, self.manufacturer_code,
                        self.brand, self.product_name])

    def display_label(self) -> str:
        """Short label for progress UI — uses the first populated identifier."""
        for v in (self.sku, self.manufacturer_code, self.product_name):
            if v:
                return str(v)
        return "(unknown)"


@dataclass
class SearchConfig:
    country_code:  str   = "AU"
    urls_per_sku:  int   = 2
    max_chars:     int   = 4000
    timeout:       int   = 25
    max_results:   int   = 10
    delay_between: float = 0.5


class RateLimitError(Exception):
    pass


# ── Search ────────────────────────────────────────────────────────────────────

def _ddg_search(query: str, region: str, max_results: int) -> list[dict]:
    """One DuckDuckGo query. Returns [{url, title, snippet}, ...]."""
    if not query.strip():
        return []
    try:
        with DDGS() as ddgs:
            raw = ddgs.text(query, region=region, max_results=max_results) or []
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("ratelimit", "rate limit", "too many", "202", "429")):
            raise RateLimitError(f"DuckDuckGo rate limited: {e}")
        # Other transport errors: treat as empty result, let the cascade try next
        return []

    results = []
    for r in raw:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        results.append({
            "url":     url,
            "title":   r.get("title", "") or "",
            "snippet": r.get("body", "") or r.get("description", "") or "",
        })
    return results


def _result_matches_any(result: dict, identifiers: list[str]) -> bool:
    """True if any identifier appears in the result's URL, title, or snippet."""
    if not identifiers:
        return False
    haystack = " ".join([
        result.get("url", "") or "",
        result.get("title", "") or "",
        result.get("snippet", "") or "",
    ]).lower()
    return any(i.lower() in haystack for i in identifiers if i)


def _looks_relevant(results: list[dict], codes: list[str]) -> bool:
    if not results:
        return False
    if not codes:
        return len(results) >= 2  # no codes to match against — use volume
    return any(_result_matches_any(r, codes) for r in results)


def _build_cascade(ids: IdentifierSet) -> list[str]:
    """Build the ordered list of search queries to try."""
    stages = []

    mfr  = ids.manufacturer_code.strip() if ids.manufacturer_code else ""
    sku  = ids.sku.strip() if ids.sku else ""
    brand = ids.brand.strip() if ids.brand else ""
    name  = ids.product_name.strip() if ids.product_name else ""

    # 1. Manufacturer code is the strongest signal — try it quoted first
    if mfr:
        stages.append(f'"{mfr}"')

    # 2. SKU quoted (skip if it's identical to mfr code)
    if sku and sku.lower() != mfr.lower():
        stages.append(f'"{sku}"')

    # 3. Mfr code + brand (helps when the bare code is ambiguous)
    if mfr and brand:
        stages.append(f'"{mfr}" {brand}')

    # 4. SKU + brand
    if sku and brand and sku.lower() != mfr.lower():
        stages.append(f'"{sku}" {brand}')

    # 5. Product name as the last-resort wide net
    if name:
        # Combine with brand if available — often sharpens niche product searches
        stages.append(f'{name} {brand}'.strip() if brand else name)

    return stages


def search_for_product(ids: IdentifierSet, cfg: SearchConfig) -> tuple[list[dict], str]:
    """
    Run the cascade. Returns (results, query_used).
    Stops at the first stage whose results contain at least one of the
    identifier codes in URL/title/snippet. Falls back to the largest
    non-empty result set if no stage matches strongly.
    """
    region = DDG_REGIONS.get(cfg.country_code.upper(), "au-en")
    codes  = ids.code_identifiers()

    cascade = _build_cascade(ids)
    if not cascade:
        return [], ""

    best_results = []
    best_query   = ""

    for query in cascade:
        results = _ddg_search(query, region, cfg.max_results)

        if _looks_relevant(results, codes):
            return results, query

        if len(results) > len(best_results):
            best_results = results
            best_query   = query

    return best_results, best_query


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
    Direct httpx GET → trafilatura content extraction. JSON-LD is pulled from
    the raw HTML before extraction so the structured-data hint survives.

    Returns {content, jsonld, raw_status} or None if the page couldn't be
    usefully fetched (4xx/5xx, blocked, JS-only, or extracted < 200 chars).
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

    # 1. JSON-LD from raw HTML — must run before trafilatura strips scripts
    jsonld = extract_jsonld(html)

    # 2. Main content via trafilatura. favor_precision drops more boilerplate
    #    at the cost of occasional content loss on ambiguous pages — safer for
    #    LLM extraction where a clean signal beats a noisy maximum.
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

    # 3. Truncate at max_chars on a paragraph boundary
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
    Full per-product pipeline: cascade search → score → fetch.
    Returns (pages, status, query_used).

    Each page is {url, content, jsonld_hint}.
    """
    if ids.is_empty():
        return [], "not_found", ""

    try:
        results, query = search_for_product(ids, cfg)
    except RateLimitError:
        return [], "rate_limited", ""
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
    """
    Sanity check: do any identifier codes appear in the fetched page content?
    Returns True if no codes are configured (nothing to validate against) OR
    at least one code appears in the content.

    Used to flag pages that may be about a different product entirely.
    """
    codes = ids.code_identifiers()
    if not codes:
        return True
    if not content:
        return False
    lower = content.lower()
    return any(c.lower() in lower for c in codes if c)
