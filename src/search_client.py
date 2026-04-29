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

# Google `lr` language-restrict codes — filters out pages in other languages.
# This is the cleanest fix for the "Polish/Czech reseller pages keep showing up
# for AU searches" problem since SerpAPI's gl param only biases, doesn't filter.
SERPAPI_LANGUAGES = {
    "AU": "lang_en", "US": "lang_en", "UK": "lang_en",
    "NZ": "lang_en", "CA": "lang_en", "SG": "lang_en", "IN": "lang_en",
    "DE": "lang_de", "FR": "lang_fr", "JP": "lang_ja",
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
    serpapi_api_key:    str   = ""
    blocked_domains:    tuple = ()      # extra user-blocked domains (substring match)
    country_code:       str   = "AU"
    restrict_language:  bool  = True    # send lr=lang_X — filters foreign-language pages
    restrict_country:   bool  = False   # send cr=country — stricter, can exclude legit pages
    urls_per_sku:       int   = 2
    max_chars:          int   = 4000
    timeout:            int   = 25
    max_results:        int   = 10
    delay_between:      float = 0.5


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

    # Optional language restriction — filters out foreign-language pages.
    # Without this, an AU search for "MPN-1234" can still surface Polish or
    # German reseller pages that happen to mention the same code.
    if cfg.restrict_language:
        lr = SERPAPI_LANGUAGES.get(cfg.country_code.upper())
        if lr:
            params["lr"] = lr

    # Optional country restriction — much stricter than gl. Google's "country
    # of origin" detection can occasionally exclude legitimate pages, so this
    # is OFF by default. Useful when you genuinely only want, say, AU domains.
    if cfg.restrict_country:
        cc = "GB" if cfg.country_code.upper() == "UK" else cfg.country_code.upper()
        params["cr"] = f"country{cc}"

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


def _has_relevant_match(results: list[dict], identifier: str) -> bool:
    """
    Did Google actually find what we asked for, or just adjacent junk?

    Uses normalised matching (strips hyphens, spaces, punctuation, lowercases)
    against the URL + title + snippet of each result. If at least one result
    contains the normalised identifier as a substring, we consider the search
    "relevant" and don't trigger the fallback. Otherwise the primary search
    is treated as a miss even though it returned hits.

    The normalisation lets 'LDS-MAUI5' match 'LDS-MAUI-5-W-SUB' but reject
    pages that only mention the brand without the specific model code.
    """
    norm_id = _normalise_for_match(identifier)
    if not norm_id:
        return True   # nothing to check against — assume relevant
    for r in results:
        haystack = " ".join([
            r.get("url", "") or "",
            r.get("title", "") or "",
            r.get("snippet", "") or "",
        ])
        if norm_id in _normalise_for_match(haystack):
            return True
    return False


def search_for_product(ids: IdentifierSet, cfg: SearchConfig) -> tuple[list[dict], str]:
    """
    Two-stage exact-match search with relevance trigger:

      1. `"primary_value"` — quoted Google search via SerpAPI.
         If results are returned AND at least one of them mentions the
         primary identifier in URL/title/snippet (after hyphen-insensitive
         normalisation), use them.

      2. `"fallback_value"` — runs when stage 1 returned zero results OR
         when stage 1 returned results that don't actually contain the
         primary identifier (Google's quote-handling sometimes loosens up
         on rare codes and returns adjacent products).

    Returns (results, query_used). If both stages fail the relevance bar but
    the primary returned *something*, those weak results are returned anyway
    so the row gets at least one fetch attempt.
    """
    primary_val  = (ids.primary or "").strip()
    fallback_val = (ids.fallback or "").strip()

    primary_results: list[dict] = []
    primary_query   = ""

    # Stage 1: primary
    if primary_val:
        primary_query = f'"{primary_val}"'
        primary_results = search(primary_query, cfg)
        if primary_results and _has_relevant_match(primary_results, primary_val):
            return primary_results, primary_query

    # Stage 2: fallback — runs on either zero results or weak/irrelevant results
    if fallback_val and fallback_val.lower() != primary_val.lower():
        fallback_query  = f'"{fallback_val}"'
        fallback_results = search(fallback_query, cfg)
        if fallback_results:
            return fallback_results, fallback_query

    # Fallback didn't help either. If primary returned weak-but-nonempty
    # results, return them — better to attempt a fetch on a borderline page
    # than mark the row NOT_FOUND when Google did surface something.
    if primary_results:
        return primary_results, primary_query
    return [], ""


def score_url(result: dict, extra_blocked: set | None = None) -> int:
    url    = (result.get("url") or "").lower()
    title  = (result.get("title") or "").lower()
    domain = url.split("/")[2] if url.count("/") >= 2 else url
    if url.endswith(".pdf"):
        return -1
    blocked = LOW_VALUE_DOMAINS | (extra_blocked or set())
    if any(bad in domain for bad in blocked):
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


def select_top_urls(results: list[dict], n: int, extra_blocked: set | None = None) -> list[dict]:
    scored = [(score_url(r, extra_blocked), r) for r in results]
    scored = [(s, r) for s, r in scored if s >= 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for _, r in scored[:n]]
    return top if top else []


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_page(url: str, cfg: SearchConfig) -> dict:
    """
    Direct httpx GET → trafilatura content extraction. JSON-LD is pulled
    from raw HTML before extraction so the structured-data hint survives.

    Always returns a dict. On success: {ok: True, content, jsonld, status}.
    On failure: {ok: False, error: "..."} with a brief reason — used by the
    pipeline to surface BLOCKED diagnostics in the UI.
    """
    try:
        with httpx.Client(headers=_BROWSER_HEADERS, follow_redirects=True,
                          timeout=cfg.timeout) as client:
            r = client.get(url)
            status = r.status_code
            if status >= 400:
                return {"ok": False, "error": f"HTTP {status}"}
            html = r.text
    except httpx.TimeoutException:
        return {"ok": False, "error": "Timeout"}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"Connection: {type(e).__name__}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}

    if not html or len(html) < 100:
        return {"ok": False, "error": f"Empty response ({len(html or '')} chars)"}

    jsonld = extract_jsonld(html)

    # Pull page metadata (title, meta description, og:description, etc.) up
    # front. On Shopify, e-commerce platforms and many manufacturer sites the
    # meta description has solid product copy that trafilatura's body
    # extractor misses because it classifies the description container as
    # boilerplate. Combining metadata with body extraction gives the LLM more
    # to work with on these pages.
    meta_lines: list[str] = []
    try:
        meta = trafilatura.extract_metadata(html)
        if meta is not None:
            t = getattr(meta, "title", None)
            d = getattr(meta, "description", None)
            s = getattr(meta, "sitename", None)
            if t: meta_lines.append(f"PAGE TITLE: {t}")
            if d: meta_lines.append(f"META DESCRIPTION: {d}")
            if s: meta_lines.append(f"SITE: {s}")
    except Exception:
        pass

    # Two-pass body extraction. Precision mode is the right default for
    # product pages — it drops more boilerplate (nav, related products, FAQs)
    # and gives the LLM a cleaner signal. But on awkwardly-structured pages
    # it sometimes drops too much and we'd otherwise mark the row BLOCKED.
    # When that happens, retry with recall mode which keeps more text and
    # rescues the page. No extra network cost — same HTML, second parse.
    body = trafilatura.extract(
        html,
        include_links=False,
        include_tables=True,
        include_comments=False,
        output_format="markdown",
        favor_precision=True,
    )

    used_recall = False
    if not body or len(body) < 200:
        recall_body = trafilatura.extract(
            html,
            include_links=False,
            include_tables=True,
            include_comments=False,
            output_format="markdown",
            favor_recall=True,
        )
        if recall_body and len(recall_body) >= 200:
            body = recall_body
            used_recall = True

    # Combine metadata header + body. Metadata alone is often enough on
    # SPA pages where the body extraction returns nothing useful.
    parts = []
    if meta_lines:
        parts.append("\n".join(meta_lines))
    if body:
        parts.append(body)
    extracted = "\n\n".join(parts) if parts else ""

    # Threshold check: 150 chars total across metadata + body. Pages with
    # genuinely no content (failed JS, blank pages) won't clear this floor
    # even with metadata, but pages with just title + meta description (a
    # typical SPA with server-rendered meta tags) WILL pass through.
    if not extracted or len(extracted) < 150:
        return {
            "ok": False,
            "error": f"Extracted only {len(extracted)} chars after cleaning "
                     "(JS-only page, login wall, anti-bot, or page genuinely empty)",
        }

    if len(extracted) > cfg.max_chars:
        truncated = extracted[:cfg.max_chars]
        cut       = truncated.rfind("\n\n")
        if cut > int(cfg.max_chars * 0.7):
            truncated = truncated[:cut]
        extracted = truncated

    return {
        "ok":          True,
        "content":     extracted,
        "jsonld":      jsonld,
        "status":      status,
        "used_recall": used_recall,
    }


def fetch_pages_for_product(
    ids: IdentifierSet,
    cfg: SearchConfig,
) -> tuple[list[dict], str, str, list[dict]]:
    """
    Per-product pipeline: search → score top URLs → fetch.
    Returns (pages, status, query_used, fetch_errors).

    fetch_errors is a list of {url, error} entries for URLs that were tried
    and failed, plus a synthetic entry if all results were filtered out by
    the domain blocklist before any fetch was attempted. Used by the pipeline
    to surface BLOCKED diagnostics in the UI.
    """
    if ids.is_empty():
        return [], "not_found", "", []

    try:
        results, query = search_for_product(ids, cfg)
    except RateLimitError:
        return [], "rate_limited", "", []
    except BackendConfigError:
        # Misconfigured SerpAPI propagates so the pipeline shows a real error
        # instead of silently logging not_found for every row in the batch.
        raise
    except Exception:
        return [], "not_found", "", []

    if not results:
        return [], "not_found", query, []

    extra_blocked = set(cfg.blocked_domains or ())
    top_urls = select_top_urls(results, cfg.urls_per_sku, extra_blocked)

    fetch_errors: list[dict] = []

    if not top_urls:
        # Every search result was filtered out — most often by the user's
        # blocklist. Surface that explicitly so they understand why.
        sample = ", ".join((r.get("url") or "").split("/")[2] for r in results[:3])
        fetch_errors.append({
            "url": "",
            "error": f"All {len(results)} search results filtered by domain blocklist (saw: {sample})",
        })
        return [], "blocked", query, fetch_errors

    pages: list[dict] = []
    for result in top_urls:
        try:
            fetched = fetch_page(result["url"], cfg)
        except Exception as e:
            fetched = {"ok": False, "error": f"Unexpected: {type(e).__name__}"}

        if fetched.get("ok"):
            pages.append({
                "url":         result["url"],
                "content":     fetched["content"],
                "jsonld_hint": fetched.get("jsonld"),
            })
        else:
            fetch_errors.append({
                "url":   result["url"],
                "error": fetched.get("error", "Unknown"),
            })
        time.sleep(cfg.delay_between)

    if not pages:
        return [], "blocked", query, fetch_errors
    return pages, "ok", query, fetch_errors


def _normalise_for_match(s: str) -> str:
    """
    Strip non-alphanumeric characters and lowercase. Used to make identifier
    matching tolerant of formatting differences — so 'LDS-MAUI5' in the
    user's data matches against page content containing 'LDS-MAUI-5-W-SUB'
    (both normalise to 'ldsmaui5' / 'ldsmaui5wsub' where the first is a
    substring of the second).
    """
    return "".join(c for c in s.lower() if c.isalnum())


def content_validates_product(content: str, ids: IdentifierSet) -> bool:
    """Sanity check: does primary or fallback value appear in the fetched page?

    Uses normalised matching so 'LDS-MAUI5' matches 'LDS-MAUI-5-W-SUB' and
    'AAF7075' matches 'AAF7075-00'. Without this, legitimate product pages
    were being flagged as non-validating purely because of hyphen/space
    formatting differences.
    """
    codes = ids.codes()
    if not codes:
        return True
    if not content:
        return False
    norm_content = _normalise_for_match(content)
    return any(_normalise_for_match(c) in norm_content for c in codes if c)
