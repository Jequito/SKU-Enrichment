"""
src/search_client.py
SerpAPI Google search + direct httpx fetch + trafilatura content extraction.

Strategy: per-row Google search on a user-mapped Primary column, with optional
exact-match (quoted) phrasing. If that returns no relevant hits, retry on a
user-mapped Fallback column. Both column mappings and both quoting decisions
are user-configured in the sidebar.

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

    primary  is searched first.
    fallback is searched only if the primary search returns no relevant hits.
    Both are also passed to the LLM as match anchors and used for content
    validation.
    """
    primary:  str = ""
    fallback: str = ""
    # Optional per-row category hint. When non-empty it's appended to the
    # search query as a loose refinement (never quoted, even when
    # primary_exact / fallback_exact are on) — e.g. `"1204086639" sheet music`.
    # Also passed to the LLM prompt as context so it can reject pages that
    # describe a product in a totally different industry. Useful for
    # ambiguous numeric SKUs that get reused across industries (the user's
    # debug log shows `105666` matching Thermo Fisher reagents AND Danfoss
    # switch parts — neither was the intended music product).
    category: str = ""

    def codes(self) -> list[str]:
        """All non-empty identifier values, deduped.

        Category is deliberately excluded — it's a search refinement, not an
        identifier, so we don't want it counted toward content validation.
        """
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
    # Exact-match toggles. When True the corresponding term is wrapped in
    # double quotes before being sent to Google ("ABC123"), which forces a
    # verbatim match. Turn OFF for codes whose formatting varies across sites
    # — e.g. an SKU stored as "WVE-T60S" in the user's data but rendered as
    # "WVE T60 S" or "WVET60S" on retailer pages. Without quotes Google is
    # free to do its usual loosening (stemming, ignored hyphens, synonyms).
    primary_exact:      bool  = True
    fallback_exact:     bool  = True
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


def _tokenise_identifier(identifier: str) -> list[str]:
    """Split an identifier into alphanumeric word tokens, lowercased.

    Splits on whitespace AND punctuation, so 'LD Systems MAUI 5' and
    'LD-Systems-MAUI-5' both produce ['ld','systems','maui','5'].
    Single-character noise tokens are dropped unless they're digits —
    a stray 'a' or 'e' isn't strong enough evidence of a match, but a
    single-digit '5' inside a model name carries real signal.
    """
    if not identifier:
        return []
    buf = [(ch if ch.isalnum() else " ") for ch in identifier.lower()]
    raw = "".join(buf).split()
    seen: set = set()
    out: list[str] = []
    for t in raw:
        if (len(t) >= 2 or t.isdigit()) and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _has_relevant_match(results: list[dict], identifier: str) -> bool:
    """
    Did Google actually find what we asked for, or just adjacent junk?

    Tokenises the identifier into alphanumeric words and checks how many
    appear in the URL + title + snippet of each result. A result is
    considered a match when at least 50% of the identifier tokens appear
    somewhere in its haystack — this tolerates Google's snippet truncation
    (which routinely cuts mid-phrase with `...`) while still rejecting
    results that share only the brand or category. Single-token
    identifiers still need that one token, so the bar stays meaningful.

    Used by search_for_product to decide whether to fire the fallback
    search. With a "secondary on no extracted results" rule this is the
    only place fallback can trigger from the search stage — so getting
    the threshold right matters: too lax, fallback never runs; too
    strict, fallback runs constantly and wastes SerpAPI quota.
    """
    tokens = _tokenise_identifier(identifier)
    if not tokens:
        return True   # nothing to check against — assume relevant

    needed = max(1, (len(tokens) + 1) // 2)

    # Punctuation-stripped concatenation as a bonus path — catches the
    # case where 'WVE-T60S' is rendered as 'WVET60S' in a page title and
    # ended up tokenised differently from the haystack version.
    concat = "".join(tokens)

    for r in results:
        haystack = " ".join([
            r.get("url", "") or "",
            r.get("title", "") or "",
            r.get("snippet", "") or "",
        ]).lower()
        if not haystack.strip():
            continue
        if len(concat) >= 4 and concat in _normalise_for_match(haystack):
            return True
        hits = sum(1 for t in tokens if t in haystack)
        if hits >= needed:
            return True
    return False


def _format_query(value: str, exact: bool, category: str = "") -> str:
    """Build a Google query for one identifier value.

    Quoting: applied to `value` only when exact=True. The category, if
    provided, is always appended *unquoted* — Google should match category
    terms loosely so they refine without over-constraining the result set.
    Examples:
        ('ABC123',     True,  '')           -> '"ABC123"'
        ('ABC123',     False, '')           -> 'ABC123'
        ('1204086639', True,  'sheet music')-> '"1204086639" sheet music'
        ('WVE-T60S',   False, 'cooktop')    -> 'WVE-T60S cooktop'
    """
    v = (value or "").strip()
    if not v:
        return ""
    base = f'"{v}"' if exact else v
    cat  = (category or "").strip()
    return f"{base} {cat}" if cat else base


def search_for_product(ids: IdentifierSet, cfg: SearchConfig) -> tuple[list[dict], str]:
    """
    Two-stage search with optional per-stage exact-match (quoted) phrasing
    and a relevance trigger:

      1. Primary search via SerpAPI. The query is quoted iff cfg.primary_exact
         is True. If results are returned AND at least one of them mentions
         the primary identifier in URL/title/snippet (after hyphen-insensitive
         normalisation), use them.

      2. Fallback search — runs when stage 1 returned zero results OR
         when stage 1 returned results that don't actually contain the
         primary identifier. The query is quoted iff cfg.fallback_exact
         is True.

    Returns (results, query_used). If both stages fail the relevance bar but
    the primary returned *something*, those weak results are returned anyway
    so the row gets at least one fetch attempt.
    """
    primary_val  = (ids.primary  or "").strip()
    fallback_val = (ids.fallback or "").strip()
    category_val = (ids.category or "").strip()

    primary_results: list[dict] = []
    primary_query   = ""

    # Stage 1: primary
    if primary_val:
        primary_query   = _format_query(primary_val, cfg.primary_exact, category_val)
        primary_results = search(primary_query, cfg)
        if primary_results and _has_relevant_match(primary_results, primary_val):
            return primary_results, primary_query

    # Stage 2: fallback — runs on either zero results or weak/irrelevant results
    if fallback_val and fallback_val.lower() != primary_val.lower():
        fallback_query   = _format_query(fallback_val, cfg.fallback_exact, category_val)
        fallback_results = search(fallback_query, cfg)
        if fallback_results:
            return fallback_results, fallback_query

    # Fallback didn't help either. If primary returned weak-but-nonempty
    # results, return them — better to attempt a fetch on a borderline page
    # than mark the row NOT_FOUND when Google did surface something.
    if primary_results:
        return primary_results, primary_query
    return [], ""


def _is_blocked_domain(url: str, blocked: set) -> bool:
    """True if the URL's domain is blocked (exact match or a subdomain).

    Exact-or-subdomain matching only — substring matching used to block
    'rolex.com', 'fedex.com', 'swish.com' purely because the blocklist
    contains 'x.com' and 'wish.com'. Now 'shop.x.com' is blocked (a
    real x.com subdomain), but 'rolex.com' is fine.
    """
    domain = url.split("/")[2] if url.count("/") >= 2 else url
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    for bad in blocked:
        bad_norm = (bad or "").lower().lstrip(".")
        if not bad_norm:
            continue
        if domain == bad_norm or domain.endswith("." + bad_norm):
            return True
    return False


def select_top_urls(results: list[dict], n: int, extra_blocked: set | None = None) -> list[dict]:
    """Walk the SERP top-to-bottom and return the first N results that
    aren't on the blocklist and aren't PDFs (trafilatura can't extract
    those — they'd just fail and burn a fetch).

    Order is exactly Google's order. No re-ranking, no scoring rubric.
    If you want different URLs, add to the blocklist or raise n.
    """
    blocked = LOW_VALUE_DOMAINS | (extra_blocked or set())
    out: list[dict] = []
    for r in results:
        if len(out) >= n:
            break
        url = (r.get("url") or "").strip()
        if not url:
            continue
        if url.lower().endswith(".pdf"):
            continue
        if _is_blocked_domain(url, blocked):
            continue
        out.append(r)
    return out


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
    # Threshold raised from 200 → 800. Many product pages return 200–500 chars
    # of specs, headings and footer text in precision mode while the actual
    # description paragraph is classified as boilerplate and dropped. The old
    # 200-char threshold meant those pages bypassed recall entirely, leaving
    # the LLM with specs but no description. 800 is empirical — most genuine
    # product pages clear it once descriptions are included; spec-only stubs
    # don't, which is exactly the case where recall mode helps.
    if not body or len(body) < 800:
        recall_body = trafilatura.extract(
            html,
            include_links=False,
            include_tables=True,
            include_comments=False,
            output_format="markdown",
            favor_recall=True,
        )
        # Defensive: only adopt recall if it's strictly longer than precision.
        # In rare cases recall returns less (different boilerplate scoring),
        # in which case precision's cleaner output is still preferable.
        if recall_body and len(recall_body) > len(body or ""):
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
    cancel_event=None,
) -> tuple[list[dict], str, str, list[dict]]:
    """
    Per-product pipeline: search → top-to-bottom URL pick → fetch.
    Returns (pages, status, query_used, fetch_errors).

    fetch_errors is a list of {url, error} entries for URLs that were tried
    and failed, plus a synthetic entry if all results were filtered out by
    the domain blocklist before any fetch was attempted.

    cancel_event: optional threading.Event. When set, this function aborts
    at the next check-point and returns ([], "rate_limited", "", []) without
    making any further network calls. Used by process_batch to halt
    in-flight workers as soon as one hits a SerpAPI 429.
    """
    if ids.is_empty():
        return [], "not_found", "", []

    # Pre-search cancellation check.
    if cancel_event is not None and cancel_event.is_set():
        return [], "rate_limited", "", []

    try:
        results, query = search_for_product(ids, cfg)
    except RateLimitError:
        if cancel_event is not None:
            cancel_event.set()
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
        # Per-fetch cancellation check — peer thread may have hit a 429
        # since we entered the loop.
        if cancel_event is not None and cancel_event.is_set():
            return [], "rate_limited", query, fetch_errors

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
    """Strip non-alphanumeric characters and lowercase.

    Used by the bonus-path concatenation match in _has_relevant_match and
    by content_overlaps_identifier for the punctuation-tolerant compact-SKU
    case (e.g. 'LDS-MAUI5' should match 'LDS-MAUI-5-W-SUB').
    """
    return "".join(c for c in s.lower() if c.isalnum())


def content_overlaps_identifier(content: str, ids: IdentifierSet) -> bool:
    """Does the fetched content plausibly correspond to the searched identifier?

    Used by the pipeline to set a LOW_CONFIDENCE flag — NOT to gate
    extraction or trigger fallbacks. Returns True under either of:

      1. Concatenation match — punctuation-stripped identifier appears as
         a substring of the punctuation-stripped content. This is the
         right tool for compact SKU codes like 'LDS-MAUI5' that should
         match against 'LDS-MAUI-5-W-SUB' on a retailer page.
      2. Token-majority match — at least half the identifier's word
         tokens appear in the lowercased content. This is what handles
         multi-word product names like 'LD Systems MAUI 5 GO E W SET EU'
         whose concatenated form almost never appears verbatim on real
         pages even when the page is clearly about the right product.

    Either path counts as overlap. Both paths are conservative: a page
    that mentions only the brand without the model code won't pass either.
    """
    codes = ids.codes()
    if not codes:
        return True
    if not content:
        return False

    norm_content  = _normalise_for_match(content)
    lower_content = content.lower()

    for code in codes:
        if not code:
            continue
        norm_code = _normalise_for_match(code)
        if len(norm_code) >= 4 and norm_code in norm_content:
            return True
        tokens = _tokenise_identifier(code)
        if not tokens:
            continue
        needed = max(1, (len(tokens) + 1) // 2)
        hits = sum(1 for t in tokens if t in lower_content)
        if hits >= needed:
            return True
    return False
