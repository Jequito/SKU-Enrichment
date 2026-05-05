"""
src/pipeline.py
Per-product processing with cascade search.

Each work item is (IdentifierSet, original_row). The pipeline:
  1. Search → fetch (primary, then fallback if primary fails at HTTP level)
  2. JSON-LD extraction from raw HTML as a trusted hint
  3. Identifier validation against fetched content
  4. LLM extraction (always runs, takes JSON-LD hint as foundation)
  5. NEW: extraction-triggered fallback — if LLM returned empty descriptions,
     try the fallback column as a fresh search before returning the row.
"""

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .search_client import (
    SearchConfig, IdentifierSet,
    fetch_pages_for_product, content_validates_product,
    RateLimitError, BackendConfigError,
)
from .extractors import LLMConfig, extract


@dataclass
class SKUResult:
    sku:            str
    status:         str   # success | review | not_found | blocked | rate_limited | error
    data:           dict
    sources:        list
    had_jsonld:     bool = False
    error_msg:      str  = ""
    debug_pages:    list = field(default_factory=list)
    jsonld_hint:    dict = field(default_factory=dict)
    query_used:     str  = ""
    primary_value:  str  = ""    # original Primary column value for this row
    fallback_value: str  = ""    # original Fallback column value for this row
    category_value: str  = ""    # original Category column value for this row (if mapped)


def _empty_row(original_row: dict, output_fields: list, flag: str) -> dict:
    row = dict(original_row)
    for f in output_fields:
        row.setdefault(f, "")
    row["review_flag"] = flag
    return row


def _has_descriptions(extracted: dict) -> bool:
    """True iff the LLM result contains a non-empty short or long description."""
    return (
        bool(str(extracted.get("short_description", "") or "").strip()) or
        bool(str(extracted.get("long_description",  "") or "").strip())
    )


def _build_debug_pages(pages: list, ids: IdentifierSet) -> list:
    """Materialise debug entries for the active set of pages."""
    return [{
        "url":           p["url"],
        "cleaned_chars": len(p["content"]),
        "cleaned_text":  p["content"],
        "validates":     content_validates_product(p["content"], ids),
    } for p in pages]


def process_product(
    ids:           IdentifierSet,
    original_row:  dict,
    output_fields: list,
    search_cfg:    SearchConfig,
    llm_cfg:       LLMConfig,
    debug:         bool = False,
) -> SKUResult:
    """Process a single product end-to-end. Stateless — safe under threading."""

    sku_label  = ids.display_label()
    primary_v  = (ids.primary  or "").strip()
    fallback_v = (ids.fallback or "").strip()
    category_v = (ids.category or "").strip()

    # 1. Search + fetch
    try:
        pages, fetch_status, query_used, fetch_errors = fetch_pages_for_product(ids, search_cfg)
    except RateLimitError as e:
        return SKUResult(sku=sku_label, status="rate_limited",
                         data=_empty_row(original_row, output_fields, "RATE_LIMITED"),
                         sources=[], error_msg=str(e),
                         primary_value=primary_v, fallback_value=fallback_v,
                         category_value=category_v)
    except Exception as e:
        return SKUResult(sku=sku_label, status="error",
                         data=_empty_row(original_row, output_fields, "ERROR"),
                         sources=[], error_msg=str(e),
                         primary_value=primary_v, fallback_value=fallback_v,
                         category_value=category_v)

    # Pre-compute attribution helpers used by both fallback hooks below.
    # `fallback_distinct` doesn't change once set; `used_primary_query`
    # reflects the FIRST successful query and gets stale if the post-fetch
    # fallback later promotes a different query — we recompute it
    # against the current `query_used` whenever we need a fresh read.
    fallback_distinct = (
        bool(fallback_v) and
        fallback_v.lower() != primary_v.lower()
    )
    used_primary_query = (
        bool(primary_v) and
        query_used.strip().strip('"').lower() == primary_v.lower()
    )

    # 1b. Post-fetch fallback retry. Triggers when the first attempt came back
    # blocked/not-found AND the winning query was the primary (which means the
    # search-stage relevance trigger didn't already fire the fallback). Catches
    # cases like: Google found a relevant URL, but it was on your blocklist /
    # returned 403 / was a JS-only page. Without this, those rows would just
    # die as BLOCKED even when the user has a perfectly good fallback column.
    if (fetch_status in ("blocked", "not_found")
            and used_primary_query
            and fallback_distinct):
        # Construct as fallback-in-the-fallback-slot so search_for_product's
        # existing logic naturally applies cfg.fallback_exact (not
        # cfg.primary_exact). Earlier this was put in the primary slot,
        # which silently flipped the exact-match setting — quoting fallback
        # terms even when the user had Fallback exact match OFF.
        fallback_only = IdentifierSet(primary="", fallback=fallback_v, category=category_v)
        try:
            pages_fb, status_fb, query_fb, errors_fb = fetch_pages_for_product(
                fallback_only, search_cfg
            )
            if status_fb == "ok" and pages_fb:
                # Fallback rescued the row — adopt its results, but combine
                # the failure errors from attempt 1 so the user can still
                # see what went wrong with the primary.
                pages         = pages_fb
                fetch_status  = "ok"
                query_used    = query_fb
                fetch_errors  = (fetch_errors or []) + (errors_fb or [])
            else:
                # Fallback also didn't yield pages — record both attempts'
                # failures in fetch_errors so the diagnostic panel shows the
                # full journey.
                fetch_errors = (fetch_errors or []) + [
                    {"url": "", "error":
                     f"Post-fetch fallback '{fallback_v}' also returned {status_fb}"}
                ] + (errors_fb or [])
        except RateLimitError as e:
            fetch_errors = (fetch_errors or []) + [
                {"url": "", "error": f"Fallback search rate-limited: {e}"}
            ]
        except BackendConfigError:
            raise
        except Exception as e:
            fetch_errors = (fetch_errors or []) + [
                {"url": "", "error": f"Fallback search error: {type(e).__name__}: {e}"}
            ]

    if fetch_status in ("not_found", "blocked", "rate_limited"):
        # Build a readable summary of why fetches failed so the user sees
        # actual reasons (HTTP 403, timeout, etc.) instead of an opaque
        # BLOCKED flag. Limited to first 3 attempts to keep the panel tidy.
        block_msg = ""
        if fetch_errors:
            parts = []
            for fe in fetch_errors[:3]:
                if fe.get("url"):
                    parts.append(f"{fe['url']} → {fe['error']}")
                else:
                    parts.append(fe["error"])
            block_msg = " | ".join(parts)
        return SKUResult(sku=sku_label, status=fetch_status,
                         data=_empty_row(original_row, output_fields, fetch_status.upper()),
                         sources=[], error_msg=block_msg, query_used=query_used,
                         primary_value=primary_v, fallback_value=fallback_v,
                         category_value=category_v)

    sources = [p["url"] for p in pages]

    # 2. Use the first available JSON-LD hint as trusted metadata
    jsonld_hint = None
    for page in pages:
        candidate = page.get("jsonld_hint")
        if candidate and candidate.get("product_name"):
            jsonld_hint = candidate
            break

    # 3. Validate page content references at least one identifier code.
    #    If no page mentions our codes, the search may have surfaced the
    #    wrong product entirely — flag for review even if extraction succeeds.
    any_validates = any(
        content_validates_product(p["content"], ids) for p in pages
    )

    # 4. Capture debug info if requested
    debug_pages = _build_debug_pages(pages, ids) if debug else []

    # 5. LLM extraction (primary attempt)
    try:
        extracted = extract(ids, pages, output_fields, llm_cfg,
                            jsonld_hint=jsonld_hint,
                            max_chars=search_cfg.max_chars)
    except Exception as e:
        return SKUResult(sku=sku_label, status="error",
                         data=_empty_row(original_row, output_fields, "ERROR"),
                         sources=sources, error_msg=str(e),
                         debug_pages=debug_pages, query_used=query_used,
                         primary_value=primary_v, fallback_value=fallback_v,
                         category_value=category_v)

    # 5b. Extraction-triggered fallback. The HTTP-level fallback above only
    # fires when the page fetch fails outright. But Google can return a
    # HTTP-200 page that's content-thin (replacement-parts listing, sparse
    # PDF spec sheet, manufacturer's discontinued-product stub) — those
    # finish "successful" with empty short/long descriptions and the user
    # never gets the rich page that the fallback column might have led to.
    #
    # Trigger conditions:
    #   - LLM result has no descriptions, AND
    #   - the winning query came from the primary (re-read fresh in case
    #     the post-fetch fallback already promoted query_used), AND
    #   - a distinct fallback value exists, AND
    #   - the LLM didn't error out (otherwise the row is already broken
    #     and a second extraction won't help diagnose what went wrong).
    flag_now = str(extracted.get("review_flag", "") or "").upper()
    current_used_primary = (
        bool(primary_v) and
        query_used.strip().strip('"').lower() == primary_v.lower()
    )

    if (not _has_descriptions(extracted)
            and "ERROR" not in flag_now
            and current_used_primary
            and fallback_distinct):
        # Same slot fix as the post-fetch fallback: put fallback_v in the
        # FALLBACK slot so search_for_product applies cfg.fallback_exact
        # rather than cfg.primary_exact. Without this the user's "Fallback
        # exact match: OFF" setting was being silently ignored every time
        # this fallback level fired.
        fallback_only = IdentifierSet(primary="", fallback=fallback_v, category=category_v)
        try:
            pages_fb, status_fb, query_fb, errors_fb = fetch_pages_for_product(
                fallback_only, search_cfg
            )
        except RateLimitError:
            pages_fb, status_fb, query_fb, errors_fb = [], "rate_limited", "", []
        except BackendConfigError:
            raise
        except Exception:
            pages_fb, status_fb, query_fb, errors_fb = [], "error", "", []

        if status_fb == "ok" and pages_fb:
            # Pull a fresh JSON-LD hint from the fallback pages — we want
            # the LLM to have the strongest possible structured-data
            # foundation for this second attempt.
            jsonld_hint_fb = None
            for page in pages_fb:
                candidate = page.get("jsonld_hint")
                if candidate and candidate.get("product_name"):
                    jsonld_hint_fb = candidate
                    break

            try:
                extracted_fb = extract(fallback_only, pages_fb, output_fields, llm_cfg,
                                       jsonld_hint=jsonld_hint_fb,
                                       max_chars=search_cfg.max_chars)
            except Exception:
                extracted_fb = {}

            # Only adopt the fallback result when it actually rescues the
            # missing descriptions. If it ALSO came back description-less,
            # the original primary result (which at least has product_name,
            # specs, etc.) is still the better answer.
            if _has_descriptions(extracted_fb):
                extracted     = extracted_fb
                pages         = pages_fb
                sources       = [p["url"] for p in pages]
                query_used    = query_fb
                jsonld_hint   = jsonld_hint_fb
                # Re-validate against the ORIGINAL identifier set — we still
                # want the validation flag to reflect whether the new pages
                # mention the user's actual codes, not the standalone
                # fallback term we used for searching.
                any_validates = any(
                    content_validates_product(p["content"], ids) for p in pages
                )
                if debug:
                    debug_pages = _build_debug_pages(pages, ids)

    # 6. If validation failed, override the flag to REVIEW_NEEDED — content
    # may be about a different product even if extraction looked clean.
    if not any_validates:
        existing = str(extracted.get("review_flag", "") or "").upper()
        if "ERROR" not in existing:
            extracted["review_flag"] = "REVIEW_NEEDED"

    # Capture the LLM-side error (if any) before stripping it from the row so
    # it doesn't leak into CSV/Excel exports. The extract() function catches
    # provider exceptions internally and returns {review_flag: "ERROR",
    # _error: "..."} — the previous version threw away that error message
    # AND counted these rows as success because "REVIEW" wasn't in "ERROR".
    llm_error = str(extracted.pop("_error", "") or "")

    flag = str(extracted.get("review_flag", "") or "").upper()
    if "ERROR" in flag:
        status = "error"
    elif "REVIEW" in flag:
        status = "review"
    else:
        status = "success"

    enriched = {**original_row, **extracted}

    return SKUResult(
        sku=sku_label, status=status,
        data=enriched, sources=sources,
        had_jsonld=(jsonld_hint is not None),
        error_msg=llm_error,
        debug_pages=debug_pages,
        jsonld_hint=jsonld_hint or {},
        query_used=query_used,
        primary_value=primary_v, fallback_value=fallback_v,
        category_value=category_v,
    )


def process_batch(
    items:         list,
    output_fields: list,
    search_cfg:    SearchConfig,
    llm_cfg:       LLMConfig,
    max_workers:   int  = 5,
    debug:         bool = False,
) -> list:
    """
    Process a list of (IdentifierSet, original_row) tuples concurrently.
    Returns SKUResult objects in completion order.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_product, ids, row, output_fields,
                            search_cfg, llm_cfg, debug): ids
            for ids, row in items
        }
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as e:
                ids = future_map[future]
                results.append(SKUResult(
                    sku=ids.display_label(), status="error",
                    data={"review_flag": "ERROR", "_error": str(e)},
                    sources=[],
                    primary_value=(ids.primary  or "").strip(),
                    fallback_value=(ids.fallback or "").strip(),
                    category_value=(ids.category or "").strip(),
                ))
    return results
