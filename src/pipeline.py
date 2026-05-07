"""
src/pipeline.py
Per-product processing.

Each work item is (IdentifierSet, original_row). The pipeline:
  1. Search via SerpAPI on the Primary column (or Fallback if Primary
     returned zero relevant results), then fetch the top URLs in SERP
     order, skipping blocked domains and PDFs.
  2. JSON-LD extraction from raw HTML as a trusted hint for the LLM.
  3. LLM extraction.
  4. Extraction-triggered fallback — if the LLM produced no descriptions
     AND a Fallback column was mapped, re-run the search against the
     fallback term and adopt the result iff it actually has descriptions.
  5. Low-confidence flag — if neither set of fetched pages overlaps with
     the searched identifier (token-majority test, tolerant of multi-word
     product names), set review_flag=LOW_CONFIDENCE so the user knows the
     LLM extracted data but the source might not be the right product.
"""

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .search_client import (
    SearchConfig, IdentifierSet,
    fetch_pages_for_product, content_overlaps_identifier,
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
    """Materialise debug entries for the active set of pages.

    `overlaps_identifier` records the LOW_CONFIDENCE check's per-page
    verdict so the debug panel can show which fetched pages did or didn't
    appear to be about the searched product.
    """
    return [{
        "url":                  p["url"],
        "cleaned_chars":        len(p["content"]),
        "cleaned_text":         p["content"],
        "overlaps_identifier":  content_overlaps_identifier(p["content"], ids),
    } for p in pages]


def process_product(
    ids:           IdentifierSet,
    original_row:  dict,
    output_fields: list,
    search_cfg:    SearchConfig,
    llm_cfg:       LLMConfig,
    debug:         bool = False,
    cancel_event:  "threading.Event | None" = None,
) -> SKUResult:
    """Process a single product end-to-end. Stateless — safe under threading.

    cancel_event: shared across the worker pool. When set (typically by a
    peer worker that just hit a SerpAPI 429), this function returns a
    rate_limited result immediately without making any outbound calls.
    Lets process_batch halt the rest of the pool cleanly instead of
    letting in-flight workers crash into the same 429.
    """

    sku_label  = ids.display_label()
    primary_v  = (ids.primary  or "").strip()
    fallback_v = (ids.fallback or "").strip()
    category_v = (ids.category or "").strip()

    # Fast-exit if a peer thread has already tripped the rate limit.
    if cancel_event is not None and cancel_event.is_set():
        return SKUResult(sku=sku_label, status="rate_limited",
                         data=_empty_row(original_row, output_fields, "RATE_LIMITED"),
                         sources=[],
                         error_msg="Aborted: peer thread hit SerpAPI rate limit",
                         primary_value=primary_v, fallback_value=fallback_v,
                         category_value=category_v)

    # 1. Search + fetch
    try:
        pages, fetch_status, query_used, fetch_errors = fetch_pages_for_product(
            ids, search_cfg, cancel_event=cancel_event,
        )
    except RateLimitError as e:
        if cancel_event is not None:
            cancel_event.set()
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

    if fetch_status == "rate_limited":
        if cancel_event is not None:
            cancel_event.set()
        return SKUResult(sku=sku_label, status="rate_limited",
                         data=_empty_row(original_row, output_fields, "RATE_LIMITED"),
                         sources=[], error_msg="SerpAPI rate limited",
                         query_used=query_used,
                         primary_value=primary_v, fallback_value=fallback_v,
                         category_value=category_v)

    if fetch_status in ("not_found", "blocked"):
        # Build a readable summary of why every fetch attempt failed so the
        # user sees actual reasons (HTTP 403, timeout, blocklisted, etc.)
        # instead of an opaque flag. With the walk-until-N-successes
        # iterator a single SKU on a heavily blocked site can produce many
        # entries; we surface all of them rather than truncating, so the
        # diagnostic panel reflects the full attempt list.
        block_msg = ""
        if fetch_errors:
            parts = []
            for fe in fetch_errors:
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

    # 3. Capture debug info if requested
    debug_pages = _build_debug_pages(pages, ids) if debug else []

    # 4. LLM extraction (primary attempt)
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

    # 5. Extraction-triggered fallback. If the LLM came back with no
    # descriptions and we have a distinct Fallback column value to try,
    # re-run the search using the fallback term and adopt that result iff
    # it actually populates descriptions. This is the ONLY pipeline-level
    # fallback trigger — fetch failures (blocked/not_found) used to also
    # trigger a fallback search, but that's been removed to match the
    # "secondary if primary returns no extracted results" rule.
    fallback_distinct = (
        bool(fallback_v) and fallback_v.lower() != primary_v.lower()
    )
    flag_now = str(extracted.get("review_flag", "") or "").upper()
    used_primary_query = (
        bool(primary_v) and
        query_used.strip().strip('"').lower() == primary_v.lower()
    )

    if (not _has_descriptions(extracted)
            and "ERROR" not in flag_now
            and used_primary_query
            and fallback_distinct):
        # Put fallback_v in the FALLBACK slot so search_for_product
        # applies cfg.fallback_exact (not cfg.primary_exact).
        fallback_only = IdentifierSet(primary="", fallback=fallback_v, category=category_v)
        try:
            pages_fb, status_fb, query_fb, errors_fb = fetch_pages_for_product(
                fallback_only, search_cfg, cancel_event=cancel_event,
            )
        except RateLimitError:
            if cancel_event is not None:
                cancel_event.set()
            pages_fb, status_fb, query_fb, errors_fb = [], "rate_limited", "", []
        except BackendConfigError:
            raise
        except Exception:
            pages_fb, status_fb, query_fb, errors_fb = [], "error", "", []

        if status_fb == "ok" and pages_fb:
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

            # Only adopt the fallback result when it actually rescues
            # missing descriptions. If it ALSO came back description-less
            # the original primary result (which at least has product_name,
            # specs, etc.) is still the better answer.
            if _has_descriptions(extracted_fb):
                extracted   = extracted_fb
                pages       = pages_fb
                sources     = [p["url"] for p in pages]
                query_used  = query_fb
                jsonld_hint = jsonld_hint_fb
                if debug:
                    debug_pages = _build_debug_pages(pages, ids)

    # 6. LOW_CONFIDENCE flag. This is a soft warning, not a gate: data
    # has been extracted and is being returned to the user, but no fetched
    # page appears to be about the searched product based on token overlap.
    # Useful for catching wrong-product matches (numeric SKU collisions,
    # SEO spam pages that surface for a model number but actually sell a
    # different product). Does NOT downgrade rows the LLM already flagged
    # as REVIEW_NEEDED or ERROR — those flags are preserved.
    any_overlaps = any(
        content_overlaps_identifier(p["content"], ids) for p in pages
    )
    if not any_overlaps:
        existing = str(extracted.get("review_flag", "") or "").upper()
        # Don't overwrite ERROR/REVIEW_NEEDED — LOW_CONFIDENCE is a softer
        # signal than either, and stacking them would bury the stronger
        # warning the LLM already raised.
        if not any(tag in existing for tag in ("ERROR", "REVIEW")):
            extracted["review_flag"] = "LOW_CONFIDENCE"

    # Capture LLM-side error before stripping it so it doesn't leak into
    # the CSV/Excel exports.
    llm_error = str(extracted.pop("_error", "") or "")

    # Populate all_source_urls — every URL whose content went into the
    # LLM, comma-separated. Distinct from the LLM's `source_url` field
    # (which is the model's pick of "most authoritative single source").
    # Set unconditionally so the column is always present in the output
    # when the user selects it. Reflects the FINAL pages list, so if the
    # extraction-triggered fallback fired and replaced the pages, this
    # is the fallback's URLs not the primary's.
    extracted["all_source_urls"] = ", ".join(sources) if sources else ""

    flag = str(extracted.get("review_flag", "") or "").upper()
    if "ERROR" in flag:
        status = "error"
    elif "REVIEW" in flag or "LOW_CONFIDENCE" in flag:
        # LOW_CONFIDENCE rides the same status bucket as REVIEW_NEEDED so
        # the UI's existing review counter and filtering still picks it
        # up. They're distinguished by the review_flag value itself.
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

    Threading + rate limits: a threading.Event is shared across all
    workers. The first worker to hit a SerpAPI 429 sets the event, which
    causes:
      * all queued-but-not-yet-started futures to be cancelled outright
      * all currently-running workers to abort at their next check-point
        (entry of process_product, or before any outbound network call
        inside fetch_pages_for_product)
    Without this, when one worker out of N hits a 429 the other N-1 hit
    the same rate-limited endpoint at the same instant and end up logged
    as ERROR/BLOCKED instead of pausing cleanly for resume.
    """
    cancel_event = threading.Event()

    results: list = []
    completed_futures: set = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_product, ids, row, output_fields,
                            search_cfg, llm_cfg, debug, cancel_event): (ids, row)
            for ids, row in items
        }
        for future in as_completed(future_map):
            ids, row = future_map[future]
            completed_futures.add(future)
            try:
                result = future.result()
            except Exception as e:
                result = SKUResult(
                    sku=ids.display_label(), status="error",
                    data={"review_flag": "ERROR", "_error": str(e)},
                    sources=[],
                    primary_value=(ids.primary  or "").strip(),
                    fallback_value=(ids.fallback or "").strip(),
                    category_value=(ids.category or "").strip(),
                )
            results.append(result)

            if result.status == "rate_limited" and not cancel_event.is_set():
                cancel_event.set()
            if cancel_event.is_set():
                for f in future_map:
                    f.cancel()

    # Workers cancelled while still queued never produce a result via
    # as_completed. Synthesise rate_limited stubs for them so the UI can
    # put them back in the work queue when the user resumes — and so we
    # never silently drop input rows.
    if cancel_event.is_set():
        for future, (ids, row) in future_map.items():
            if future in completed_futures:
                continue
            results.append(SKUResult(
                sku=ids.display_label(), status="rate_limited",
                data=_empty_row(row, output_fields, "RATE_LIMITED"),
                sources=[],
                error_msg="Cancelled: peer thread hit SerpAPI rate limit",
                primary_value=(ids.primary  or "").strip(),
                fallback_value=(ids.fallback or "").strip(),
                category_value=(ids.category or "").strip(),
            ))

    return results
