"""
src/pipeline.py
Per-product processing with IdentifierSet cascade search.

Each work item is (IdentifierSet, original_row). The pipeline:
  1. Cascade search via DuckDuckGo
  2. Fetch top URLs via httpx + trafilatura
  3. Extract JSON-LD from raw HTML as a trusted hint
  4. Validate fetched content references at least one identifier code
  5. LLM extraction (always runs, takes JSON-LD hint as foundation)
"""

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .search_client import (
    SearchConfig, IdentifierSet,
    fetch_pages_for_product, content_validates_product,
    RateLimitError,
)
from .extractors import LLMConfig, extract


@dataclass
class SKUResult:
    sku:          str
    status:       str   # success | review | not_found | blocked | rate_limited | error
    data:         dict
    sources:      list
    had_jsonld:   bool = False
    error_msg:    str  = ""
    debug_pages:  list = field(default_factory=list)
    jsonld_hint:  dict = field(default_factory=dict)
    query_used:   str  = ""


def _empty_row(original_row: dict, output_fields: list, flag: str) -> dict:
    row = dict(original_row)
    for f in output_fields:
        row.setdefault(f, "")
    row["review_flag"] = flag
    return row


def process_product(
    ids:           IdentifierSet,
    original_row:  dict,
    output_fields: list,
    search_cfg:    SearchConfig,
    llm_cfg:       LLMConfig,
    debug:         bool = False,
) -> SKUResult:
    """Process a single product end-to-end. Stateless — safe under threading."""

    sku_label = ids.display_label()

    # 1. Search + fetch
    try:
        pages, fetch_status, query_used = fetch_pages_for_product(ids, search_cfg)
    except RateLimitError as e:
        return SKUResult(sku=sku_label, status="rate_limited",
                         data=_empty_row(original_row, output_fields, "RATE_LIMITED"),
                         sources=[], error_msg=str(e))
    except Exception as e:
        return SKUResult(sku=sku_label, status="error",
                         data=_empty_row(original_row, output_fields, "ERROR"),
                         sources=[], error_msg=str(e))

    if fetch_status in ("not_found", "blocked", "rate_limited"):
        return SKUResult(sku=sku_label, status=fetch_status,
                         data=_empty_row(original_row, output_fields, fetch_status.upper()),
                         sources=[], query_used=query_used)

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
    debug_pages = []
    if debug:
        for page in pages:
            debug_pages.append({
                "url":           page["url"],
                "cleaned_chars": len(page["content"]),
                "cleaned_text":  page["content"],
                "validates":     content_validates_product(page["content"], ids),
            })

    # 5. LLM extraction
    try:
        extracted = extract(ids, pages, output_fields, llm_cfg,
                            jsonld_hint=jsonld_hint,
                            max_chars=search_cfg.max_chars)
    except Exception as e:
        return SKUResult(sku=sku_label, status="error",
                         data=_empty_row(original_row, output_fields, "ERROR"),
                         sources=sources, error_msg=str(e),
                         debug_pages=debug_pages, query_used=query_used)

    # If validation failed, override the flag to REVIEW_NEEDED — content may
    # be about a different product even if extraction looked clean.
    if not any_validates:
        existing = str(extracted.get("review_flag", "") or "").upper()
        if "ERROR" not in existing:
            extracted["review_flag"] = "REVIEW_NEEDED"

    enriched = {**original_row, **extracted}
    flag     = str(extracted.get("review_flag", "") or "").upper()
    status   = "review" if "REVIEW" in flag else "success"

    return SKUResult(
        sku=sku_label, status=status,
        data=enriched, sources=sources,
        had_jsonld=(jsonld_hint is not None),
        debug_pages=debug_pages,
        jsonld_hint=jsonld_hint or {},
        query_used=query_used,
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
                ))
    return results
