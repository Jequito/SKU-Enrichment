"""
src/pipeline.py
Per-SKU processing with:
  - JSON-LD metadata hint passed to LLM (never bypasses it)
  - Concurrent batch processing via ThreadPoolExecutor
"""

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .jina_client     import JinaConfig, fetch_pages_for_sku, RateLimitError
from .extractors      import LLMConfig, extract
from .content_cleaner import extract_jsonld


@dataclass
class SKUResult:
    sku:          str
    status:       str     # success | review | not_found | blocked | rate_limited | error
    data:         dict
    sources:      list
    had_jsonld:   bool = False
    error_msg:    str  = ""
    debug_pages:  list = field(default_factory=list)
    jsonld_hint:  dict = field(default_factory=dict)


def _empty_row(original_row: dict, output_fields: list, flag: str) -> dict:
    row = dict(original_row)
    for f in output_fields:
        row.setdefault(f, "")
    row["review_flag"] = flag
    return row


def process_sku(
    sku:            str,
    original_row:   dict,
    output_fields:  list,
    jina_cfg:       JinaConfig,
    llm_cfg:        LLMConfig,
    secondary_term: str  = "",
    debug:          bool = False,
) -> SKUResult:
    """
    Process a single SKU end-to-end. Fully stateless — safe to call concurrently.

    secondary_term is an optional value from a secondary column (e.g. product
    name or brand). It is used as a standalone fallback search query if the
    primary SKU search returns irrelevant results — the two are never combined.

    Flow:
      1. Fetch pages via Jina (search + content-cleaned fetch)
      2. Try to extract JSON-LD from page content
      3. Pass JSON-LD (if found) as a trusted metadata hint to the LLM prompt
      4. LLM always runs — it uses the hint as a foundation and fills gaps
         from the full page content

    The LLM is NEVER bypassed. JSON-LD improves accuracy and reduces token
    usage by giving the model pre-validated structured data to build on.
    """

    # 1. Fetch
    try:
        pages, fetch_status = fetch_pages_for_sku(sku, jina_cfg, secondary_term=secondary_term)
    except RateLimitError as e:
        return SKUResult(sku=sku, status="rate_limited",
                         data=_empty_row(original_row, output_fields, "RATE_LIMITED"),
                         sources=[], error_msg=str(e))
    except Exception as e:
        return SKUResult(sku=sku, status="error",
                         data=_empty_row(original_row, output_fields, "ERROR"),
                         sources=[], error_msg=str(e))

    if fetch_status in ("not_found", "blocked", "rate_limited"):
        return SKUResult(sku=sku, status=fetch_status,
                         data=_empty_row(original_row, output_fields, fetch_status.upper()),
                         sources=[])

    sources = [p["url"] for p in pages]

    # 2. Extract JSON-LD hint from any page that has it
    jsonld_hint = None
    for page in pages:
        candidate = extract_jsonld(page["content"])
        if candidate and candidate.get("product_name"):
            jsonld_hint = candidate
            break

    # 3. Capture debug info when enabled (avoids memory cost on large runs)
    debug_pages = []
    if debug:
        for page in pages:
            debug_pages.append({
                "url":           page["url"],
                "cleaned_chars": len(page["content"]),
                "cleaned_text":  page["content"],
            })

    # 4. LLM extraction — always runs, JSON-LD injected as trusted metadata if available
    try:
        extracted = extract(sku, pages, output_fields, llm_cfg,
                            jsonld_hint=jsonld_hint,
                            max_chars=jina_cfg.max_chars)
    except Exception as e:
        return SKUResult(sku=sku, status="error",
                         data=_empty_row(original_row, output_fields, "ERROR"),
                         sources=sources, error_msg=str(e),
                         debug_pages=debug_pages)

    enriched = {**original_row, **extracted}
    flag     = str(extracted.get("review_flag", "") or "").upper()
    status   = "review" if "REVIEW" in flag else "success"

    return SKUResult(
        sku=sku, status=status,
        data=enriched, sources=sources,
        had_jsonld=(jsonld_hint is not None),
        debug_pages=debug_pages,
        jsonld_hint=jsonld_hint or {},
    )


def process_batch(
    items:         list,
    output_fields: list,
    jina_cfg:      JinaConfig,
    llm_cfg:       LLMConfig,
    max_workers:   int  = 5,
    debug:         bool = False,
) -> list:
    """
    Process a list of (sku, secondary_term, original_row) tuples concurrently.
    secondary_term may be an empty string when no secondary column is configured.
    Returns results in completion order.
    debug=True populates SKUResult.debug_pages with the cleaned text per page.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_sku, sku, row, output_fields, jina_cfg, llm_cfg, secondary_term, debug): sku
            for sku, secondary_term, row in items
        }
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as e:
                sku = future_map[future]
                results.append(SKUResult(
                    sku=sku, status="error",
                    data={"review_flag": "ERROR", "_error": str(e)},
                    sources=[],
                ))
    return results
