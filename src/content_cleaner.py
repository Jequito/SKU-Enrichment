"""
src/content_cleaner.py
JSON-LD extraction from raw HTML (or already-cleaned text).
Used as a trusted metadata hint passed to the LLM, not to bypass it.

Note: the previous version of this module also contained a regex-based
markdown content cleaner. That work is now done by trafilatura inside
search_client.fetch_page, so the cleaner has been removed.
"""

import re
import json
from typing import Optional


# ── JSON-LD extraction ────────────────────────────────────────────────────────
#
# Two safe patterns — both anchored on script tag boundaries or markdown code
# fences. We deliberately avoid open-ended patterns over large strings to dodge
# catastrophic backtracking on malformed content.

_JSONLD_PATTERNS = [
    re.compile(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        re.S | re.I,
    ),
    re.compile(
        r'```(?:json)?\s*(\{.*?\})\s*```',
        re.S | re.I,
    ),
]


def extract_jsonld(content: str) -> Optional[dict]:
    """
    Search the input (raw HTML or markdown) for a JSON-LD Product schema block.
    Returns a normalised field dict if a valid Product is found, else None.

    The result is passed to the LLM as a trusted metadata hint —
    it is NOT used to bypass the LLM entirely.
    """
    if not content:
        return None

    for pattern in _JSONLD_PATTERNS:
        for raw in pattern.findall(content):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

            result = _find_product(data)
            if result and result.get("product_name"):
                return result

    return None


def _find_product(data) -> Optional[dict]:
    """Recursively search a parsed JSON-LD structure for a Product entity."""
    if isinstance(data, list):
        for item in data:
            result = _find_product(item)
            if result:
                return result
        return None

    if not isinstance(data, dict):
        return None

    # @graph — iterate ALL items, return the first valid Product
    if "@graph" in data:
        for item in data["@graph"]:
            if isinstance(item, dict) and "Product" in str(item.get("@type", "")):
                result = _normalise_product(item)
                if result.get("product_name"):
                    return result
        return None

    # Direct Product object
    if "Product" in str(data.get("@type", "")):
        return _normalise_product(data)

    return None


def _unwrap(val) -> str:
    """
    Safely unwrap a JSON-LD value that may be a plain string or an
    internationalised object like {"@language": "en", "@value": "..."}.
    """
    if isinstance(val, dict):
        return str(val.get("@value") or val.get("name") or "").strip()
    if isinstance(val, list):
        for item in val:
            if isinstance(item, dict) and item.get("@language", "").startswith("en"):
                return str(item.get("@value", "")).strip()
        first = val[0] if val else ""
        return _unwrap(first)
    return str(val).strip()


def _normalise_product(data: dict) -> dict:
    """Map a JSON-LD Product object to our output field schema."""
    out = {}

    if name := data.get("name"):
        out["product_name"] = _unwrap(name)

    brand = data.get("brand") or data.get("manufacturer", {})
    if isinstance(brand, dict):
        out["brand"] = _unwrap(brand.get("name", ""))
    elif brand:
        out["brand"] = _unwrap(brand)

    if desc := data.get("description"):
        desc_str = _unwrap(desc)
        if len(desc_str) > 300:
            out["short_description"] = desc_str[:300].rsplit(" ", 1)[0] + "…"
            out["long_description"]  = desc_str
        else:
            out["short_description"] = desc_str

    for mpn_key in ("sku", "mpn", "productID", "model", "gtin14"):
        if val := data.get(mpn_key):
            out["model_number"] = _unwrap(val)
            break

    for gtin_key in ("gtin13", "gtin12", "gtin8", "gtin", "isbn"):
        if val := data.get(gtin_key):
            out["barcode"] = re.sub(r"\D", "", _unwrap(val))
            break

    if cat := data.get("category"):
        out["category"] = _unwrap(cat)

    if coo := data.get("countryOfOrigin"):
        out["country_of_origin"] = _unwrap(coo)

    props = data.get("additionalProperty", [])
    if isinstance(props, list) and props:
        parts = []
        for prop in props:
            if isinstance(prop, dict):
                k = _unwrap(prop.get("name", ""))
                v = _unwrap(prop.get("value", "") or prop.get("unitText", ""))
                if k and v:
                    parts.append(f"{k}: {v}")
        if parts:
            out["specifications"] = " | ".join(parts)

    return out
