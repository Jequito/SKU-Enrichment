"""
src/content_cleaner.py
Two jobs:
  1. JSON-LD extraction — find structured product data in page content.
     Used as a trusted metadata hint passed to the LLM, not to bypass it.
  2. Content-aware cleaning — strip navigation menus, footers, and junk
     before passing to the LLM, reducing token usage.
"""

import re
import json
from typing import Optional


# ── JSON-LD extraction ────────────────────────────────────────────────────────
#
# Only two safe patterns — both are anchored on script tag boundaries or
# markdown code fences. We deliberately do NOT use open-ended patterns
# like [^}]{50,}.*? over large strings as these can trigger catastrophic
# backtracking (ReDoS) on malformed content.
# json.loads() handles all validation once the raw block is extracted.

_JSONLD_PATTERNS = [
    # Script tags (Jina sometimes preserves these in its output)
    re.compile(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        re.S | re.I
    ),
    # Markdown code fences (Jina wraps some JSON in ```json blocks)
    re.compile(
        r'```(?:json)?\s*(\{.*?\})\s*```',
        re.S | re.I
    ),
]


def extract_jsonld(content: str) -> Optional[dict]:
    """
    Search page content for a JSON-LD Product schema block.
    Returns a normalised field dict if a valid Product is found, else None.

    This result is passed to the LLM as a trusted metadata hint —
    it is NOT used to bypass the LLM entirely.
    """
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
    """
    Recursively search a parsed JSON-LD structure for a Product entity.
    Handles: plain object, top-level array, @graph array.
    Continues past stubs with no name rather than returning the first hit.
    """
    if isinstance(data, list):
        for item in data:
            result = _find_product(item)
            if result:
                return result
        return None

    if not isinstance(data, dict):
        return None

    # Handle @graph — iterate ALL items, skip stubs without a name
    if "@graph" in data:
        for item in data["@graph"]:
            if isinstance(item, dict) and "Product" in str(item.get("@type", "")):
                result = _normalise_product(item)
                if result.get("product_name"):
                    return result   # Return first valid one, not first Product
        return None

    # Direct Product object
    if "Product" in str(data.get("@type", "")):
        return _normalise_product(data)

    return None


def _unwrap(val) -> str:
    """
    Safely unwrap a JSON-LD value that may be a plain string or an
    internationalised object like {"@language": "en", "@value": "..."}.
    Platforms such as Shopify and Magento commonly emit the latter — without
    this helper the raw dict would be coerced to its Python repr string and
    injected verbatim into the LLM prompt.
    """
    if isinstance(val, dict):
        return str(val.get("@value") or val.get("name") or "").strip()
    if isinstance(val, list):
        # Some platforms emit an array of language-tagged values — take the
        # first English one, or the first entry if none is tagged.
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

    # Brand
    brand = data.get("brand") or data.get("manufacturer", {})
    if isinstance(brand, dict):
        out["brand"] = _unwrap(brand.get("name", ""))
    elif brand:
        out["brand"] = _unwrap(brand)

    # Descriptions
    if desc := data.get("description"):
        desc_str = _unwrap(desc)
        if len(desc_str) > 300:
            out["short_description"] = desc_str[:300].rsplit(" ", 1)[0] + "…"
            out["long_description"]  = desc_str
        else:
            out["short_description"] = desc_str

    # Model / MPN
    for mpn_key in ("sku", "mpn", "productID", "model", "gtin14"):
        if val := data.get(mpn_key):
            out["model_number"] = _unwrap(val)
            break

    # Barcode
    for gtin_key in ("gtin13", "gtin12", "gtin8", "gtin", "isbn"):
        if val := data.get(gtin_key):
            out["barcode"] = re.sub(r"\D", "", _unwrap(val))
            break

    # Category
    if cat := data.get("category"):
        out["category"] = _unwrap(cat)

    # Country of origin
    if coo := data.get("countryOfOrigin"):
        out["country_of_origin"] = _unwrap(coo)

    # Specifications from additionalProperty
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


# ── Content cleaning ──────────────────────────────────────────────────────────

_LINK_LINE  = re.compile(r'^\s*\[.{1,80}\]\(.*?\)\s*$')
_LIST_LINK  = re.compile(r'^\s*[\*\-]\s*\[.{1,80}\]\(.*?\)\s*$')
_BREADCRUMB = re.compile(r'.{0,40}[›»>\/].{0,40}[›»>\/]')
_COOKIE     = re.compile(
    r'(we use cookies|cookie settings|accept all cookies|cookie preferences'
    r'|this site uses cookies|privacy preferences|gdpr|manage cookies)',
    re.I
)
_SOCIAL     = re.compile(
    r'^.{0,20}(share\s+on|follow\s+us|subscribe|newsletter'
    r'|facebook|twitter|instagram|pinterest).{0,30}$',
    re.I
)
_RELATED    = re.compile(
    r'^#{0,3}\s*(related products?|you may also|customers (also|who)|similar products?'
    r'|recommended|also (viewed|bought)|more (from|like this)|people also)',
    re.I | re.M
)
_FOOTER     = re.compile(
    r'(©\s*20\d\d|all rights reserved|terms of (use|service)'
    r'|privacy policy|sitemap\b|abn\s*\d)',
    re.I
)


def _is_junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _LINK_LINE.match(s) or _LIST_LINK.match(s):
        return True
    if _BREADCRUMB.match(s) and len(s) < 150:
        return True
    return False


def _find_content_start(lines: list) -> int:
    nav_streak      = 0
    content_started = False
    for i, line in enumerate(lines[:100]):
        if _is_junk_line(line) or (not line.strip() and nav_streak > 3):
            nav_streak += 1
        else:
            s = line.strip()
            if s and len(s) > 35 and not _is_junk_line(line):
                if nav_streak > 5 and not content_started:
                    content_started = True
                    return max(0, i - 1)
                content_started = True
            nav_streak = 0
    return 0


def _find_content_end(lines: list) -> int:
    midpoint = len(lines) // 2
    for i, line in enumerate(lines):
        if i < midpoint:
            continue
        if _RELATED.match(line.strip()):
            return i
    footer_idx  = len(lines)
    junk_streak = 0
    for i in range(len(lines) - 1, max(len(lines) - 80, midpoint), -1):
        if _is_junk_line(lines[i]) or _FOOTER.search(lines[i]):
            junk_streak += 1
            if junk_streak > 6:
                footer_idx = i
        else:
            junk_streak = 0
    return footer_idx


def clean_content(content: str, max_chars: int = 4000) -> str:
    """
    Clean Jina markdown before sending to the LLM:
      1. Skip leading navigation / mega menus
      2. Cut trailing footer / related products
      3. Remove cookie notices and social sharing lines
      4. Collapse excessive blank lines
      5. Truncate at a paragraph boundary
    """
    if not content:
        return ""

    lines = content.split("\n")
    start = _find_content_start(lines)
    end   = _find_content_end(lines)
    if end <= start:
        end = len(lines)

    trimmed          = lines[start:end]
    cleaned          = []
    skip_cookie      = False

    for line in trimmed:
        if _COOKIE.search(line):
            skip_cookie = True
        if skip_cookie:
            if not line.strip():
                skip_cookie = False
            continue
        if _SOCIAL.match(line.strip()):
            continue
        cleaned.append(line)

    out, blanks = [], 0
    for line in cleaned:
        if not line.strip():
            blanks += 1
            if blanks <= 2:
                out.append(line)
        else:
            blanks = 0
            out.append(line)

    result = "\n".join(out).strip()

    if len(result) > max_chars:
        truncated = result[:max_chars]
        para_cut  = truncated.rfind("\n\n")
        if para_cut > int(max_chars * 0.7):
            truncated = truncated[:para_cut]
        result = truncated

    return result
