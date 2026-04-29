"""
src/extractors.py
LLM extraction backends: OpenAI, Gemini, Claude.

Accepts an IdentifierSet so the prompt can include all known identifiers
(SKU, manufacturer code, brand, product name) as match-anchors, plus an
optional jsonld_hint dict for trusted structured-data context.
"""

import json
import threading
from dataclasses import dataclass

# Threading lock for Gemini's global SDK state — configure() mutates a
# module-level singleton which is not thread-safe under concurrent workers.
_gemini_lock = threading.Lock()


# ── Field definitions ─────────────────────────────────────────────────────────

FIELD_DEFINITIONS = {
    "product_name":      "Full cleaned product name/title",
    "brand":             "Brand or manufacturer name",
    "short_description": "Concise 1-2 sentence factual product description",
    "long_description":  "Detailed 3-6 sentence product description paragraph",
    "specifications":    "Technical specs as pipe-separated pairs: Key: Value | Key: Value",
    "category":          "Product category (e.g. Induction Cooktop, 4K Monitor)",
    "model_number":      "Manufacturer model number or MPN",
    "barcode":           "EAN/UPC/GTIN barcode — numbers only",
    "country_of_origin": "Country of manufacture if explicitly stated",
    "source_url":        "Most authoritative source URL",
    "confidence_score":  "High (3+ sources agree) / Medium (1-2 good sources) / Low (weak source)",
    "review_flag":       "VERIFIED or REVIEW_NEEDED (if values conflict across sources)",
}

SYSTEM_PROMPT = (
    "You are a precise product data extraction assistant. "
    "Respond ONLY with a valid JSON object — no markdown fences, no preamble."
)


def _identifier_block(ids) -> str:
    """Render IdentifierSet as a labelled block for the prompt."""
    parts = []
    if ids.primary:  parts.append(f"  Primary identifier:   {ids.primary}")
    if ids.fallback: parts.append(f"  Fallback identifier:  {ids.fallback}")
    return "\n".join(parts) if parts else "  (none provided)"


def build_prompt(
    ids,
    pages:        list,
    fields:       list,
    jsonld_hint:  dict | None = None,
    max_chars:    int = 0,
) -> str:
    """
    Build the extraction prompt. The IdentifierSet is rendered prominently
    so the LLM can verify the page is about the correct product before
    extracting fields. If the page describes a different product, the LLM
    should set review_flag=REVIEW_NEEDED.
    """
    field_list = "\n".join(
        f'  "{f}": {FIELD_DEFINITIONS.get(f, f)}'
        for f in fields if f in FIELD_DEFINITIONS
    )

    sources = ""
    for i, page in enumerate(pages, 1):
        text = page["content"]
        if max_chars and max_chars > 0:
            text = text[:max_chars]
        sources += f"\n\n--- SOURCE {i}: {page['url']} ---\n"
        sources += text

    all_fields = list(fields)
    for auto in ["review_flag", "confidence_score", "source_url"]:
        if auto not in all_fields:
            all_fields.append(auto)

    empty_json = json.dumps({f: "" for f in all_fields}, indent=2)

    hint_block = ""
    if jsonld_hint:
        hint_block = f"""
TRUSTED METADATA (from page's structured data — high confidence):
{json.dumps(jsonld_hint, indent=2)}

Use the above values directly where they are populated.
For any field that is empty or missing above, extract it from the sources below.
"""

    primary_label = ids.display_label() if hasattr(ids, "display_label") else "(unknown)"

    return f"""Extract product data for the product identified below.

PRODUCT IDENTIFIERS:
{_identifier_block(ids)}

RULES:
- ABSOLUTELY DO NOT return a fully empty result. If ANY source page provides
  ANY information about a product — even a different variant, a different
  language, or a partial match — extract that information and set
  review_flag=REVIEW_NEEDED. The user will manually verify whether it's the
  right product. Returning empty is worse than returning REVIEW_NEEDED data
  because the user cannot verify what they cannot see.
- Minimum required fields when source pages are provided: product_name (from
  page title or H1 if nothing else), source_url, review_flag. Never leave
  these three empty if you have any source content.
- If the page content is in a non-English language, translate description
  fields (short_description, long_description) into clear English. Keep the
  product name as-is from the page.
- Match the primary or fallback identifier against the page content to
  determine confidence. A clean match → review_flag=VERIFIED.
  A partial/uncertain match → review_flag=REVIEW_NEEDED with extracted data.
  No match at all → still extract what's there, review_flag=REVIEW_NEEDED.
- Only extract values explicitly present in the sources or trusted metadata —
  never invent data. But page metadata (title, meta description) IS valid
  source data; use it.
- For specifications use pipe format: "Width: 60cm | Power: 7.4kW | Colour: Black"
- If values conflict across sources, include both and set review_flag to REVIEW_NEEDED.
- ALWAYS populate source_url with the URL of the page that provided the most
  data, even when review_flag is REVIEW_NEEDED. Never leave source_url empty
  if any source page was provided.
{hint_block}
FIELDS TO EXTRACT:
{field_list}

Return ONLY valid JSON — no markdown, no explanation:
{empty_json}

SOURCES:
{sources}

JSON output for "{primary_label}":"""


def parse_json_response(text: str) -> dict:
    """
    Extract the first valid JSON object from LLM output.
    Uses raw_decode so trailing/extra content after the object doesn't break
    parsing on chatty models.
    """
    if not text:
        return {}
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        start = text.index("{")
    except ValueError:
        return {}
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


# ── OpenAI ────────────────────────────────────────────────────────────────────

OPENAI_MODELS = [
    # Current GPT-5 family (April 2026 — see developers.openai.com/api/docs/models)
    "gpt-5.4-mini",   # cheap + fast, recommended for batch extraction
    "gpt-5.4-nano",   # cheapest, lower quality
    "gpt-5.5",        # flagship — only use for hard SKUs
    # Legacy — still in API as of April 2026 but deprecated in ChatGPT
    "gpt-4o-mini",
    "gpt-4.1-mini",
]

def extract_openai(ids, pages, fields, api_key, model, jsonld_hint=None, max_chars=0):
    from openai import OpenAI
    client   = OpenAI(api_key=api_key)
    prompt   = build_prompt(ids, pages, fields, jsonld_hint, max_chars=max_chars)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    return parse_json_response(response.choices[0].message.content)


# ── Gemini ────────────────────────────────────────────────────────────────────

GEMINI_MODELS = [
    "gemini-2.5-flash",   # current cheap+fast — recommended default
    "gemini-2.5-pro",     # higher quality, more expensive
]

def extract_gemini(ids, pages, fields, api_key, model, jsonld_hint=None, max_chars=0):
    import google.generativeai as genai

    with _gemini_lock:
        genai.configure(api_key=api_key)

    m        = genai.GenerativeModel(model)
    prompt   = f"{SYSTEM_PROMPT}\n\n{build_prompt(ids, pages, fields, jsonld_hint, max_chars=max_chars)}"
    response = m.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=2048),
    )
    try:
        text = response.text
    except ValueError:
        return {
            "review_flag":      "ERROR",
            "confidence_score": "Low",
            "_error":           "Gemini safety filter blocked this response",
        }
    return parse_json_response(text)


# ── Claude ────────────────────────────────────────────────────────────────────

CLAUDE_MODELS = [
    # Current valid API model strings (April 2026 — see Anthropic models docs)
    "claude-haiku-4-5-20251001",   # cheapest — recommended for batch
    "claude-sonnet-4-6",           # default-quality balance
    "claude-opus-4-7",             # latest flagship — only for hard SKUs
    "claude-opus-4-6",             # previous flagship
]

def extract_claude(ids, pages, fields, api_key, model, jsonld_hint=None, max_chars=0):
    import anthropic
    client   = anthropic.Anthropic(api_key=api_key)
    prompt   = build_prompt(ids, pages, fields, jsonld_hint, max_chars=max_chars)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_json_response(response.content[0].text)


# ── Unified interface ─────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: str   # openai | gemini | claude
    api_key:  str
    model:    str


def extract(
    ids,
    pages:       list,
    fields:      list,
    cfg:         LLMConfig,
    jsonld_hint: dict | None = None,
    max_chars:   int = 0,
) -> dict:
    """Unified extraction interface."""
    if not pages:
        return {f: "" for f in fields}

    all_fields = list(fields)
    for auto in ["review_flag", "confidence_score", "source_url"]:
        if auto not in all_fields:
            all_fields.append(auto)

    try:
        if cfg.provider == "openai":
            result = extract_openai(ids, pages, all_fields, cfg.api_key, cfg.model, jsonld_hint, max_chars)
        elif cfg.provider == "gemini":
            result = extract_gemini(ids, pages, all_fields, cfg.api_key, cfg.model, jsonld_hint, max_chars)
        elif cfg.provider == "claude":
            result = extract_claude(ids, pages, all_fields, cfg.api_key, cfg.model, jsonld_hint, max_chars)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")
    except Exception as e:
        return {f: "" for f in all_fields} | {"review_flag": "ERROR", "_error": str(e)}

    for f in all_fields:
        if f not in result:
            result[f] = ""

    # Safety net: if the LLM bailed and returned everything empty despite us
    # providing source pages, populate the bare minimum so the row isn't
    # useless to the user. This handles cases where some LLMs refuse to
    # extract data when they suspect a product mismatch — even with explicit
    # prompt instructions, Gemini in particular is prone to this.
    important = ("product_name", "brand", "short_description", "long_description",
                 "specifications", "category", "model_number")
    has_any = any(str(result.get(f, "") or "").strip() for f in important)

    if not has_any and pages:
        # First fallback: borrow from JSON-LD hint if it exists
        if jsonld_hint:
            for f, v in jsonld_hint.items():
                if f in all_fields and not result.get(f):
                    result[f] = v
            has_any = any(str(result.get(f, "") or "").strip() for f in important)

        # Always populate source_url and flag for review — even if everything
        # else is still empty, the user gets a URL to manually check
        if not result.get("source_url"):
            result["source_url"] = pages[0]["url"]

        existing_flag = str(result.get("review_flag", "") or "").upper()
        if "ERROR" not in existing_flag:
            result["review_flag"] = "REVIEW_NEEDED"

        if not result.get("confidence_score"):
            result["confidence_score"] = "Low"

    return result
