"""
src/extractors.py
LLM extraction backends: OpenAI, Gemini, Claude.
Supports an optional jsonld_hint dict — pre-extracted structured data that is
injected into the prompt as trusted metadata, giving the LLM a head start
without bypassing it entirely.
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


def build_prompt(
    sku:          str,
    pages:        list,
    fields:       list,
    jsonld_hint:  dict | None = None,
    max_chars:    int = 0,
) -> str:
    """
    Build the extraction prompt.
    If jsonld_hint is provided it is injected as a TRUSTED METADATA block
    before the page sources. The LLM uses it as a reliable starting point
    and fills any missing or empty fields from the page content.

    max_chars: if > 0, each page's content is additionally capped here.
    Content is already cleaned and truncated by clean_content() in jina_client —
    this cap is a safety net only. Pass 0 to trust the pre-cleaned length.
    """
    field_list = "\n".join(
        f'  "{f}": {FIELD_DEFINITIONS.get(f, f)}'
        for f in fields if f in FIELD_DEFINITIONS
    )

    # Build source content — respect the configurable max_chars setting
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

    # Build optional trusted metadata block
    hint_block = ""
    if jsonld_hint:
        hint_block = f"""
TRUSTED METADATA (from page's structured data — high confidence):
{json.dumps(jsonld_hint, indent=2)}

Use the above values directly where they are populated.
For any field that is empty or missing above, extract it from the sources below.
"""

    return f"""Extract product data for SKU "{sku}" from the sources below.

RULES:
- Only extract values explicitly present in the sources or trusted metadata — never invent data
- If a field is not found anywhere, return empty string ""
- For specifications use pipe format: "Width: 60cm | Power: 7.4kW | Colour: Black"
- If values conflict across sources, include both and set review_flag to REVIEW_NEEDED
- Set source_url to the most authoritative URL used
{hint_block}
FIELDS TO EXTRACT:
{field_list}

Return ONLY valid JSON — no markdown, no explanation:
{empty_json}

SOURCES:
{sources}

JSON output for SKU "{sku}":"""


def parse_json_response(text: str) -> dict:
    """
    Extract the first valid JSON object from LLM output.

    Uses Python's raw JSON decoder to stop at the exact closing brace of the
    first object — this correctly handles "chatty" models that append extra
    text or a second JSON block after the primary response, both of which
    would cause json.loads() to raise an 'Extra data' error if we naively
    sliced from first { to last }.
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
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

def extract_openai(sku, pages, fields, api_key, model, jsonld_hint=None, max_chars=0):
    from openai import OpenAI
    client   = OpenAI(api_key=api_key)
    prompt   = build_prompt(sku, pages, fields, jsonld_hint, max_chars=max_chars)
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
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

def extract_gemini(sku, pages, fields, api_key, model, jsonld_hint=None, max_chars=0):
    import google.generativeai as genai

    # configure() mutates global SDK state — must be serialised across threads
    with _gemini_lock:
        genai.configure(api_key=api_key)

    m        = genai.GenerativeModel(model)
    prompt   = f"{SYSTEM_PROMPT}\n\n{build_prompt(sku, pages, fields, jsonld_hint, max_chars=max_chars)}"
    response = m.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=2048),
    )
    # Gemini safety filters can block responses — .text raises ValueError on blocked content
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
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]

def extract_claude(sku, pages, fields, api_key, model, jsonld_hint=None, max_chars=0):
    import anthropic
    client   = anthropic.Anthropic(api_key=api_key)
    prompt   = build_prompt(sku, pages, fields, jsonld_hint, max_chars=max_chars)
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
    sku:         str,
    pages:       list,
    fields:      list,
    cfg:         LLMConfig,
    jsonld_hint: dict | None = None,
    max_chars:   int = 0,
) -> dict:
    """
    Unified extraction interface.
    jsonld_hint: optional pre-extracted JSON-LD data injected into the prompt
    as trusted metadata — the LLM uses it as a foundation and fills gaps
    from the page content. Always goes through the LLM regardless.
    max_chars: passed to build_prompt to cap each page's content slice.
    """
    if not pages:
        return {f: "" for f in fields}

    all_fields = list(fields)
    for auto in ["review_flag", "confidence_score", "source_url"]:
        if auto not in all_fields:
            all_fields.append(auto)

    try:
        if cfg.provider == "openai":
            result = extract_openai(sku, pages, all_fields, cfg.api_key, cfg.model, jsonld_hint, max_chars)
        elif cfg.provider == "gemini":
            result = extract_gemini(sku, pages, all_fields, cfg.api_key, cfg.model, jsonld_hint, max_chars)
        elif cfg.provider == "claude":
            result = extract_claude(sku, pages, all_fields, cfg.api_key, cfg.model, jsonld_hint, max_chars)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")
    except Exception as e:
        return {f: "" for f in all_fields} | {"review_flag": "ERROR", "_error": str(e)}

    for f in all_fields:
        if f not in result:
            result[f] = ""

    return result
