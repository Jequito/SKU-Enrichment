"""
app.py — SKU Bulk Product Enrichment
Streamlit frontend. Searches Google via SerpAPI, fetches via httpx +
trafilatura, extracts via OpenAI / Gemini / Claude.

Pause/Stop work because the pipeline runs ONE BATCH PER SCRIPT RUN — between
batches the script returns to the top and re-renders, allowing button clicks
to be processed.
"""

import math
import time
import streamlit as st
import pandas as pd

from src.search_client import SearchConfig, IdentifierSet, SERPAPI_COUNTRIES, COUNTRY_LABELS
from src.extractors    import LLMConfig, OPENAI_MODELS, GEMINI_MODELS, CLAUDE_MODELS
from src.file_handler  import read_file, to_csv_bytes, to_xlsx_bytes, build_fieldnames
from src.pipeline      import process_batch

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SKU Product Enrichment",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background-color: #0d0f14; color: #e8e8e8; }

  [data-testid="stSidebar"] {
    background-color: #12141a;
    border-right: 1px solid #1e2130;
  }
  h1 { font-weight: 800; letter-spacing: -0.03em; color: #ffffff; }
  h2, h3 { font-weight: 700; color: #e0e0e0; }

  .stTextInput input, .stSelectbox select, .stNumberInput input {
    background: #1a1d27 !important; border: 1px solid #2a2d3e !important;
    color: #e8e8e8 !important;
    border-radius: 6px !important;
  }

  /* Sliders */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
    background: #4f6ef7 !important;
  }
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #4f6ef7 !important;
    border-color: #4f6ef7 !important;
    box-shadow: 0 0 0 4px rgba(79,110,247,0.2) !important;
  }
  [data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #4f6ef7 !important;
    font-size: 12px !important;
  }
  [data-testid="stSlider"] [data-testid="stTickBarMin"],
  [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #4a4f6a !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #4f6ef7, #7c4ff7);
    color: white; border: none; border-radius: 8px;
    font-weight: 700; font-size: 15px; padding: 0.6rem 2rem;
    transition: all 0.2s ease; width: 100%;
  }
  .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(79,110,247,0.4); }
  .stButton > button:disabled { background: #2a2d3e !important; color: #4a4f6a !important; transform: none !important; }

  .stDownloadButton > button {
    background: #1a1d27; color: #4f6ef7;
    border: 1px solid #4f6ef7; border-radius: 8px; font-weight: 600;
  }
  .stDownloadButton > button:hover { background: #4f6ef7; color: white; }

  [data-testid="stMetric"] { background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 10px; padding: 1rem; }
  [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #4f6ef7; }

  .stProgress > div > div { background: linear-gradient(90deg, #4f6ef7, #7c4ff7); border-radius: 4px; }
  hr { border-color: #1e2130; }

  [data-testid="stExpander"] {
    background: #1a1d27 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 8px !important;
  }
  [data-testid="stExpander"] summary {
    color: #e8e8e8 !important;
    font-weight: 600 !important;
    padding: 0.6rem 0.8rem !important;
    border-radius: 8px !important;
  }
  [data-testid="stExpander"] summary:hover { background: #20243a !important; }

  .sidebar-section {
    font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4f6ef7;
    margin: 1.5rem 0 0.5rem 0; padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2130;
  }
  code { background: #1a1d27; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

defaults = {
    "results":         [],
    "running":         False,
    "paused":          False,
    "work_queue":      [],   # remaining (IdentifierSet, original_row) tuples
    "total_count":     0,    # total items in this run, for progress %
    "completed_count": 0,    # items already processed in this run
    "input_rows":      [],
    "input_columns":   [],
    "fieldnames":      [],
    "selected_fields": [],
    "run_export_fields": [],
    "sku_column":      "",
    "debug_log":       [],
    "error_log":       [],   # [{sku, error}, ...] across the whole run
    "stats":           {"success": 0, "review": 0, "not_found": 0,
                        "error": 0, "rate_limited": 0, "jsonld_hint": 0},
    "rate_limit_hit":  False,
    # Captured at Start. This is the SearchConfig that the pipeline ACTUALLY
    # used for the run, surfaced in the debug log so the user can verify
    # what was applied (independent of any UI/state confusion).
    "run_settings":    {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Widget state defaults ───────────────────────────────────────────────────
# Pre-initialise every toggle key in session_state. This is the recommended
# Streamlit pattern when you also pass `key=` — it avoids a documented quirk
# where mixing `value=` with `key=` can silently re-apply the default on
# re-renders triggered by changes to other widget params (e.g. `disabled=`
# flipping when Start is pressed). After this init block, the widgets below
# are called with `key=` only — no `value=` — so session_state is the single
# source of truth.
_widget_defaults = {
    "primary_exact_toggle":    True,
    "fallback_exact_toggle":   True,
    "restrict_language_toggle": True,
    "restrict_country_toggle": False,
    "debug_mode_toggle":       False,
}
for k, v in _widget_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

is_running = st.session_state["running"]
is_paused  = st.session_state["paused"]

key_label = "API Key"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 SKU Enrichment")
    st.markdown("---")
    st.markdown("## ⚙️ Configuration")
    if is_running:
        st.warning("⏳ Job running — settings locked", icon="🔒")

    # ── LLM ──
    st.markdown('<div class="sidebar-section">LLM Extraction Engine</div>', unsafe_allow_html=True)

    provider = st.selectbox("Provider", ["Gemini", "OpenAI", "Claude"], disabled=is_running)
    provider_key = provider.lower()

    if provider_key == "openai":
        model_list = OPENAI_MODELS;  key_label = "OpenAI API Key";  key_help = "platform.openai.com — gpt-4o-mini recommended for large batches"
    elif provider_key == "gemini":
        model_list = GEMINI_MODELS;  key_label = "Gemini API Key";  key_help = "aistudio.google.com — gemini-2.0-flash cheapest for large batches"
    else:
        model_list = CLAUDE_MODELS;  key_label = "Claude API Key";  key_help = "console.anthropic.com — claude-haiku recommended for large batches"

    llm_api_key = st.text_input(key_label, type="password", help=key_help, disabled=is_running)
    llm_model   = st.selectbox("Model", model_list, disabled=is_running)

    # ── Search ──
    st.markdown('<div class="sidebar-section">SerpAPI Search</div>', unsafe_allow_html=True)

    serpapi_api_key = st.text_input(
        "SerpAPI Key", type="password", disabled=is_running,
        help="Get a key at serpapi.com/manage-api-key.",
    )

    country_code = st.selectbox(
        "Search Region",
        options=list(SERPAPI_COUNTRIES.keys()), index=0, disabled=is_running,
        format_func=lambda c: COUNTRY_LABELS.get(c, c),
        help="Passed to Google as the gl parameter — biases results to that locale.",
    )

    restrict_language = st.toggle(
        "Restrict to region's language",
        disabled=is_running,
        key="restrict_language_toggle",
        help=(
            "Adds Google's lr=lang_X parameter to filter out foreign-language pages. "
            "Recommended ON — without it, an Australia search for an MPN can still return "
            "Polish, Czech, or German reseller pages that happen to match the code."
        ),
    )

    restrict_country = st.toggle(
        "Restrict to region only (strict)",
        disabled=is_running,
        key="restrict_country_toggle",
        help=(
            "Adds Google's cr=country parameter — much stricter than the region bias. "
            "OFF by default because Google's country detection occasionally excludes "
            "legitimate manufacturer pages on .com domains. Turn ON when you need "
            "results purely from your region's domains."
        ),
    )

    urls_per_sku = st.slider("URLs per product", 1, 5, 1, disabled=is_running,
                              help=(
                                  "Number of SUCCESSFUL fetches to collect per product. "
                                  "The pipeline walks the SERP top-to-bottom in Google's "
                                  "order, skipping blocked domains and PDFs, and keeps "
                                  "trying URLs until it has this many successful fetches "
                                  "or the SerpAPI results run out. Failed fetches "
                                  "(HTTP 403, timeouts, JS-only pages) don't count "
                                  "toward the target — the next URL gets tried. "
                                  "1 is enough most of the time."
                              ))

    blocked_domains_text = st.text_area(
        "Blocked domains (one per line)",
        value="",
        height=80,
        disabled=is_running,
        placeholder="yourcompany.com\nyourothersite.com.au",
        help=(
            "Domains to exclude from search results — useful for skipping your own "
            "sites so you don't enrich a SKU using data from the very page you're "
            "trying to populate. Exact-or-subdomain match: 'yourcompany.com' blocks "
            "yourcompany.com itself and any subdomain like shop.yourcompany.com, "
            "but does NOT block unrelated domains that just happen to contain the "
            "same letters. Social, marketplace and aggregator sites are already "
            "excluded by default."
        ),
    )

    with st.expander("Advanced fetch settings"):
        max_chars = st.slider("Max chars per page", 2000, 12000, 4000, step=500,
                               disabled=is_running,
                               help="Trafilatura output is truncated at this size before going to the LLM")
        timeout = st.slider("Request timeout (seconds)", 10, 60, 25, disabled=is_running,
                             help="Applied to SerpAPI calls and page fetches")
        max_results = st.slider("Google results per search", 5, 25, 10, disabled=is_running,
                                 help="Number of organic results to consider before scoring and picking the top URLs to fetch")
        delay_between = st.slider("Delay between fetches (s)", 0.0, 3.0, 0.5, step=0.25,
                                   disabled=is_running,
                                   help="Pause between page fetches within a single product")

    # ── Performance ──
    st.markdown('<div class="sidebar-section">Performance</div>', unsafe_allow_html=True)

    max_workers = st.slider(
        "Concurrent workers", 1, 20, 5, disabled=is_running,
        help=(
            "Products processed in parallel per batch. "
            "SerpAPI free tier handles ~5 concurrent. Paid plans go higher. "
            "Above the plan's per-second limit you'll see 429s."
        ),
    )

    delay_between_skus = st.slider(
        "Delay between batches (s)", 0, 10, 1, disabled=is_running,
        help="Pause between concurrent batches — useful on free-tier SerpAPI",
    )

    st.divider()
    debug_mode = st.toggle(
        "🔬 Debug mode", disabled=is_running,
        key="debug_mode_toggle",
        help=(
            "Stores the cleaned text sent to the LLM for each product. "
            "Turn off for large batches — keeps everything in memory."
        ),
    )

    if st.session_state["input_rows"]:
        n        = len(st.session_state["input_rows"])
        secs_low  = math.ceil(n / max_workers) * 4
        secs_high = math.ceil(n / max_workers) * 7
        hrs_low   = round(secs_low / 3600, 1)
        hrs_high  = round(secs_high / 3600, 1)
        if hrs_high < 1:
            est = f"~{round(secs_low/60)}–{round(secs_high/60)} min"
        else:
            est = f"~{hrs_low}–{hrs_high} hrs"
        st.caption(f"⏱ Estimated runtime for {n:,} products: **{est}** at {max_workers} workers")

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("# 🔍 SKU Product Enrichment")
st.markdown("Bulk enrich product codes with descriptions, specs and metadata — Google search via SerpAPI.")

with st.expander("ℹ️ How this tool works", expanded=False):
    st.markdown("""
This tool takes a list of products from a CSV or Excel file and looks each
one up on **Google via SerpAPI**. It fetches the top product pages with
`httpx`, extracts clean content with `trafilatura`, and uses an **LLM**
(OpenAI, Gemini, or Claude) to extract structured fields. Results are
downloadable as CSV or Excel.

#### Optional fallback column

For each row, the tool runs a Google search on your **Primary** column.
By default it's wrapped in double quotes for an exact-phrase match —
toggle that off in the sidebar if your values are formatted differently
across sites or if you're searching by product name where loose matching
helps.

Mapping a **Fallback** column is optional. When you do, it kicks in only
if the primary search produced no extracted descriptions: the LLM ran
fine, but the page Google surfaced didn't have descriptive copy. The
tool then re-runs the search using the fallback term and adopts that
result if it actually has descriptions. This catches thin pages — PDF
spec sheets, replacement-parts listings, discontinued-product stubs.

Each column has its own exact-match toggle, so you can quote a SKU
strictly while leaving a product-name primary loose, or vice versa.

#### URL selection

The "URLs per product" slider sets the number of **successful** fetches
to collect per product. The pipeline walks the SERP top-to-bottom in
Google's order, skipping blocked domains and PDFs (PDFs can't be cleaned
by `trafilatura`), and keeps trying URLs until it has that many
successful fetches or the SerpAPI results are exhausted. Failed fetches
— HTTP 403, Cloudflare blocks, timeouts, JS-only pages — don't count
toward the target. The next URL just gets tried.

So if you set the slider to 2 and the top 4 results all fail, the
pipeline keeps walking down to position 5, 6, etc. until it hits 2
successes or runs out. Every URL it tried, successful or not, is
recorded in the diagnostic panel.

#### Confidence flags

After extraction, rows are flagged based on what the LLM and the page
content tell us:

- `VERIFIED` — clean match with strong content.
- `REVIEW_NEEDED` — the LLM itself flagged the row (conflicting values
  across sources, partial match, etc.).
- `LOW_CONFIDENCE` — extraction succeeded but no fetched page appears
  to be about the searched product based on token overlap. Useful for
  catching wrong-product matches on ambiguous identifiers (numeric SKUs
  that collide across industries, SEO spam pages that surface for a
  model number but actually sell something else).
- `ERROR` — extraction failed.

#### JSON-LD hints

When a page contains `<script type="application/ld+json">` Product schema, the
structured data is passed to the LLM as a trusted starting point AND used as
a per-field safety net — any field the LLM leaves empty gets backfilled from
the JSON-LD if available. This produces significantly more accurate results
on retailer and manufacturer sites.
    """)

st.divider()

if st.session_state["rate_limit_hit"]:
    st.error("⚠️ **SerpAPI rate limit or quota hit.** Wait, then click Resume. Results so far are safe to download.", icon="🚫")

# ── Upload & config ───────────────────────────────────────────────────────────

col_upload, col_config = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📂 Upload Product File")
    st.caption("CSV or Excel — formulas and multi-sheet structures are not preserved in output.")

    uploaded = st.file_uploader("CSV or Excel file",
                                 type=["csv","xlsx","xls"], disabled=is_running)
    if uploaded:
        try:
            rows, columns = read_file(uploaded.read(), uploaded.name)
            st.session_state["input_rows"]    = rows
            st.session_state["input_columns"] = columns
            st.success(f"✓ Loaded **{len(rows):,}** rows · **{len(columns)}** columns")
            if len(rows) > 1000:
                n = len(rows)
                secs_low  = math.ceil(n / 5) * 4
                secs_high = math.ceil(n / 5) * 7
                hrs_low   = round(secs_low / 3600, 1)
                hrs_high  = round(secs_high / 3600, 1)
                st.info(
                    f"ℹ️ **Large batch ({n:,} products).** "
                    f"Estimated runtime at 5 workers: **{hrs_low}–{hrs_high} hrs**. "
                    f"Keep this tab active. For 5k+ products run locally."
                )
        except Exception as e:
            st.error(f"Could not read file: {e}")
            rows, columns = [], []
    else:
        rows    = st.session_state["input_rows"]
        columns = st.session_state["input_columns"]

with col_config:
    st.markdown("### 🏷️ Identifier Columns")
    st.caption("Pick the column to search on. Fallback is optional — when set, it's used only if the primary search produces no extracted descriptions.")

    # Defaults so SearchConfig has values even before the file is uploaded.
    # Streamlit's `key=` makes the toggles below remember the user's choice
    # across reruns regardless of which branch we hit here.
    primary_exact  = True
    fallback_exact = True

    if columns:
        # Auto-detect: manufacturer/product code typically goes in Primary,
        # internal SKU in Fallback. User can flip either dropdown.
        primary_hints = ["product_code","mpn","model","manufacturer","mfr","part_no","part_number","name","title"]

        def _auto_index(opts, hints):
            for i, c in enumerate(opts):
                if any(h in str(c).lower() for h in hints):
                    return i
            return 0

        primary_column = st.selectbox(
            "Primary search column (required)",
            columns,
            index=_auto_index(columns, primary_hints),
            disabled=is_running,
            help="Searched first. Can be a product name, MPN, manufacturer code, or any string identifier.",
        )
        # Per-column exact-match toggle. By default the value is wrapped in
        # double quotes for a verbatim Google match. Turn OFF when this
        # column's values are formatted differently across vendor sites
        # (e.g. you store 'WVE-T60S' but pages render it as 'WVE T60 S').
        # Without quotes Google does its usual loosening — handy for those
        # cases, but it can also pull in adjacent products.
        primary_exact = st.toggle(
            "Exact match (quoted)",
            disabled=is_running,
            key="primary_exact_toggle",
            help=(
                "Wraps the Primary search in double quotes to force Google to "
                "match the value verbatim. ON by default for unambiguous codes. "
                "Turn OFF when the code is formatted differently across sites — "
                "e.g. you store 'WVE-T60S' but pages render it as 'WVE T60 S', "
                "or when searching by product name where quoting kills loose "
                "matching."
            ),
        )

        # Fallback column is OPTIONAL. The "(none)" option disables the
        # whole fallback path — no fallback search runs, the per-column
        # exact-match toggle below is hidden, and rows whose primary
        # search returned no descriptions just come back as-is.
        fallback_options = ["(none)"] + [c for c in columns if c != primary_column]
        fallback_column_choice = st.selectbox(
            "Fallback search column (optional)",
            fallback_options,
            index=0,   # default to "(none)" — fallback is opt-in, not auto
            disabled=is_running,
            help=(
                "Optional. Used only if the primary search returns no "
                "extracted descriptions. Typically your internal SKU. "
                "Leave as '(none)' if you only want to search the primary."
            ),
        )
        fallback_column = "" if fallback_column_choice == "(none)" else fallback_column_choice

        if fallback_column:
            fallback_exact = st.toggle(
                "Exact match (quoted)",
                disabled=is_running,
                key="fallback_exact_toggle",
                help=(
                    "Same as the Primary toggle, applied to the Fallback search. "
                    "Turn OFF if your fallback column holds values whose formatting "
                    "varies across vendors."
                ),
            )

        # Optional category column. Independently useful: codes like
        # "1204086639" or "105666" can legitimately match different
        # products in different industries. When set, the category value
        # is appended to the search query as a loose refinement (never
        # quoted) and is also passed to the LLM prompt as context.
        category_options = ["(none)"] + [
            c for c in columns
            if c != primary_column and c != fallback_column
        ]
        category_column = st.selectbox(
            "Category hint column (optional)",
            category_options,
            index=0,
            disabled=is_running,
            help=(
                "Optional. When set, the row's category value is appended to "
                "Google searches as a refinement (e.g. `\"1204086639\" sheet "
                "music`) AND included in the LLM prompt as context. Helps "
                "disambiguate ambiguous numeric codes that match products in "
                "unrelated industries. Leave as '(none)' if your codes are "
                "already unique enough."
            ),
        )
    else:
        primary_column  = "— upload a file first —"
        fallback_column = ""
        category_column = "(none)"
        st.selectbox("Primary search column (required)",  ["— upload a file first —"], disabled=True)
        st.selectbox("Fallback search column (optional)", ["— upload a file first —"], disabled=True)
        st.selectbox("Category hint column (optional)",   ["— upload a file first —"], disabled=True)

    # The download/export logic uses sku_column as the row label — point it at
    # the primary column so the existing export plumbing keeps working unchanged.
    st.session_state["sku_column"] = primary_column

    st.markdown("**Fields to extract**")
    field_options = {
        "product_name":      "Product Name",
        "brand":              "Brand",
        "short_description": "Short Description",
        "long_description":  "Long Description",
        "specifications":    "Specifications",
        "category":          "Category",
        "model_number":      "Model Number",
        "barcode":           "Barcode (EAN/UPC)",
        "country_of_origin": "Country of Origin",
        "source_url":        "Source URL",
        "all_source_urls":   "All Source URLs",
        "confidence_score":  "Confidence Score",
        "review_flag":       "Review Flag",
    }
    # Defaults: same first 6 as before (product_name → category) plus
    # all_source_urls and review_flag. all_source_urls is on by default
    # because it gives the user the full audit trail of which pages went
    # into the LLM, which is more transparent than the LLM's single
    # "most authoritative" pick.
    default_checked = list(field_options.keys())[:6] + ["all_source_urls", "review_flag"]
    selected_fields = []
    ca, cb = st.columns(2)
    for i, (key, label) in enumerate(field_options.items()):
        col = ca if i % 2 == 0 else cb
        if col.checkbox(label, value=(key in default_checked), key=f"field_{key}", disabled=is_running):
            selected_fields.append(key)
    if "review_flag" not in selected_fields:
        selected_fields.append("review_flag")

# ── Controls ──────────────────────────────────────────────────────────────────

st.divider()
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

valid_columns = bool(columns) and primary_column != "— upload a file first —"

can_start = (
    bool(rows) and bool(llm_api_key) and bool(selected_fields)
    and valid_columns and bool(serpapi_api_key)
)

with c1:
    if is_paused:
        start_btn = st.button("▶  Resume", disabled=not can_start)
    else:
        start_btn = st.button("▶  Start Enrichment", disabled=not can_start or is_running)
with c2:
    pause_btn = st.button("⏸  Pause", disabled=not is_running)
with c3:
    stop_btn  = st.button("⏹  Stop",  disabled=not is_running and not is_paused)
with c4:
    clear_btn = st.button("🗑  Clear", disabled=is_running)

if is_paused:
    done_so_far = len(st.session_state["results"])
    remaining   = len(st.session_state["work_queue"])
    st.info(f"⏸ **Paused** — {done_so_far} products done, {remaining} remaining. Adjust settings if needed, then click Resume.")

if not can_start and not is_running and not is_paused:
    missing = []
    if not rows:               missing.append("upload a file")
    if not llm_api_key:        missing.append(f"enter {key_label}")
    if not serpapi_api_key:    missing.append("enter SerpAPI key")
    if not selected_fields:    missing.append("select at least one field")
    if not valid_columns:      missing.append("map a Primary column")
    if missing:
        st.caption(f"⚠️  To start: {', '.join(missing)}")

# ── Button handlers ───────────────────────────────────────────────────────────
#
# Pause and Stop set state and rerun. They take effect at the START of the
# next script run. Because the pipeline runs ONE BATCH PER RUN (see further
# below), these buttons are guaranteed to be processed between any two batches.

if pause_btn:
    st.session_state["paused"]  = True
    st.session_state["running"] = False
    st.rerun()

if stop_btn:
    st.session_state["running"]    = False
    st.session_state["paused"]     = False
    st.session_state["work_queue"] = []
    st.rerun()

if clear_btn:
    for k in ("results", "fieldnames", "work_queue", "run_export_fields", "debug_log", "error_log"):
        st.session_state[k] = []
    st.session_state["stats"] = {"success": 0, "review": 0, "not_found": 0,
                                  "error": 0, "rate_limited": 0, "jsonld_hint": 0}
    st.session_state["rate_limit_hit"] = False
    st.session_state["paused"]         = False
    st.session_state["completed_count"] = 0
    st.session_state["total_count"]     = 0
    st.rerun()

# ── Start button: build queue and launch ──────────────────────────────────────

if start_btn and not is_paused:
    # Resolve the optional category column once. "(none)" means no category.
    _cat_col = category_column if category_column and category_column != "(none)" else ""

    # Fresh start — reset everything
    work_items = []
    for row in rows:
        ids = IdentifierSet(
            primary  = str(row.get(primary_column, "")  or "").strip(),
            fallback = str(row.get(fallback_column, "") or "").strip() if fallback_column else "",
            category = str(row.get(_cat_col, "") or "").strip() if _cat_col else "",
        )
        if not ids.is_empty():
            work_items.append((ids, row))

    fieldnames = build_fieldnames(columns, selected_fields)

    # Snapshot the SearchConfig values that will be applied for this run.
    # Saved to session_state so the debug log can show them definitively
    # — independent of whether the toggles get changed afterwards.
    run_settings_snapshot = {
        "primary_exact":     primary_exact,
        "fallback_exact":    fallback_exact,
        "country_code":      country_code,
        "restrict_language": restrict_language,
        "restrict_country":  restrict_country,
        "urls_per_sku":      urls_per_sku,
        "max_chars":         max_chars,
        "primary_column":    primary_column,
        "fallback_column":   fallback_column or "(none)",
        "category_column":   _cat_col or "(none)",
        "blocked_domains":   [d for d in (blocked_domains_text or "").splitlines() if d.strip()],
        "provider":          provider_key,
        "model":             llm_model,
    }

    st.session_state.update({
        "running":           True,
        "paused":            False,
        "work_queue":        work_items,
        "total_count":       len(work_items),
        "completed_count":   0,
        "results":           [],
        "debug_log":         [],
        "error_log":         [],
        "rate_limit_hit":    False,
        "stats":             {"success": 0, "review": 0, "not_found": 0,
                              "error": 0, "rate_limited": 0, "jsonld_hint": 0},
        "fieldnames":        fieldnames,
        "selected_fields":   selected_fields,
        "run_export_fields": fieldnames,
        "sku_column":        primary_column,
        "run_settings":      run_settings_snapshot,
    })
    st.rerun()

if start_btn and is_paused:
    # Resume — work_queue already holds whatever is left
    st.session_state["running"] = True
    st.session_state["paused"]  = False
    st.rerun()

# ── Pipeline: ONE BATCH PER SCRIPT RUN ────────────────────────────────────────
#
# This is the key fix for the disabled Pause/Stop buttons. The previous version
# ran every batch inside one synchronous loop, which blocked Streamlit from
# re-rendering and processing button clicks for the entire duration of the run.
#
# Here we process exactly one batch, then call st.rerun(). On the next run,
# the script returns to the top, button clicks are processed, and (if still
# running) we process the next batch. This means Pause/Stop are guaranteed
# to take effect within one batch's worth of latency.

if is_running and st.session_state["work_queue"]:
    # Parse the user blocklist textarea — normalise to bare domain substrings.
    # Substring match is intentional: "yourcompany.com" also matches
    # shop.yourcompany.com without requiring the user to enumerate subdomains.
    def _normalise_domain(s: str) -> str:
        s = s.strip().lower()
        for prefix in ("https://", "http://", "www."):
            if s.startswith(prefix):
                s = s[len(prefix):]
        return s.rstrip("/")

    user_blocked = tuple(
        _normalise_domain(d)
        for d in (blocked_domains_text or "").splitlines()
        if d.strip()
    )

    search_cfg = SearchConfig(
        serpapi_api_key   = serpapi_api_key,
        blocked_domains   = user_blocked,
        country_code      = country_code,
        restrict_language = restrict_language,
        restrict_country  = restrict_country,
        primary_exact     = primary_exact,
        fallback_exact    = fallback_exact,
        urls_per_sku      = urls_per_sku,
        max_chars         = max_chars,
        timeout           = timeout,
        max_results       = max_results,
        delay_between     = delay_between,
    )
    llm_cfg = LLMConfig(provider=provider_key, api_key=llm_api_key, model=llm_model)

    fieldnames = st.session_state["fieldnames"]
    total      = st.session_state["total_count"]
    work_queue = st.session_state["work_queue"]

    # Take one batch off the front of the queue
    batch_size = min(max_workers, len(work_queue))
    batch      = work_queue[:batch_size]
    remaining  = work_queue[batch_size:]
    st.session_state["work_queue"] = remaining

    # Render progress for THIS batch
    completed       = st.session_state["completed_count"]
    progress_global = completed / max(total, 1)

    skus_label = ", ".join(ids.display_label() for ids, _ in batch[:3])
    if len(batch) > 3:
        skus_label += f" +{len(batch)-3} more"

    progress_bar = st.progress(
        progress_global,
        text=f"Processing **{skus_label}** ({completed + 1}–{completed + len(batch)}/{total})"
    )
    status_box = st.empty()
    table_box  = st.empty()

    # Process the batch
    batch_results = process_batch(batch, st.session_state["selected_fields"],
                                   search_cfg, llm_cfg,
                                   max_workers=max_workers, debug=debug_mode)

    rate_limited = False
    batch_errors = []   # collected (sku, error_msg) for display below the progress bar
    rate_limited_keys: set = set()  # identifier keys of rows that need re-queueing

    for result in batch_results:
        s = st.session_state["stats"]
        if result.status == "success":
            s["success"] += 1
        elif result.status == "review":
            s["review"] += 1
        elif result.status == "rate_limited":
            # Don't count or persist this row — it's going back into the
            # work queue and will be retried on resume. Counting/persisting
            # now would either double-count it (success-after-retry while
            # stats["rate_limited"] still reflects the failed attempt) or
            # leave a permanent RATE_LIMITED stub in the output.
            st.session_state["rate_limit_hit"] = True
            rate_limited = True
            rate_limited_keys.add((
                (result.primary_value or "").lower(),
                (result.fallback_value or "").lower(),
            ))
            status_box.error(f"🚫 Rate limit hit on **{result.sku}**. Pausing — results so far are safe to download.")
            continue   # skip the data persistence below
        elif result.status in ("not_found", "blocked"):
            s["not_found"] += 1
            # Surface the actual reason (HTTP 403, timeout, recall-and-precision
            # both empty, blocklisted, etc.) so the user can act on it.
            if result.error_msg:
                kind = "blocked" if result.status == "blocked" else "not_found"
                st.session_state.setdefault("error_log", []).append(
                    {"sku": result.sku, "error": result.error_msg, "kind": kind}
                )
        else:
            s["error"] += 1
            if result.error_msg:
                batch_errors.append((result.sku, result.error_msg))
                # Persist for the post-run summary panel
                st.session_state.setdefault("error_log", []).append(
                    {"sku": result.sku, "error": result.error_msg, "kind": "llm_error"}
                )

        if result.had_jsonld:
            s["jsonld_hint"] += 1

        st.session_state["results"].append(result.data)

        if debug_mode and result.debug_pages:
            st.session_state["debug_log"].append({
                "sku":            result.sku,
                "primary_value":  result.primary_value,
                "fallback_value": result.fallback_value,
                "category_value": result.category_value,
                "status":         result.status,
                "pages":          result.debug_pages,
                "jsonld_hit":     result.had_jsonld,
                "jsonld_hint":    result.jsonld_hint,
                "query_used":     result.query_used,
                "error_msg":      result.error_msg,
            })

    # Re-queue rate-limited rows at the FRONT of the work queue so resume
    # retries them first. Without this, a 429 mid-batch silently drops
    # every rate-limited row from the run — the row never gets retried,
    # and the output shows a RATE_LIMITED stub instead of enriched data.
    requeued_count = 0
    if rate_limited_keys:
        requeue: list = []
        for ids, row in batch:
            key = (
                (ids.primary  or "").strip().lower(),
                (ids.fallback or "").strip().lower(),
            )
            if key in rate_limited_keys:
                requeue.append((ids, row))
        if requeue:
            st.session_state["work_queue"] = requeue + st.session_state["work_queue"]
            requeued_count = len(requeue)

    # Increment completed_count by the number of rows that ACTUALLY
    # finished. Otherwise the progress bar overshoots and the run looks
    # "done" before the queued retries finish.
    st.session_state["completed_count"] += (len(batch) - requeued_count)

    # Surface errors from this batch directly under the progress bar so the
    # user sees what's failing in real time, not just an opaque ERROR flag.
    if batch_errors and not rate_limited:
        first_sku, first_err = batch_errors[0]
        more = f" (+{len(batch_errors)-1} more in this batch)" if len(batch_errors) > 1 else ""
        status_box.error(
            f"💥 **{first_sku}** errored: {first_err[:300]}{'…' if len(first_err) > 300 else ''}{more}"
        )

    # Show a quick preview of the latest results
    preview_df   = pd.DataFrame(st.session_state["results"][-20:])
    preview_cols = [c for c in fieldnames if c in preview_df.columns]
    table_box.dataframe(
        preview_df[preview_cols] if preview_cols else preview_df,
        use_container_width=True, height=280,
    )

    # End-of-run handling
    if rate_limited:
        # Pause the run so the user can adjust settings and resume
        st.session_state["running"] = False
        st.session_state["paused"]  = True
        st.rerun()
    elif not st.session_state["work_queue"]:
        # All done
        st.session_state["running"] = False
        st.session_state["paused"]  = False
        progress_bar.progress(1.0, text="✅ Complete!")
        st.rerun()
    else:
        # More batches to go — pause briefly then rerun
        time.sleep(delay_between_skus)
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state["results"]:
    stats      = st.session_state["stats"]
    total_done = stats["success"] + stats["review"] + stats["not_found"] + stats["error"] + stats["rate_limited"]
    jsonld_pct = round(stats["jsonld_hint"] / max(total_done, 1) * 100)

    st.divider()
    st.markdown("### 📊 Results")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Processed",       total_done)
    m2.metric("✅ Success",       stats["success"])
    m3.metric("📐 JSON-LD Hint",  stats["jsonld_hint"], help="Products where structured page data was found and used as a trusted hint")
    m4.metric("⚠️ Review",        stats["review"])
    m5.metric("❌ Not Found",     stats["not_found"])
    m6.metric("🚫 Rate Limited",  stats["rate_limited"])
    m7.metric("💥 Error",         stats["error"])

    if stats["jsonld_hint"] > 0:
        st.caption(f"📐 **{jsonld_pct}%** of products had JSON-LD structured data — used as trusted metadata for higher accuracy.")

    # ── Diagnostic panel ───────────────────────────────────────────────────
    # Surfaces what actually went wrong on each problem row, grouped by kind
    # and message so a single global issue doesn't drown out per-row ones.
    error_log = st.session_state.get("error_log", [])
    if error_log:
        from collections import Counter

        llm_errors    = [e for e in error_log if e.get("kind") == "llm_error"]
        fetch_blocks  = [e for e in error_log if e.get("kind") == "blocked"]

        n_llm = len(llm_errors)
        # n_block counts URL-level fetch failures, not SKUs. Each fetch_block
        # entry's `error` field looks like "url1 → reason1 | url2 → reason2",
        # so the URL count is the number of " | " separators + 1 per entry.
        # Falls back to 1 if the entry shape is unexpected.
        n_block = sum(
            max(1, len(str(e.get("error", "")).split(" | ")))
            for e in fetch_blocks
        )
        n_block_skus = len(fetch_blocks)

        header_parts = []
        if n_llm:
            header_parts.append(f"{n_llm} extraction error(s)")
        if n_block:
            # Show URL count, with SKU count in parens when they differ.
            if n_block == n_block_skus:
                header_parts.append(f"{n_block} fetch block(s)")
            else:
                header_parts.append(f"{n_block} fetch block(s) across {n_block_skus} SKU(s)")
        header = " · ".join(header_parts) or f"{len(error_log)} issue(s)"

        with st.expander(f"💥 **Diagnostics — {header}**", expanded=True):

            if llm_errors:
                st.markdown("**LLM extraction errors**")
                st.caption(
                    "Returned by your LLM provider. Common causes: invalid API key, "
                    "wrong model identifier, quota/rate limits, content filters."
                )
                msg_counts = Counter(e["error"][:400] for e in llm_errors)
                for msg, count in msg_counts.most_common():
                    example_sku = next(
                        (e["sku"] for e in llm_errors if e["error"][:400] == msg),
                        "?",
                    )
                    badge = f"`×{count}`" if count > 1 else ""
                    st.markdown(f"**{example_sku}** {badge}")
                    st.code(msg, language=None)

            if fetch_blocks:
                if llm_errors:
                    st.divider()
                st.markdown("**Fetch blocks** (search found URLs but couldn't usefully read them)")
                st.caption(
                    "Common causes: HTTP 403/Cloudflare, JS-only SPA pages, login walls, "
                    "broken links, or domain blocked by your blocklist. The actual reason "
                    "for each URL attempted is shown below."
                )
                msg_counts = Counter(e["error"][:400] for e in fetch_blocks)
                for msg, count in msg_counts.most_common():
                    example_sku = next(
                        (e["sku"] for e in fetch_blocks if e["error"][:400] == msg),
                        "?",
                    )
                    badge = f"`×{count}`" if count > 1 else ""
                    st.markdown(f"**{example_sku}** {badge}")
                    st.code(msg, language=None)

    st.divider()

    fieldnames = st.session_state["fieldnames"]
    df         = pd.DataFrame(st.session_state["results"])
    show_cols  = [c for c in fieldnames if c in df.columns]
    st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, height=400)

    st.markdown("### ⬇️ Download")

    export_mode = st.radio(
        "Export format",
        ["Combined — original columns + enriched fields",
         "Enriched fields only — SKU + selected output fields"],
        horizontal=True,
        help=(
            "Combined keeps all your original input columns and appends the enriched fields. "
            "Enriched only outputs just the SKU column plus the fields you selected."
        ),
    )

    all_rows = st.session_state["results"]

    _locked_fields    = st.session_state.get("run_export_fields") or fieldnames
    _locked_sku_col   = st.session_state.get("sku_column", "")
    # The actual enrichment fields the user checked at run time. Distinct
    # from _locked_fields, which is `original_columns + enrichment_fields`
    # (used for "Combined" export). Without this, "Enriched only" was
    # iterating over the merged list and exporting every original column too.
    _enrichment_only  = st.session_state.get("selected_fields", []) or []

    if export_mode.startswith("Enriched"):
        # SKU column + only the enrichment fields. No original columns.
        _enriched_cols = []
        _seen = set()
        for _c in ([_locked_sku_col] if _locked_sku_col else []) + list(_enrichment_only):
            if _c and _c not in _seen:
                _enriched_cols.append(_c)
                _seen.add(_c)
        export_fieldnames = _enriched_cols or list(_enrichment_only)
        export_label      = "enriched"
    else:
        export_fieldnames = _locked_fields
        export_label      = "enriched_products"

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Download CSV",
            data=to_csv_bytes(all_rows, export_fieldnames),
            file_name=f"{export_label}.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "⬇️ Download Excel",
            data=to_xlsx_bytes(all_rows, export_fieldnames),
            file_name=f"{export_label}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Debug panel
    debug_log = st.session_state.get("debug_log", [])
    if debug_log:
        st.divider()
        st.markdown("### 🔬 Debug Log")

        # Build a readable plaintext serialisation for download
        from datetime import datetime

        def _build_debug_txt(log: list, error_log: list, run_settings: dict) -> str:
            lines = []
            lines.append("=" * 70)
            lines.append("SKU ENRICHMENT — DEBUG LOG")
            lines.append(f"Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Debug entries: {len(log)}")
            lines.append(f"Error entries: {len(error_log)}")
            lines.append("=" * 70)
            lines.append("")

            # Run configuration as actually applied to the pipeline. Captured
            # at Start; surfaces what each toggle / dropdown was set to so the
            # user can verify the run honoured their settings.
            if run_settings:
                lines.append("RUN SETTINGS (as applied to the pipeline)")
                lines.append("-" * 70)
                lines.append(f"  Provider / model:      {run_settings.get('provider', '?')} / {run_settings.get('model', '?')}")
                lines.append(f"  Region (gl):           {run_settings.get('country_code', '?')}")
                lines.append(f"  Restrict language:     {'yes' if run_settings.get('restrict_language') else 'no'}")
                lines.append(f"  Restrict country:      {'yes' if run_settings.get('restrict_country') else 'no'}")
                lines.append(f"  Primary column:        {run_settings.get('primary_column', '?')}")
                lines.append(f"  Primary EXACT match:   {'yes (quoted)' if run_settings.get('primary_exact') else 'NO (loose)'}")
                lines.append(f"  Fallback column:       {run_settings.get('fallback_column', '?')}")
                lines.append(f"  Fallback EXACT match:  {'yes (quoted)' if run_settings.get('fallback_exact') else 'NO (loose)'}")
                lines.append(f"  Category column:       {run_settings.get('category_column', '(none)')}")
                lines.append(f"  URLs per product:      {run_settings.get('urls_per_sku', '?')}")
                lines.append(f"  Max chars per page:    {run_settings.get('max_chars', '?')}")
                blocked = run_settings.get("blocked_domains") or []
                lines.append(f"  Blocked domains:       {', '.join(blocked) if blocked else '(none)'}")
                lines.append("")

            # Per-row debug entries
            for i, entry in enumerate(log, 1):
                lines.append("-" * 70)
                lines.append(f"[{i}/{len(log)}] {entry.get('sku', '?')}")
                lines.append(f"   primary value:   {entry.get('primary_value') or '(empty)'}")
                lines.append(f"   fallback value:  {entry.get('fallback_value') or '(empty)'}")
                if entry.get("category_value"):
                    lines.append(f"   category value:  {entry.get('category_value')}")
                lines.append(f"   query used:      {entry.get('query_used') or '(none)'}")
                # Indicate which column actually triggered the winning query.
                # Strip surrounding quotes AND any appended category words so
                # the comparison works regardless of exact-match/category state.
                pv = (entry.get('primary_value') or '').strip().lower()
                fv = (entry.get('fallback_value') or '').strip().lower()
                cat = (entry.get('category_value') or '').strip().lower()
                qu_raw = (entry.get('query_used') or '').strip()
                qu = qu_raw.strip('"').lower()
                # Remove the category suffix (if any) for the comparison
                if cat and qu.endswith(" " + cat):
                    qu = qu[:-(len(cat) + 1)].rstrip().strip('"')
                if qu and qu == pv:
                    src = "primary"
                elif qu and qu == fv:
                    src = "fallback (primary returned no relevant results)"
                else:
                    src = "?"
                lines.append(f"   query source:    {src}")
                lines.append(f"   status:          {entry.get('status', '?')}")
                lines.append(f"   JSON-LD hit:     {'yes' if entry.get('jsonld_hit') else 'no'}")
                if entry.get("error_msg"):
                    lines.append(f"   error:           {entry['error_msg']}")
                lines.append("")

                hint = entry.get("jsonld_hint") or {}
                if hint:
                    lines.append("   JSON-LD hint passed to LLM:")
                    for k, v in hint.items():
                        lines.append(f"     {k}: {v}")
                    lines.append("")

                for j, page in enumerate(entry.get("pages", []), 1):
                    lines.append(f"   Source {j}: {page.get('url', '?')}")
                    lines.append(f"     chars:     {page.get('cleaned_chars', 0):,}")
                    lines.append(f"     validates: {'yes' if page.get('validates') else 'no'}")
                    lines.append("     content:")
                    text = page.get("cleaned_text", "") or ""
                    for tline in text.splitlines():
                        lines.append(f"       {tline}")
                    lines.append("")
                lines.append("")

            # Error log section
            if error_log:
                lines.append("=" * 70)
                lines.append("ERROR LOG")
                lines.append("=" * 70)
                for e in error_log:
                    lines.append(f"  [{e.get('kind', 'error')}] {e.get('sku', '?')}")
                    lines.append(f"    {e.get('error', '')}")
                    lines.append("")

            return "\n".join(lines)

        debug_txt = _build_debug_txt(
            debug_log,
            st.session_state.get("error_log", []),
            st.session_state.get("run_settings", {}),
        )
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        col_caption, col_download = st.columns([3, 1])
        with col_caption:
            st.caption(
                f"{len(debug_log)} products captured · "
                f"Toggle **Debug mode** in the sidebar off to disable logging on future runs"
            )
        with col_download:
            st.download_button(
                "⬇️ Download as .txt",
                data=debug_txt.encode("utf-8"),
                file_name=f"debug_log_{ts}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        for entry in debug_log:
            sku        = entry["sku"]
            status     = entry["status"]
            jsonld_hit = entry["jsonld_hit"]
            pages      = entry["pages"]
            hint       = entry.get("jsonld_hint", {})
            query      = entry.get("query_used", "")

            status_icon = {"success": "✅", "review": "⚠️", "error": "❌",
                           "not_found": "🔍", "rate_limited": "🚫"}.get(status, "•")
            jsonld_icon = "🏷️ JSON-LD hit" if jsonld_hit else "⬜ No JSON-LD"
            total_chars = sum(p["cleaned_chars"] for p in pages)

            with st.expander(
                f"{status_icon} **{sku}** — {len(pages)} page(s) · {total_chars:,} chars · {jsonld_icon}",
                expanded=False,
            ):
                if entry.get("error_msg"):
                    st.error(f"**LLM error:** {entry['error_msg']}")

                # Show all column values + which one the winning query came from
                pv  = (entry.get("primary_value")  or "").strip()
                fv  = (entry.get("fallback_value") or "").strip()
                cat = (entry.get("category_value") or "").strip()
                qu_raw = (query or "").strip()
                qu = qu_raw.strip('"').lower()
                # Strip the appended category (if any) from the query for the
                # source-attribution comparison.
                if cat and qu.endswith(" " + cat.lower()):
                    qu = qu[:-(len(cat) + 1)].rstrip().strip('"')
                if qu and qu == pv.lower():
                    fired = "**primary** column"
                elif qu and qu == fv.lower():
                    fired = "**fallback** column (primary returned no relevant matches)"
                else:
                    fired = "—"

                value_line = (
                    f"**Primary value:** `{pv or '(empty)'}` &nbsp;·&nbsp; "
                    f"**Fallback value:** `{fv or '(empty)'}`"
                )
                if cat:
                    value_line += f" &nbsp;·&nbsp; **Category hint:** `{cat}`"
                value_line += f" &nbsp;·&nbsp; **Query fired from:** {fired}"
                st.markdown(value_line)

                if query:
                    st.markdown(f"**Search query used:** `{query}`")
                if hint:
                    st.markdown("**JSON-LD hint injected into prompt:**")
                    st.json(hint, expanded=False)

                for i, page in enumerate(pages, 1):
                    url       = page["url"]
                    chars     = page["cleaned_chars"]
                    text      = page["cleaned_text"]
                    # Backward-compat with old debug logs: the field was
                    # renamed from `validates` to `overlaps_identifier` when
                    # the validation flag became the LOW_CONFIDENCE flag.
                    overlaps  = page.get("overlaps_identifier",
                                         page.get("validates", True))
                    max_c     = max_chars

                    pct_used = min(100, round(chars / max_c * 100)) if max_c else 100
                    bar_fill = "▓" * (pct_used // 5) + "░" * (20 - pct_used // 5)

                    overlaps_label = (
                        "✓ identifier tokens found in content"
                        if overlaps
                        else "⚠️ no token overlap — flagged LOW_CONFIDENCE"
                    )

                    st.markdown(
                        f"**Source {i}:** [{url}]({url})  \n"
                        f"`{chars:,} / {max_c:,} chars [{bar_fill}] {pct_used}%`  \n"
                        f"{overlaps_label}"
                    )
                    if chars == max_c:
                        st.warning(
                            "⚠️ Content was truncated at the character limit — "
                            "specs or key details may have been cut. Consider increasing Max chars per page.",
                        )
                    # IMPORTANT: do NOT pass a `key=` here. With a key, Streamlit
                    # binds the text_area's value to that key in session_state
                    # and the `value=` arg becomes only the initial default —
                    # subsequent renders show the cached state instead. That
                    # caused a real bug: when the same SKU was scraped twice
                    # in one session (re-run without Clear, or rate-limit
                    # retry), the second scrape's clean content went into
                    # debug_log[*][pages][*][cleaned_text] correctly, the
                    # downloaded debug log showed it, but the UI text_area
                    # kept rendering the FIRST scrape's privacy-policy noise
                    # because it was cached under the stable key. Without a
                    # key, widget identity is derived from the args (including
                    # `value=`) so a content change forces a fresh widget.
                    st.text_area(
                        f"Cleaned text — Source {i}",
                        value=text,
                        height=280,
                        disabled=True,
                        label_visibility="collapsed",
                    )

elif not is_running:
    st.markdown("""
    <div style="background:#12141a;border:1px dashed #2a2d3e;border-radius:12px;
                padding:3rem;text-align:center;color:#4a4f6a;margin-top:2rem;">
        <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
        <div style="font-size:1.1rem;font-weight:600">No results yet</div>
        <div style="font-size:0.9rem;margin-top:0.5rem">
            Upload a file, configure your settings, and click Start Enrichment
        </div>
    </div>
    """, unsafe_allow_html=True)
