"""
app.py — SKU Bulk Product Enrichment
Streamlit frontend with concurrent batch processing and JSON-LD fast path.
"""

import math
import time
import streamlit as st
import pandas as pd

from src.jina_client  import JinaConfig, COUNTRY_PARAMS
from src.extractors   import LLMConfig, OPENAI_MODELS, GEMINI_MODELS, CLAUDE_MODELS
from src.file_handler import read_file, to_csv_bytes, to_xlsx_bytes, build_fieldnames
from src.pipeline     import process_batch

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

  /* ── Sliders ── */
  /* Track */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
    background: #4f6ef7 !important;
  }
  /* Thumb */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #4f6ef7 !important;
    border-color: #4f6ef7 !important;
    box-shadow: 0 0 0 4px rgba(79,110,247,0.2) !important;
  }
  /* Value label above thumb */
  [data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #4f6ef7 !important;
    font-size: 12px !important;
  }
  /* Min/max tick labels */
  [data-testid="stSlider"] [data-testid="stTickBarMin"],
  [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #4a4f6a !important;
  }

  /* ── Buttons ── */
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

  /* ── Metrics ── */
  [data-testid="stMetric"] { background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 10px; padding: 1rem; }
  [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #4f6ef7; }

  /* ── Progress bar ── */
  .stProgress > div > div { background: linear-gradient(90deg, #4f6ef7, #7c4ff7); border-radius: 4px; }
  hr { border-color: #1e2130; }

  /* ── Expander — use current Streamlit testid selectors, not deprecated class names ── */
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
  [data-testid="stExpander"] summary:hover {
    background: #20243a !important;
  }
  /* ── Sidebar section labels ── */
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
    "results":        [],
    "running":        False,
    "stop_flag":      False,
    # Pause/stop use a queue-based approach (not a flag) so that Streamlit's
    # ScriptControlException (thrown when a button is clicked mid-run) naturally
    # preserves state. remaining_items is updated after EVERY batch so the queue
    # always reflects exactly what is left, even if the thread is killed abruptly.
    "paused":           False,
    "remaining_items":  [],   # live queue updated after every batch
    "resume_items":     [],   # snapshot taken at pause time for resume
    "resume_total":     0,
    "sku_column":       "",
    "selected_fields":  [],
    "run_export_fields": [],  # locked at job start — immune to post-run UI changes
    "debug_log":         [],  # list of {sku, pages:[{url, cleaned_chars, cleaned_text}], jsonld_hint}
    "input_rows":     [],
    "input_columns":  [],
    "fieldnames":     [],
    "stats":          {"success": 0, "review": 0, "not_found": 0,
                       "error": 0, "rate_limited": 0, "jsonld_hint": 0},
    "rate_limit_hit": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

is_running = st.session_state["running"]
is_paused  = st.session_state["paused"]

# key_label must always be defined before it can be referenced anywhere
# (including the "To start" caption which runs before sidebar provider selection)
key_label = "API Key"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 SKU Enrichment")
    st.markdown("---")
    st.markdown("## ⚙️ Configuration")
    if is_running:
        st.warning("⏳ Job running — settings locked", icon="🔒")

    # ── LLM ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">LLM Extraction Engine</div>', unsafe_allow_html=True)

    provider = st.selectbox("Provider", ["OpenAI", "Gemini", "Claude"], disabled=is_running)
    provider_key = provider.lower()

    if provider_key == "openai":
        model_list = OPENAI_MODELS;  key_label = "OpenAI API Key";  key_help = "platform.openai.com — gpt-4o-mini recommended for large batches"
    elif provider_key == "gemini":
        model_list = GEMINI_MODELS;  key_label = "Gemini API Key";  key_help = "aistudio.google.com — gemini-2.0-flash cheapest for large batches"
    else:
        model_list = CLAUDE_MODELS;  key_label = "Claude API Key";  key_help = "console.anthropic.com — claude-haiku recommended for large batches"

    llm_api_key = st.text_input(key_label, type="password", help=key_help, disabled=is_running)
    llm_model   = st.selectbox("Model", model_list, disabled=is_running)

    # ── Jina ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Jina AI Settings</div>', unsafe_allow_html=True)

    jina_api_key = st.text_input("Jina API Key", type="password", disabled=is_running,
                                  help="Recommended for large batches. Free at jina.ai/reader")

    country_code = st.selectbox(
        "Search Country",
        options=list(COUNTRY_PARAMS.keys()), index=0, disabled=is_running,
        format_func=lambda c: {"AU":"🇦🇺  Australia","US":"🇺🇸  United States",
            "UK":"🇬🇧  United Kingdom","NZ":"🇳🇿  New Zealand","CA":"🇨🇦  Canada",
            "DE":"🇩🇪  Germany","FR":"🇫🇷  France","JP":"🇯🇵  Japan",
            "SG":"🇸🇬  Singapore","IN":"🇮🇳  India"}.get(c, c),
        help="Country targeting via Google — does not add country keywords to search query",
    )

    urls_per_sku = st.slider("URLs per SKU", 1, 5, 2, disabled=is_running,
                              help="Number of product pages to fetch and cross-reference per SKU")

    with st.expander("Advanced Jina Settings"):
        max_chars = st.slider("Max chars per page (after cleaning)", 2000, 12000, 4000, step=500,
                               disabled=is_running,
                               help="Applied after nav/footer stripping — 4,000 is sufficient for most product pages")
        timeout = st.slider("Request timeout (seconds)", 10, 60, 25, disabled=is_running)
        return_format = st.selectbox("Page return format", ["markdown","text","html"], disabled=is_running,
                                      help="markdown recommended — cleanest for LLM extraction")
        no_cache = st.toggle("Bypass Jina cache", value=True, disabled=is_running)
        retry_on_few = st.toggle("Retry search if < 2 results", value=True, disabled=is_running)
        delay_between = st.slider("Delay between fetches (s)", 0.0, 3.0, 0.5, step=0.25,
                                   disabled=is_running)

        st.markdown("**DOM-level content targeting**")
        st.caption(
            "Applied by Jina _before_ converting to markdown — more reliable than "
            "post-hoc heuristics because they target real HTML elements. "
            "Leave blank to use automatic extraction."
        )
        target_selector = st.text_input(
            "Include only (CSS selector)",
            value="",
            placeholder="article, .product-description, #main-content",
            disabled=is_running,
            help=(
                "Jina returns ONLY the elements matching this selector. "
                "Best for sites where product content is in a consistent container. "
                "Example: 'article' for most blogs, '.product__description' for Shopify, "
                "'#product-details' for custom sites. Use the debug panel to check results."
            ),
        )
        remove_selector = st.text_input(
            "Remove (CSS selector)",
            value="nav, footer, header, .sidebar, #ads, .cookie-banner, .related-products",
            placeholder="nav, footer, .sidebar, #ads",
            disabled=is_running,
            help=(
                "Jina strips these elements from the DOM before extraction. "
                "Safe to leave at the default — it removes the most common junk. "
                "Can be used alongside 'Include only' (remove runs first)."
            ),
        )
        use_readerlm = st.toggle(
            "Use ReaderLM-v2",
            value=False,
            disabled=is_running,
            help=(
                "Uses Jina's dedicated small language model for HTML→markdown conversion. "
                "Significantly better on complex or heavily structured pages (tables, nested specs). "
                "Costs approximately 3× more Jina tokens per page — not recommended for large batches "
                "unless you're seeing poor content quality."
            ),
        )

    # ── Performance ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Performance & Rate Limiting</div>', unsafe_allow_html=True)

    max_workers = st.slider(
        "Concurrent workers",
        min_value=1, max_value=20, value=5, disabled=is_running,
        help=(
            "SKUs processed simultaneously. "
            "5–10 is the sweet spot for most setups. "
            "Higher = faster but more likely to hit rate limits."
        ),
    )

    delay_between_skus = st.slider(
        "Delay between batches (seconds)", 0, 10, 1, disabled=is_running,
        help="Pause between each concurrent batch — helps avoid rate limits at scale",
    )

    st.divider()
    debug_mode = st.toggle(
        "🔬 Debug mode",
        value=False,
        disabled=is_running,
        help=(
            "Stores the exact cleaned text sent to the LLM for each SKU. "
            "After a run, a debug panel appears in the results area showing "
            "character counts, JSON-LD hits, and the full text per page. "
            "Turn off for large batches — it stores all cleaned content in memory."
        ),
    )

    # Runtime estimate
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
        st.caption(f"⏱ Estimated runtime for {n:,} SKUs: **{est}** at {max_workers} workers")

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("# 🔍 SKU Product Enrichment")
st.markdown("Bulk enrich product codes with descriptions, specs and metadata — powered by Jina AI.")

with st.expander("ℹ️ How this tool works", expanded=False):
    st.markdown("""
<style>
  .help-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem 2rem; margin-top: 0.5rem; }
  .help-card { background: #12141a; border: 1px solid #1e2130; border-radius: 8px; padding: 0.75rem 1rem; }
  .help-card h4 { margin: 0 0 0.3rem 0; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: #4f6ef7; }
  .help-card p  { margin: 0; font-size: 0.82rem; color: #b0b4c8; line-height: 1.5; }
  .help-card .tag { display: inline-block; font-size: 0.7rem; font-weight: 700; border-radius: 4px; padding: 1px 6px; margin-right: 4px; margin-top: 0.35rem; }
  .pro  { background: rgba(79,180,115,0.15); color: #4fb473; }
  .con  { background: rgba(230,80,80,0.15);  color: #e65050; }
  .help-overview { font-size: 0.88rem; color: #b0b4c8; line-height: 1.6; margin-bottom: 0.75rem; }
  .help-section-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #4a4f6a; margin: 1rem 0 0.5rem 0; border-bottom: 1px solid #1e2130; padding-bottom: 0.3rem; }
</style>

<p class="help-overview">
  This tool takes a list of product SKUs from a CSV or Excel file and automatically fetches product pages from the web via <strong>Jina AI Reader</strong>, then uses an <strong>LLM</strong> (OpenAI, Gemini, or Claude) to extract structured fields like name, brand, description, specs, and more. Results are available to download as CSV or Excel.
</p>

<div class="help-section-label">Pipeline overview</div>
<div class="help-grid">
  <div class="help-card">
    <h4>1 · Search</h4>
    <p>Jina AI searches Google for each SKU and returns the top product page URLs. The <em>Search Country</em> setting targets results to a specific region without adding country keywords to the query.</p>
  </div>
  <div class="help-card">
    <h4>2 · Fetch &amp; Clean</h4>
    <p>Each URL is fetched and stripped of navigation, footers, and boilerplate. The <em>Page return format</em> controls the shape of the cleaned content passed to the LLM.</p>
  </div>
  <div class="help-card">
    <h4>3 · Extract</h4>
    <p>The LLM reads the cleaned page content and extracts only the fields you selected. If the page contains <strong>JSON-LD structured data</strong>, that is passed as a trusted hint to improve accuracy.</p>
  </div>
  <div class="help-card">
    <h4>4 · Review flags</h4>
    <p>Results with low confidence are automatically flagged for manual review. The <em>Review Flag</em> field in the output marks these rows so you can spot-check them quickly.</p>
  </div>
</div>

<div class="help-section-label">Output fields</div>
<div class="help-grid">
  <div class="help-card">
    <h4>Product Name</h4>
    <p>The full product title as found on the source page. This is the canonical name used by the manufacturer or retailer, not cleaned or shortened.</p>
  </div>
  <div class="help-card">
    <h4>Brand</h4>
    <p>The manufacturer or brand name extracted from the page. Useful for normalising supplier data where brand is missing or inconsistent in your source file.</p>
  </div>
  <div class="help-card">
    <h4>Short Description</h4>
    <p>A concise 1–2 sentence summary of what the product is. Suitable for catalogue listings, search results, or anywhere space is limited.</p>
  </div>
  <div class="help-card">
    <h4>Long Description</h4>
    <p>A fuller marketing-style description, typically 3–5 sentences. Drawn from the product page copy and intended for product detail pages or data sheets.</p>
  </div>
  <div class="help-card">
    <h4>Specifications</h4>
    <p>Key technical attributes — dimensions, weight, power rating, materials, compatibility, and so on — extracted from spec tables or bullet lists on the page.</p>
  </div>
  <div class="help-card">
    <h4>Category</h4>
    <p>The product category or breadcrumb path inferred from the page (e.g. <em>Power Tools &gt; Drills &gt; Hammer Drills</em>). Useful for taxonomy mapping.</p>
  </div>
  <div class="help-card">
    <h4>Model Number</h4>
    <p>The manufacturer's own model or part number found on the page. This may differ from your input SKU — if it does, it is worth investigating before importing.</p>
  </div>
  <div class="help-card">
    <h4>Barcode (EAN/UPC)</h4>
    <p>The EAN-13 or UPC-A barcode if visible on the page. Not always present — retailers often omit it. Blank does not mean the product has no barcode.</p>
  </div>
  <div class="help-card">
    <h4>Country of Origin</h4>
    <p>Where the product is manufactured, if stated. Useful for compliance, import declarations, or supplier auditing.</p>
  </div>
  <div class="help-card">
    <h4>Source URL</h4>
    <p>The exact URL the data was extracted from. Always include this field if you plan to audit results or report back to a supplier — it gives you a direct link to the evidence.</p>
  </div>
  <div class="help-card">
    <h4>Confidence Score</h4>
    <p>A 0–100 score the LLM assigns based on how well the fetched page matched your SKU. Above 80 is generally reliable. Below 60 usually means the page was a poor match and the data should be verified manually.</p>
  </div>
  <div class="help-card">
    <h4>Review Flag</h4>
    <p>Set to <strong>True</strong> when the LLM's confidence is low, the model number on the page conflicts with your SKU, or the page content was ambiguous. Use this column to filter rows for manual checking before importing results into your system.</p>
  </div>
</div>

<div class="help-section-label">Advanced settings</div>
<div class="help-grid">
  <div class="help-card">
    <h4>URLs per SKU</h4>
    <p>How many product pages are fetched and cross-referenced per SKU. More pages improve accuracy on ambiguous SKUs but increase cost and runtime.</p>
    <span class="tag pro">better accuracy</span><span class="tag con">higher cost and time</span>
  </div>
  <div class="help-card">
    <h4>Concurrent workers</h4>
    <p>Number of SKUs processed in parallel. Higher values finish large batches faster but increase the chance of hitting API rate limits on both Jina and your LLM provider.</p>
    <span class="tag pro">faster batches</span><span class="tag con">higher rate limit risk</span>
  </div>
  <div class="help-card">
    <h4>Max chars per page</h4>
    <p>Truncation limit applied after cleaning. Lower values reduce LLM token usage and cost; higher values preserve more context for complex or long product pages.</p>
    <span class="tag pro">lower cost at lower values</span><span class="tag con">lower accuracy if too small</span>
  </div>
  <div class="help-card">
    <h4>Page return format</h4>
    <p><em>Markdown</em> is the cleanest for LLM extraction and recommended for most cases. <em>Text</em> is plainer and slightly smaller. <em>HTML</em> preserves table structure but is verbose and uses more tokens.</p>
    <span class="tag pro">markdown is best default</span><span class="tag con">html uses more tokens</span>
  </div>
  <div class="help-card">
    <h4>Bypass Jina cache</h4>
    <p>Forces Jina to re-fetch live pages rather than serve a cached copy. Useful when product pages have recently changed. Turn off to speed up repeated runs on the same SKU list.</p>
    <span class="tag pro">always fresh data</span><span class="tag con">slower, more quota used</span>
  </div>
  <div class="help-card">
    <h4>Retry if fewer than 2 results</h4>
    <p>If the initial search returns only one URL, a second search is attempted with a varied query. Helps with obscure SKUs but adds latency per affected row.</p>
    <span class="tag pro">better coverage</span><span class="tag con">slower for rare SKUs</span>
  </div>
  <div class="help-card">
    <h4>Delay between fetches</h4>
    <p>Pause in seconds between each individual URL fetch within a SKU. A small delay reduces the chance of Jina throttling your requests when processing many URLs per SKU.</p>
    <span class="tag pro">reduces throttling</span><span class="tag con">increases per-SKU time</span>
  </div>
  <div class="help-card">
    <h4>Delay between batches</h4>
    <p>Pause between each concurrent worker batch. Gives both Jina and your LLM provider time to recover between bursts, especially useful for large jobs running overnight.</p>
    <span class="tag pro">reduces rate limit risk</span><span class="tag con">increases total runtime</span>
  </div>
  <div class="help-card">
    <h4>Include only (CSS selector)</h4>
    <p>Tells Jina to return <em>only</em> the DOM elements matching this CSS selector, before converting to markdown. Far more reliable than heuristic cleaning because it targets real HTML structure. Example: <code>article</code>, <code>.product-description</code>, <code>#main-content</code>. Leave blank to let Jina auto-extract.</p>
    <span class="tag pro">surgical nav removal</span><span class="tag con">needs site-specific selectors</span>
  </div>
  <div class="help-card">
    <h4>Remove (CSS selector)</h4>
    <p>Strips matching DOM elements before extraction — runs before the Include selector. The default covers the most common junk: <code>nav, footer, header, .sidebar, #ads</code>. Safe to leave at the default for most sites.</p>
    <span class="tag pro">removes nav at source</span><span class="tag con">wrong selectors may remove content</span>
  </div>
  <div class="help-card">
    <h4>Use ReaderLM-v2</h4>
    <p>Switches Jina to use its dedicated small language model for HTML-to-markdown conversion instead of the default rule-based approach. Produces significantly cleaner output on complex or heavily structured pages with spec tables and nested content.</p>
    <span class="tag pro">better on complex pages</span><span class="tag con">approx 3× Jina token cost</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

if st.session_state["rate_limit_hit"]:
    st.error("⚠️ **Rate limit hit.** Add/check your Jina API key or reduce concurrent workers. Results so far are safe to download.", icon="🚫")

# ── Upload & config ───────────────────────────────────────────────────────────

col_upload, col_config = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📂 Upload Product File")
    st.caption("⚠️ Excel files: raw cell values only — formulas, multiple sheets and formatting are not preserved in output.")

    uploaded = st.file_uploader("CSV or Excel file with SKU column",
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
                    f"ℹ️ **Large batch ({n:,} SKUs).** "
                    f"Estimated runtime at 5 workers: **{hrs_low}–{hrs_high} hrs**. "
                    f"Keep this tab active. For 5k+ SKUs run locally — Streamlit Cloud sessions may time out."
                )
        except Exception as e:
            st.error(f"Could not read file: {e}")
            rows, columns = [], []
    else:
        rows    = st.session_state["input_rows"]
        columns = st.session_state["input_columns"]

with col_config:
    st.markdown("### 🏷️ Column & Field Settings")

    if columns:
        sku_hints   = ["sku","product_code","code","part","model","item","mpn","id"]
        default_sku = next((i for i, c in enumerate(columns) if any(h in c.lower() for h in sku_hints)), 0)
        sku_column  = st.selectbox("SKU Column", columns, index=default_sku, disabled=is_running)

        secondary_options = ["— none —"] + [c for c in columns if c != sku_column]
        secondary_column  = st.selectbox(
            "Secondary search term column",
            secondary_options,
            disabled=is_running,
            help=(
                "Optional. Values from this column are appended to the Jina search query "
                "alongside the SKU — useful when SKUs are short or ambiguous (e.g. add a "
                "product name or brand column to narrow results). Leave as none if your "
                "SKUs are descriptive enough on their own."
            ),
        )
        secondary_column = None if secondary_column == "— none —" else secondary_column
    else:
        sku_column       = st.selectbox("SKU Column", ["— upload a file first —"], disabled=True)
        secondary_column = None
        st.selectbox("Secondary search term column", ["— upload a file first —"], disabled=True)

    st.session_state["sku_column"] = sku_column

    st.markdown("**Fields to extract**")
    field_options = {
        "product_name":      "Product Name",
        "brand":             "Brand",
        "short_description": "Short Description",
        "long_description":  "Long Description",
        "specifications":    "Specifications",
        "category":          "Category",
        "model_number":      "Model Number",
        "barcode":           "Barcode (EAN/UPC)",
        "country_of_origin": "Country of Origin",
        "source_url":        "Source URL",
        "confidence_score":  "Confidence Score",
        "review_flag":       "Review Flag",
    }
    default_checked = list(field_options.keys())[:6] + ["review_flag"]
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

can_start = bool(rows) and bool(llm_api_key) and bool(selected_fields) and sku_column != "— upload a file first —"

with c1:
    if is_paused:
        start_btn  = st.button("▶  Resume", disabled=not can_start)
    else:
        start_btn  = st.button("▶  Start Enrichment", disabled=not can_start or is_running)
with c2:
    pause_btn  = st.button("⏸  Pause",  disabled=not is_running)
with c3:
    stop_btn   = st.button("⏹  Stop",   disabled=not is_running and not is_paused)
with c4:
    clear_btn  = st.button("🗑  Clear",  disabled=is_running)

if is_paused:
    done_so_far = len(st.session_state["results"])
    remaining   = len(st.session_state["resume_items"])
    st.info(f"⏸ **Paused** — {done_so_far} SKUs done, {remaining} remaining. Adjust settings above if needed, then click Resume.")

if not can_start and not is_running and not is_paused:
    missing = []
    if not rows:            missing.append("upload a file")
    if not llm_api_key:     missing.append(f"enter {key_label}")
    if not selected_fields: missing.append("select at least one field")
    if missing:
        st.caption(f"⚠️  To start: {', '.join(missing)}")

if pause_btn:
    # Pause works by setting paused=True and running=False, then letting the
    # next Streamlit rerun naturally stop the loop. remaining_items is already
    # up-to-date because _run_pipeline writes it after every batch — so even
    # if Streamlit kills the thread mid-loop via ScriptControlException, the
    # queue correctly holds whatever was left at the end of the last completed
    # batch. No flag-checking inside the loop is needed or reliable.
    remaining = st.session_state.get("remaining_items", [])
    st.session_state["paused"]       = True
    st.session_state["running"]      = False
    st.session_state["resume_items"] = remaining
    st.rerun()

if stop_btn:
    st.session_state["running"]         = False
    st.session_state["paused"]          = False
    st.session_state["resume_items"]    = []
    st.session_state["remaining_items"] = []
    st.rerun()

if clear_btn:
    for k in ["results", "fieldnames", "resume_items", "remaining_items", "run_export_fields", "debug_log"]:
        st.session_state[k] = []
    st.session_state["stats"]          = {"success": 0, "review": 0, "not_found": 0, "error": 0, "rate_limited": 0, "jsonld_hint": 0}
    st.session_state["rate_limit_hit"] = False
    st.session_state["paused"]         = False
    st.session_state["resume_total"]   = 0
    st.rerun()

# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(work_items, total, fieldnames, jina_cfg, llm_cfg, max_workers, delay_between_skus):
    """
    Shared pipeline loop used by both fresh Start and Resume.

    Pause/Stop safety model
    -----------------------
    We do NOT check a pause_flag or stop_flag inside this loop. Streamlit
    button clicks throw a ScriptControlException that kills the running thread
    instantly — any flag set by the user will never be seen by a mid-loop
    check. Instead, we rely on the queue-based approach:

      * remaining_items is written to session_state AFTER every completed
        batch. If Streamlit kills the thread, the queue already holds exactly
        what is left from the last completed batch.
      * The Pause button handler reads remaining_items and copies it to
        resume_items, then sets running=False and triggers a rerun.
      * On the next page load, the loop simply does not start (running=False),
        and the paused UI is shown with an accurate resume queue.

    This means Pause always loses at most one in-flight batch (the one that
    was executing when the button was clicked), which is unavoidable in any
    synchronous Streamlit app and is clearly communicated to the user.
    """
    n_batches = math.ceil(len(work_items) / max_workers)
    completed = len(st.session_state["results"])  # already done before this run

    progress_bar = st.progress(0, text="Starting…")
    status_box   = st.empty()
    table_box    = st.empty()

    for batch_idx in range(n_batches):
        batch_start  = batch_idx * max_workers
        batch_end    = min(batch_start + max_workers, len(work_items))
        batch        = work_items[batch_start:batch_end]
        global_done  = completed + batch_start

        skus_label = ", ".join(s for s, _, __ in batch[:3])
        if len(batch) > 3:
            skus_label += f" +{len(batch)-3} more"
        pct = global_done / max(total, 1)
        progress_bar.progress(
            pct,
            text=f"Processing **{skus_label}** ({global_done + 1}–{global_done + len(batch)}/{total})"
        )

        batch_results = process_batch(batch, selected_fields, jina_cfg, llm_cfg, max_workers=max_workers, debug=debug_mode)

        rate_limited = False
        for result in batch_results:
            s = st.session_state["stats"]
            if result.status == "success":
                s["success"] += 1
            elif result.status == "review":
                s["review"] += 1
            elif result.status == "rate_limited":
                s["rate_limited"] += 1
                st.session_state["rate_limit_hit"] = True
                rate_limited = True
                status_box.error(f"🚫 Rate limit hit on **{result.sku}**. Results so far are safe to download.")
            elif result.status in ("not_found", "blocked"):
                s["not_found"] += 1
            else:
                s["error"] += 1

            if result.had_jsonld:
                s["jsonld_hint"] += 1

            st.session_state["results"].append(result.data)

            if debug_mode and result.debug_pages:
                st.session_state["debug_log"].append({
                    "sku":        result.sku,
                    "status":     result.status,
                    "pages":      result.debug_pages,
                    "jsonld_hit": result.had_jsonld,
                    "jsonld_hint": result.jsonld_hint,
                })

        # ── Update the queue AFTER every batch ──────────────────────────────
        # This is the key safety write. If Streamlit kills the thread on the
        # next button click, this value is already in session_state and the
        # Pause/Stop handlers can read it without needing any flag.
        st.session_state["remaining_items"] = work_items[batch_end:]

        preview_df   = pd.DataFrame(st.session_state["results"][-20:])
        preview_cols = [c for c in fieldnames if c in preview_df.columns]
        table_box.dataframe(
            preview_df[preview_cols] if preview_cols else preview_df,
            use_container_width=True, height=280,
        )

        if rate_limited:
            break

        time.sleep(delay_between_skus)

    done = len(st.session_state["results"])
    all_done = (done - completed) >= len(work_items)
    if all_done and not st.session_state.get("rate_limit_hit"):
        progress_bar.progress(1.0, text="✅ Complete!")
    else:
        progress_bar.progress(done / max(total, 1), text=f"⏹ Stopped at {done}/{total} SKUs")

    st.session_state["running"]         = False
    st.session_state["paused"]          = False
    st.session_state["resume_items"]    = []
    st.session_state["remaining_items"] = []
    st.rerun()


if start_btn and not is_paused:
    # Fresh start — reset everything
    st.session_state.update({
        "running":          True,
        "stop_flag":        False,
        "paused":           False,
        "resume_items":     [],
        "remaining_items":  [],
        "results":          [],
        "debug_log":        [],
        "rate_limit_hit":   False,
        "stats":            {"success": 0, "review": 0, "not_found": 0, "error": 0, "rate_limited": 0, "jsonld_hint": 0},
    })

    jina_cfg = JinaConfig(
        api_key=jina_api_key, country_code=country_code,
        urls_per_sku=urls_per_sku, max_chars=max_chars,
        timeout=timeout, no_cache=no_cache,
        return_format=return_format, retry_on_few=retry_on_few,
        delay_between=delay_between,
        target_selector=target_selector,
        remove_selector=remove_selector,
        use_readerlm=use_readerlm,
    )
    llm_cfg = LLMConfig(provider=provider_key, api_key=llm_api_key, model=llm_model)

    fieldnames = build_fieldnames(columns, selected_fields)
    st.session_state["fieldnames"]        = fieldnames
    st.session_state["selected_fields"]   = selected_fields
    # Lock the export config at start time so post-run UI changes (e.g. the
    # user fiddling with the SKU column dropdown while viewing results) cannot
    # corrupt the download with KeyErrors or mismatched columns.
    st.session_state["run_export_fields"] = fieldnames
    st.session_state["sku_column"]        = sku_column

    work_items = [
        (
            str(row.get(sku_column, "") or "").strip(),
            str(row.get(secondary_column, "") or "").strip() if secondary_column else "",
            row,
        )
        for row in rows
        if str(row.get(sku_column, "") or "").strip()
    ]
    total = len(work_items)
    st.session_state["resume_total"] = total

    _run_pipeline(work_items, total, fieldnames, jina_cfg, llm_cfg, max_workers, delay_between_skus)


if start_btn and is_paused:
    # Resume from saved position
    work_items = st.session_state["resume_items"]
    total      = st.session_state["resume_total"]
    fieldnames = st.session_state["fieldnames"]

    jina_cfg = JinaConfig(
        api_key=jina_api_key, country_code=country_code,
        urls_per_sku=urls_per_sku, max_chars=max_chars,
        timeout=timeout, no_cache=no_cache,
        return_format=return_format, retry_on_few=retry_on_few,
        delay_between=delay_between,
        target_selector=target_selector,
        remove_selector=remove_selector,
        use_readerlm=use_readerlm,
    )
    llm_cfg = LLMConfig(provider=provider_key, api_key=llm_api_key, model=llm_model)

    st.session_state.update({
        "running":         True,
        "paused":          False,
        "stop_flag":       False,
        "remaining_items": list(work_items),  # seed queue for this resume leg
    })

    _run_pipeline(work_items, total, fieldnames, jina_cfg, llm_cfg, max_workers, delay_between_skus)

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
    m3.metric("📐 JSON-LD Hint",  stats["jsonld_hint"], help="SKUs where structured page data was found and passed to the LLM as a trusted starting point")
    m4.metric("⚠️ Review",        stats["review"])
    m5.metric("❌ Not Found",     stats["not_found"])
    m6.metric("🚫 Rate Limited",  stats["rate_limited"])
    m7.metric("💥 Error",         stats["error"])

    if stats["jsonld_hint"] > 0:
        st.caption(f"📐 **{jsonld_pct}%** of SKUs had JSON-LD structured data — passed as trusted metadata to the LLM for higher accuracy.")

    st.divider()

    fieldnames = st.session_state["fieldnames"]
    df         = pd.DataFrame(st.session_state["results"])
    show_cols  = [c for c in fieldnames if c in df.columns]
    st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, height=400)

    st.markdown("### ⬇️ Download")

    # ── Export mode ──────────────────────────────────────────────────────────
    export_mode = st.radio(
        "Export format",
        ["Combined — original columns + enriched fields",
         "Enriched fields only — SKU + selected output fields"],
        horizontal=True,
        help=(
            "Combined keeps all your original input columns and appends the enriched fields. "
            "Enriched only outputs just the SKU column plus the fields you selected — "
            "useful for importing into a separate system or joining back later."
        ),
    )

    all_rows = st.session_state["results"]

    # Use the field list that was locked at job-start time. This ensures that
    # if the user changes the SKU column dropdown or unchecks a field while
    # browsing results, the download is not corrupted with KeyErrors or
    # mismatched columns from the live (post-run) UI state.
    _locked_fields  = st.session_state.get("run_export_fields") or fieldnames
    _locked_sku_col = st.session_state.get("sku_column", "")

    if export_mode.startswith("Enriched"):
        # SKU column first, then output fields only — no original input columns.
        _enriched_cols = []
        _seen = set()
        for _c in ([_locked_sku_col] if _locked_sku_col else []) + list(_locked_fields):
            if _c and _c not in _seen:
                _enriched_cols.append(_c)
                _seen.add(_c)
        export_fieldnames = _enriched_cols or _locked_fields
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

    # ── Debug panel ──────────────────────────────────────────────────────────
    debug_log = st.session_state.get("debug_log", [])
    if debug_log:
        st.divider()
        st.markdown("### 🔬 Debug Log")
        st.caption(
            f"{len(debug_log)} SKUs captured · "
            f"Toggle **Debug mode** in the sidebar off to disable logging on future runs"
        )

        for entry in debug_log:
            sku        = entry["sku"]
            status     = entry["status"]
            jsonld_hit = entry["jsonld_hit"]
            pages      = entry["pages"]
            hint       = entry.get("jsonld_hint", {})

            status_icon = {"success": "✅", "review": "⚠️", "error": "❌",
                           "not_found": "🔍", "rate_limited": "🚫"}.get(status, "•")
            jsonld_icon = "🏷️ JSON-LD hit" if jsonld_hit else "⬜ No JSON-LD"
            total_chars = sum(p["cleaned_chars"] for p in pages)

            with st.expander(
                f"{status_icon} **{sku}** — {len(pages)} page(s) · {total_chars:,} chars analysed · {jsonld_icon}",
                expanded=False,
            ):
                if hint:
                    st.markdown("**JSON-LD hint injected into prompt:**")
                    st.json(hint, expanded=False)

                for i, page in enumerate(pages, 1):
                    url    = page["url"]
                    chars  = page["cleaned_chars"]
                    text   = page["cleaned_text"]
                    max_c  = max_chars  # from sidebar

                    pct_used = min(100, round(chars / max_c * 100)) if max_c else 100
                    bar_fill = "▓" * (pct_used // 5) + "░" * (20 - pct_used // 5)

                    st.markdown(
                        f"**Source {i}:** [{url}]({url})  \n"
                        f"`{chars:,} / {max_c:,} chars used  [{bar_fill}] {pct_used}%`"
                    )
                    if chars == max_c:
                        st.warning(
                            "⚠️ Content was truncated at the character limit — "
                            "specs or key details may have been cut. Consider increasing Max chars per page.",
                            icon=None,
                        )
                    st.text_area(
                        f"Cleaned text — Source {i}",
                        value=text,
                        height=280,
                        key=f"debug_{sku}_{i}",
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
