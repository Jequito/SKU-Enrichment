"""
app.py — SKU Bulk Product Enrichment
Streamlit frontend. Searches DuckDuckGo (no API key), fetches via httpx +
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
}
for k, v in defaults.items():
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
        help="Get a key at serpapi.com/manage-api-key. Free tier is 100 searches/month.",
    )
    st.caption(
        "💰 SerpAPI free tier: **100 searches/month** (~50 SKUs). "
        "Paid plans start at **$75/month** for 5,000 searches (~2,500 SKUs). "
        "Each row uses 1 query when the primary column finds a match, 2 when it falls back."
    )

    country_code = st.selectbox(
        "Search Region",
        options=list(SERPAPI_COUNTRIES.keys()), index=0, disabled=is_running,
        format_func=lambda c: COUNTRY_LABELS.get(c, c),
        help="Passed to Google as the gl parameter — biases results to that locale.",
    )

    urls_per_sku = st.slider("URLs per product", 1, 5, 2, disabled=is_running,
                              help="Number of top-scored URLs to fetch and pass to the LLM. With exact-match Google searches the top hit is usually correct — 1 is often enough.")

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
        "🔬 Debug mode", value=False, disabled=is_running,
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

#### Two-column exact-match search

For each row, the tool runs an exact-quoted Google search on your **Primary**
column. If that returns zero results, it retries on your **Fallback** column.
That's it — no wide-net fallback queries that risk pulling in pages about
different products.

| Outcome | SerpAPI queries used |
|---|---|
| Primary search finds something | 1 |
| Primary returns nothing, fallback finds something | 2 |
| Both return nothing | 1 (only the primary is charged; empty results are free) |

#### Validation

When pages are fetched, the tool checks that the Primary or Fallback value
appears in the page content. If neither does, the row is flagged
`REVIEW_NEEDED` because Google may have surfaced a different product.

#### JSON-LD hints

When a page contains `<script type="application/ld+json">` Product schema, the
structured data is passed to the LLM as a trusted starting point. This produces
significantly more accurate results on retailer and manufacturer sites.
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
    st.caption("Map both columns. Primary is searched first, Fallback only runs if the primary search returns nothing.")

    if columns:
        # Auto-detect: manufacturer/product code typically goes in Primary,
        # internal SKU in Fallback. User can flip either dropdown.
        primary_hints  = ["product_code","mpn","model","manufacturer","mfr","part_no","part_number"]
        fallback_hints = ["sku","item","id","code"]

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
            help="Searched first as an exact-quoted Google query. Typically your Product Code / MPN.",
        )

        fallback_options = [c for c in columns if c != primary_column]
        fallback_column = st.selectbox(
            "Fallback search column (required)",
            fallback_options,
            index=_auto_index(fallback_options, fallback_hints),
            disabled=is_running,
            help="Searched only if the primary returns zero results. Typically your internal SKU.",
        )
    else:
        primary_column  = "— upload a file first —"
        fallback_column = "— upload a file first —"
        st.selectbox("Primary search column (required)",  ["— upload a file first —"], disabled=True)
        st.selectbox("Fallback search column (required)", ["— upload a file first —"], disabled=True)

    # The download/export logic uses sku_column as the row label — point it at
    # the primary column so the existing export plumbing keeps working unchanged.
    st.session_state["sku_column"] = primary_column

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

valid_columns = bool(columns) and primary_column != "— upload a file first —" and fallback_column != "— upload a file first —"

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
    if not valid_columns:      missing.append("map both Primary and Fallback columns")
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
    # Fresh start — reset everything
    work_items = []
    for row in rows:
        ids = IdentifierSet(
            primary  = str(row.get(primary_column, "")  or "").strip(),
            fallback = str(row.get(fallback_column, "") or "").strip(),
        )
        if not ids.is_empty():
            work_items.append((ids, row))

    fieldnames = build_fieldnames(columns, selected_fields)

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
    search_cfg = SearchConfig(
        serpapi_api_key = serpapi_api_key,
        country_code    = country_code,
        urls_per_sku    = urls_per_sku,
        max_chars       = max_chars,
        timeout         = timeout,
        max_results     = max_results,
        delay_between   = delay_between,
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
            status_box.error(f"🚫 Rate limit hit on **{result.sku}**. Pausing — results so far are safe to download.")
        elif result.status in ("not_found", "blocked"):
            s["not_found"] += 1
        else:
            s["error"] += 1
            if result.error_msg:
                batch_errors.append((result.sku, result.error_msg))
                # Persist for the post-run summary panel
                st.session_state.setdefault("error_log", []).append(
                    {"sku": result.sku, "error": result.error_msg}
                )

        if result.had_jsonld:
            s["jsonld_hint"] += 1

        st.session_state["results"].append(result.data)

        if debug_mode and result.debug_pages:
            st.session_state["debug_log"].append({
                "sku":         result.sku,
                "status":      result.status,
                "pages":       result.debug_pages,
                "jsonld_hit":  result.had_jsonld,
                "jsonld_hint": result.jsonld_hint,
                "query_used":  result.query_used,
                "error_msg":   result.error_msg,
            })

    st.session_state["completed_count"] += len(batch)

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

    # ── Error diagnostic panel ──────────────────────────────────────────────
    # Surfaces the actual provider-side messages so the user can see why rows
    # came back as ERROR — typically auth failures, invalid model strings,
    # quota issues, or content-filter blocks.
    error_log = st.session_state.get("error_log", [])
    if error_log:
        # Group by message so a hundred identical "invalid api key" errors
        # don't drown out the one row that failed for a different reason.
        from collections import Counter
        msg_counts = Counter(e["error"][:400] for e in error_log)

        with st.expander(
            f"💥 **{len(error_log)} extraction error(s)** — click to see what went wrong",
            expanded=True,
        ):
            st.caption(
                "These are messages returned by your LLM provider when extraction failed. "
                "Most commonly: invalid API key, invalid model identifier, or rate/quota limits."
            )
            for msg, count in msg_counts.most_common():
                # Show one representative SKU for each distinct error message
                example_sku = next(
                    (e["sku"] for e in error_log if e["error"][:400] == msg),
                    "?"
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

    _locked_fields  = st.session_state.get("run_export_fields") or fieldnames
    _locked_sku_col = st.session_state.get("sku_column", "")

    if export_mode.startswith("Enriched"):
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

    # Debug panel
    debug_log = st.session_state.get("debug_log", [])
    if debug_log:
        st.divider()
        st.markdown("### 🔬 Debug Log")
        st.caption(
            f"{len(debug_log)} products captured · "
            f"Toggle **Debug mode** in the sidebar off to disable logging on future runs"
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
                if query:
                    st.markdown(f"**Search query used:** `{query}`")
                if hint:
                    st.markdown("**JSON-LD hint injected into prompt:**")
                    st.json(hint, expanded=False)

                for i, page in enumerate(pages, 1):
                    url       = page["url"]
                    chars     = page["cleaned_chars"]
                    text      = page["cleaned_text"]
                    validates = page.get("validates", True)
                    max_c     = max_chars

                    pct_used = min(100, round(chars / max_c * 100)) if max_c else 100
                    bar_fill = "▓" * (pct_used // 5) + "░" * (20 - pct_used // 5)

                    validates_label = "✓ identifier matched in content" if validates else "⚠️ no identifier match — possible wrong product"

                    st.markdown(
                        f"**Source {i}:** [{url}]({url})  \n"
                        f"`{chars:,} / {max_c:,} chars [{bar_fill}] {pct_used}%`  \n"
                        f"{validates_label}"
                    )
                    if chars == max_c:
                        st.warning(
                            "⚠️ Content was truncated at the character limit — "
                            "specs or key details may have been cut. Consider increasing Max chars per page.",
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
