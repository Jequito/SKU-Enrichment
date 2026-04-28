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
    page_title="SKU Enrichment",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .stApp { background-color: #0d0f14; color: #e8e8e8; }

  [data-testid="stSidebar"] {
    background-color: #12141a;
    border-right: 1px solid #1e2130;
  }
  [data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

  h1 { font-weight: 800; letter-spacing: -0.03em; color: #ffffff; }
  h2, h3 { font-weight: 700; color: #e0e0e0; }

  .stTextInput input, .stSelectbox select, .stNumberInput input {
    background: #1a1d27 !important; border: 1px solid #2a2d3e !important;
    color: #e8e8e8 !important; font-family: 'DM Mono', monospace !important;
    border-radius: 6px !important;
  }
  .stButton > button {
    background: linear-gradient(135deg, #4f6ef7, #7c4ff7);
    color: white; border: none; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 15px; padding: 0.6rem 2rem;
    transition: all 0.2s ease; width: 100%;
  }
  .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(79,110,247,0.4); }
  .stButton > button:disabled { background: #2a2d3e !important; color: #4a4f6a !important; transform: none !important; }

  .stDownloadButton > button {
    background: #1a1d27; color: #4f6ef7;
    border: 1px solid #4f6ef7; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 600;
  }
  .stDownloadButton > button:hover { background: #4f6ef7; color: white; }

  [data-testid="stMetric"] { background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 10px; padding: 1rem; }
  [data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; font-size: 1.8rem !important; color: #4f6ef7; }
  .stProgress > div > div { background: linear-gradient(90deg, #4f6ef7, #7c4ff7); border-radius: 4px; }
  hr { border-color: #1e2130; }
  .streamlit-expanderHeader { background: #1a1d27 !important; border-radius: 6px !important; color: #e8e8e8 !important; }

  .sidebar-section {
    font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4f6ef7;
    margin: 1.5rem 0 0.5rem 0; padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2130;
  }
  code { font-family: 'DM Mono', monospace; background: #1a1d27; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

defaults = {
    "results":        [],
    "running":        False,
    "stop_flag":      False,
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

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
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
    else:
        sku_column = st.selectbox("SKU Column", ["— upload a file first —"], disabled=True)

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
c1, c2, c3 = st.columns([2, 1, 1])

can_start = bool(rows) and bool(llm_api_key) and bool(selected_fields) and sku_column != "— upload a file first —"

with c1:
    start_btn = st.button("▶  Start Enrichment", disabled=not can_start or is_running)
with c2:
    stop_btn  = st.button("⏹  Stop", disabled=not is_running)
with c3:
    clear_btn = st.button("🗑  Clear Results", disabled=is_running)

if not can_start and not is_running:
    missing = []
    if not rows:            missing.append("upload a file")
    if not llm_api_key:     missing.append(f"enter {key_label}")
    if not selected_fields: missing.append("select at least one field")
    if missing:
        st.caption(f"⚠️  To start: {', '.join(missing)}")

if stop_btn:
    st.session_state["stop_flag"] = True

if clear_btn:
    for k in ["results", "fieldnames"]:
        st.session_state[k] = []
    st.session_state["stats"]          = {"success": 0, "review": 0, "not_found": 0, "error": 0, "rate_limited": 0, "jsonld_hint": 0}
    st.session_state["rate_limit_hit"] = False
    st.rerun()

# ── Pipeline ──────────────────────────────────────────────────────────────────

if start_btn:
    st.session_state.update({
        "running":        True,
        "stop_flag":      False,
        "results":        [],
        "rate_limit_hit": False,
        "stats":          {"success": 0, "review": 0, "not_found": 0, "error": 0, "rate_limited": 0, "jsonld_hint": 0},
    })

    jina_cfg = JinaConfig(
        api_key=jina_api_key, country_code=country_code,
        urls_per_sku=urls_per_sku, max_chars=max_chars,
        timeout=timeout, no_cache=no_cache,
        return_format=return_format, retry_on_few=retry_on_few,
        delay_between=delay_between,
    )
    llm_cfg = LLMConfig(provider=provider_key, api_key=llm_api_key, model=llm_model)

    fieldnames = build_fieldnames(columns, selected_fields)
    st.session_state["fieldnames"] = fieldnames

    # Build work items — filter empty SKUs
    work_items = [
        (str(row.get(sku_column, "") or "").strip(), row)
        for row in rows
        if str(row.get(sku_column, "") or "").strip()
    ]
    total      = len(work_items)
    n_batches  = math.ceil(total / max_workers)

    progress_bar = st.progress(0, text="Starting…")
    status_box   = st.empty()
    table_box    = st.empty()

    for batch_idx in range(n_batches):
        if st.session_state["stop_flag"]:
            status_box.warning("⏹ Stopped by user.")
            break

        batch_start = batch_idx * max_workers
        batch_end   = min(batch_start + max_workers, total)
        batch       = work_items[batch_start:batch_end]

        # Label for progress bar
        skus_label = ", ".join(s for s, _ in batch[:3])
        if len(batch) > 3:
            skus_label += f" +{len(batch)-3} more"
        pct = batch_start / total
        progress_bar.progress(pct, text=f"Batch {batch_idx+1}/{n_batches} — **{skus_label}** ({batch_start+1}–{batch_end}/{total})")

        # Process batch concurrently
        batch_results = process_batch(batch, selected_fields, jina_cfg, llm_cfg, max_workers=max_workers)

        # Update stats and results
        for result in batch_results:
            s = st.session_state["stats"]
            if result.status == "success":
                s["success"] += 1
            elif result.status == "review":
                s["review"] += 1
            elif result.status == "rate_limited":
                s["rate_limited"] += 1
                st.session_state["rate_limit_hit"] = True
                st.session_state["stop_flag"]      = True
                status_box.error(f"🚫 Rate limit hit on **{result.sku}**. Results so far are safe to download.")
            elif result.status in ("not_found", "blocked"):
                s["not_found"] += 1
            else:
                s["error"] += 1

            if result.had_jsonld:
                s["jsonld_hint"] += 1

            st.session_state["results"].append(result.data)

        # Live preview
        preview_df   = pd.DataFrame(st.session_state["results"][-20:])
        preview_cols = [c for c in fieldnames if c in preview_df.columns]
        table_box.dataframe(
            preview_df[preview_cols] if preview_cols else preview_df,
            use_container_width=True, height=280,
        )

        time.sleep(delay_between_skus)

    if not st.session_state["stop_flag"]:
        progress_bar.progress(1.0, text="✅ Complete!")
    else:
        progress_bar.progress(
            len(st.session_state["results"]) / max(total, 1),
            text=f"⏹ Stopped at {len(st.session_state['results'])}/{total} SKUs",
        )
    st.session_state["running"] = False
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
    d1, d2, _ = st.columns([1, 1, 2])
    all_rows  = st.session_state["results"]

    with d1:
        st.download_button("Download CSV",   data=to_csv_bytes(all_rows, fieldnames),
                            file_name="enriched_products.csv",   mime="text/csv")
    with d2:
        st.download_button("Download Excel", data=to_xlsx_bytes(all_rows, fieldnames),
                            file_name="enriched_products.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
